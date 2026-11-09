//! MLX/Metal execution backend for the pinned encoder-NLI profile (feature
//! `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, and wire mapping feed the shared
//! post-LN MLX encoder body ([`crate::families::bert_mlx`]) at the pinned
//! DistilBERT geometry in FP32, with the sequence-classification head
//! (`pre_classifier` + ReLU + `classifier`) applied host-side exactly like
//! the CPU readout's linear math. The candle CPU path remains the correctness
//! oracle; parity gates compare against the committed golden fixtures.
//!
//! Every MLX operation — model load and per-pair forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use openkind_core::{ModelInfo, SystemRequest, SystemResponse};

use crate::families::bert_mlx::{
    biased_matvec, cls_row, BertMlxConfig, BertMlxKeyScheme, BertMlxModel,
};
use crate::families::qwen35_survey_mlx::read_safetensors_filtered;
use crate::families::support::{
    BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator, FamilyLimits,
};
use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

use super::engine::{evaluate_with, EncoderNliPassSource};
use super::model::VerifiedArtifacts;
use super::renderer::EncoderNliRenderer;
use super::{EncoderNliError, BACKBONE_ID, PROFILE_ID};

/// Arithmetic/device identity of the encoder-NLI MLX execution path: the
/// pinned FP32 checkpoint computed on the Metal GPU through the shared
/// post-LN MLX encoder body.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-encoder-nli";

/// Backend id reported by the MLX engine serving the pinned profile.
const MLX_BACKEND_ID: &str = "encoder-nli/mlx-fp32";

/// The pinned family model on the shared post-LN MLX encoder body plus the
/// host-side FP32 sequence-classification head.
struct EncoderNliMlxModel {
    body: BertMlxModel,
    runtime: Arc<MlxRuntime>,
    /// Row-major `[dim, dim]` float32 `pre_classifier` weight and bias.
    pre_classifier_weight: Vec<f32>,
    pre_classifier_bias: Vec<f32>,
    /// Row-major `[labels, dim]` float32 `classifier` weight and bias.
    classifier_weight: Vec<f32>,
    classifier_bias: Vec<f32>,
}

/// The pinned DistilBERT geometry for the MLX body, read from the family's
/// pinned-config contract (the same values the artifact loader enforces
/// against the digest-verified `config.json`).
fn pinned_geometry() -> Result<BertMlxConfig, FamilyError> {
    let value = |name: &str| -> Result<usize, FamilyError> {
        let (_, value) = super::pinned_config()
            .into_iter()
            .find(|(field, _)| *field == name)
            .ok_or_else(|| {
                FamilyError::InvalidInput(format!("pinned config is missing `{name}`"))
            })?;
        usize::try_from(value.as_u64().ok_or_else(|| {
            FamilyError::InvalidInput(format!("pinned config `{name}` is not an integer"))
        })?)
        .map_err(|_| FamilyError::InvalidInput(format!("pinned config `{name}` is negative")))
    };
    Ok(BertMlxConfig {
        vocab_size: value("vocab_size")?,
        hidden_size: value("dim")?,
        num_attention_heads: value("n_heads")?,
        num_hidden_layers: value("n_layers")?,
        intermediate_size: value("hidden_dim")?,
        max_position_embeddings: value("max_position_embeddings")?,
        // The candle distilbert implementation pins this epsilon in code
        // (`layer_norm(config.dim, 1e-12, ...)`) rather than reading it from
        // the checkpoint config.
        layer_norm_eps: 1e-12,
        // DistilBERT embeddings carry no token-type table.
        token_type_vocab: None,
    })
}

impl EncoderNliMlxModel {
    /// Load the digest-verified checkpoint into FP32 MLX arrays and host
    /// head vectors. Must run inside [`MlxRuntime::execute`].
    fn load(artifacts: &VerifiedArtifacts, runtime: &Arc<MlxRuntime>) -> Result<Self, FamilyError> {
        let geometry = pinned_geometry()?;
        let scheme = BertMlxKeyScheme::DistilBert;
        let hidden = geometry.hidden_size;
        let labels = 3_usize;
        let tensors = read_safetensors_filtered(&artifacts.checkpoint, |name| {
            scheme.internal_name(name).is_some()
                || matches!(
                    name,
                    "pre_classifier.weight"
                        | "pre_classifier.bias"
                        | "classifier.weight"
                        | "classifier.bias"
                )
        })?;
        let head_matrix = |name: &str, rows: usize, cols: usize| -> Result<Vec<f32>, FamilyError> {
            let tensor = tensors.get(name).ok_or_else(|| {
                FamilyError::InvalidInput(format!("checkpoint is missing tensor `{name}`"))
            })?;
            if tensor.shape != vec![rows, cols] {
                return Err(FamilyError::InvalidInput(format!(
                        "tensor `{name}` has shape {:?} but the pinned contract requires [{rows}, {cols}]",
                        tensor.shape
                    )));
            }
            Ok(tensor.values.clone())
        };
        let head_vector = |name: &str, len: usize| -> Result<Vec<f32>, FamilyError> {
            let tensor = tensors.get(name).ok_or_else(|| {
                FamilyError::InvalidInput(format!("checkpoint is missing tensor `{name}`"))
            })?;
            if tensor.shape != vec![len] {
                return Err(FamilyError::InvalidInput(format!(
                    "tensor `{name}` has shape {:?} but the pinned contract requires [{len}]",
                    tensor.shape
                )));
            }
            Ok(tensor.values.clone())
        };
        let pre_classifier_weight = head_matrix("pre_classifier.weight", hidden, hidden)?;
        let pre_classifier_bias = head_vector("pre_classifier.bias", hidden)?;
        let classifier_weight = head_matrix("classifier.weight", labels, hidden)?;
        let classifier_bias = head_vector("classifier.bias", labels)?;

        let mut body = BTreeMap::new();
        for (name, tensor) in &tensors {
            if let Some(internal) = scheme.internal_name(name) {
                if body
                    .insert(internal.clone(), tensor.array.clone())
                    .is_some()
                {
                    return Err(FamilyError::InvalidInput(format!(
                        "checkpoint maps `{name}` onto `{internal}` twice"
                    )));
                }
            }
        }
        let body = BertMlxModel::from_tensor_map(&body, &geometry)?;
        Ok(Self {
            body,
            runtime: Arc::clone(runtime),
            pre_classifier_weight,
            pre_classifier_bias,
            classifier_weight,
            classifier_bias,
        })
    }

    /// Forward one premise–hypothesis pair and return the three-way NLI
    /// logits in checkpoint label order
    /// (`[ENTAILMENT, NEUTRAL, CONTRADICTION]`), mirroring the CPU readout:
    /// CLS row 0 → `pre_classifier` → ReLU → `classifier`, all host FP32
    /// after the MLX body.
    fn nli_logits(&self, pair_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        if pair_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "encoded pair is empty".to_owned(),
            ));
        }
        let hidden_size = self.body.hidden_size();
        let cls = self.runtime.execute(|| {
            self.body
                .forward(pair_ids)
                .and_then(|hidden| cls_row(&hidden, hidden_size))
        })??;
        let pooled: Vec<f32> = biased_matvec(
            &self.pre_classifier_weight,
            &self.pre_classifier_bias,
            hidden_size,
            &cls,
        )?
        .into_iter()
        .map(|value| value.max(0.0))
        .collect();
        let logits = biased_matvec(
            &self.classifier_weight,
            &self.classifier_bias,
            hidden_size,
            &pooled,
        )?;
        if logits.len() != 3 {
            return Err(FamilyError::ContractMismatch {
                field: "classifier.labels",
                expected: "3 NLI logits".to_owned(),
                actual: format!("{} logits", logits.len()),
            });
        }
        let logits = [
            f64::from(logits[super::LABEL_ENTAILMENT]),
            f64::from(logits[super::LABEL_NEUTRAL]),
            f64::from(logits[super::LABEL_CONTRADICTION]),
        ];
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(FamilyError::Numerical(
                "nli head produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }
}

/// Loaded pinned encoder-NLI engine on the MLX backend.
pub struct EncoderNliMlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: EncoderNliRenderer,
    model: EncoderNliMlxModel,
    backend_id: &'static str,
}

impl EncoderNliPassSource for Inner {
    fn renderer(&self) -> &EncoderNliRenderer {
        &self.renderer
    }

    fn nli_logits(&self, pair_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        self.model.nli_logits(pair_ids)
    }
}

/// Filesystem configuration for the pinned encoder-NLI profile on the MLX
/// backend.
#[derive(Debug, Clone)]
pub struct EncoderNliMlxEngineConfig {
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// and `vocab.txt` at the pinned revision. Artifacts are verified in
    /// place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl EncoderNliMlxEngineConfig {
    /// Fill admission defaults around the given model root.
    pub fn new(model_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 2,
                retry_after_ms: 1_000,
                evaluation_timeout: Some(Duration::from_secs(600)),
            },
        }
    }
}

impl EncoderNliMlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(config: EncoderNliMlxEngineConfig) -> Result<BoundedFamilyEngine, EncoderNliError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = EncoderNliRenderer::load(&artifacts.vocab)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        // The loader reads the checkpoint through its host-side safetensors
        // reader and allocates MLX arrays, so it holds the process-wide
        // execution lock itself.
        let model = runtime.execute(|| EncoderNliMlxModel::load(&artifacts, &runtime))??;
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                backend_id: MLX_BACKEND_ID,
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for EncoderNliMlxEngine {
    fn backend_id(&self) -> &str {
        self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} premise-hypothesis entailment engine ({PROFILE_ID}, {}).",
                self.inner.backend_id
            ),
            release_date: "2026-09-26".into(),
        }
    }

    fn reports_input_tokens(&self) -> bool {
        true
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        evaluate_with(self.inner.as_ref(), request, control)
    }
}
