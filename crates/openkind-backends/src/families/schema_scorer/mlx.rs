//! MLX/Metal execution backend for the pinned schema-scorer profile (feature
//! `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, and wire mapping feed the shared
//! post-LN MLX encoder body ([`crate::families::bert_mlx`]) at the pinned
//! BERT MiniLM geometry in FP32, with the cross-encoder head (tanh pooler +
//! scalar classifier) applied host-side exactly like the CPU readout's linear
//! math. The candle CPU path remains the correctness oracle; parity gates
//! compare against the committed golden fixtures.
//!
//! Every MLX operation — model load and per-pair forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use candle_transformers::models::bert::HiddenAct;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};

use crate::families::bert_mlx::{
    biased_matvec, cls_row, BertMlxConfig, BertMlxKeyScheme, BertMlxModel,
};
use crate::families::qwen35_survey_mlx::read_safetensors_filtered;
use crate::families::support::{
    BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator, FamilyLimits,
};
use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

use super::engine::{evaluate_with, SchemaScorerPassSource};
use super::model::VerifiedArtifacts;
use super::renderer::SchemaScorerRenderer;
use super::{SchemaScorerError, BACKBONE_ID, PROFILE_ID};

/// Arithmetic/device identity of the schema-scorer MLX execution path: the
/// pinned FP32 checkpoint computed on the Metal GPU through the shared
/// post-LN MLX encoder body.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-schema-scorer";

/// Backend id reported by the MLX engine serving the pinned profile.
const MLX_BACKEND_ID: &str = "schema-scorer/mlx-fp32";

/// The pinned family model on the shared post-LN MLX encoder body plus the
/// host-side FP32 cross-encoder head.
struct SchemaScorerMlxModel {
    body: BertMlxModel,
    runtime: Arc<MlxRuntime>,
    /// Row-major `[hidden, hidden]` float32 pooler weight and bias.
    pooler_weight: Vec<f32>,
    pooler_bias: Vec<f32>,
    /// Row-major `[1, hidden]` float32 classifier weight and bias.
    classifier_weight: Vec<f32>,
    classifier_bias: Vec<f32>,
}

impl SchemaScorerMlxModel {
    /// Load the digest-verified checkpoint into FP32 MLX arrays and host
    /// head vectors. Must run inside [`MlxRuntime::execute`].
    fn load(artifacts: &VerifiedArtifacts, runtime: &Arc<MlxRuntime>) -> Result<Self, FamilyError> {
        // The candle config was decoded from the digest-verified, pinned
        // `config.json`, so its geometry doubles as the MLX body contract.
        let pinned = &artifacts.config;
        if pinned.hidden_act != HiddenAct::Gelu {
            return Err(FamilyError::InvalidInput(format!(
                "MLX execution implements the exact-erf GELU only, the pinned checkpoint pins {:?}",
                pinned.hidden_act
            )));
        }
        let geometry = BertMlxConfig {
            vocab_size: pinned.vocab_size,
            hidden_size: pinned.hidden_size,
            num_attention_heads: pinned.num_attention_heads,
            num_hidden_layers: pinned.num_hidden_layers,
            intermediate_size: pinned.intermediate_size,
            max_position_embeddings: pinned.max_position_embeddings,
            layer_norm_eps: pinned.layer_norm_eps,
            // The CPU forward passes all-zero token-type ids, which reads
            // exactly the row-0 token-type embedding.
            token_type_vocab: Some(pinned.type_vocab_size),
        };
        let scheme = BertMlxKeyScheme::Bert;
        let hidden = geometry.hidden_size;
        let tensors = read_safetensors_filtered(&artifacts.checkpoint, |name| {
            scheme.internal_name(name).is_some()
                || matches!(
                    name,
                    "bert.pooler.dense.weight"
                        | "bert.pooler.dense.bias"
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
        let pooler_weight = head_matrix("bert.pooler.dense.weight", hidden, hidden)?;
        let pooler_bias = head_vector("bert.pooler.dense.bias", hidden)?;
        let classifier_weight = head_matrix("classifier.weight", 1, hidden)?;
        let classifier_bias = head_vector("classifier.bias", 1)?;

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
            pooler_weight,
            pooler_bias,
            classifier_weight,
            classifier_bias,
        })
    }

    /// Forward one query–passage pair and return the scalar relevance logit,
    /// mirroring the CPU readout: CLS row 0 → pooler → tanh → classifier,
    /// all host FP32 after the MLX body.
    fn relevance_logit(&self, pair_ids: &[u32]) -> Result<f64, FamilyError> {
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
        let pooled: Vec<f32> =
            biased_matvec(&self.pooler_weight, &self.pooler_bias, hidden_size, &cls)?
                .into_iter()
                .map(|value| value.tanh())
                .collect();
        let logits = biased_matvec(
            &self.classifier_weight,
            &self.classifier_bias,
            hidden_size,
            &pooled,
        )?;
        if logits.len() != 1 {
            return Err(FamilyError::ContractMismatch {
                field: "classifier.outputs",
                expected: "1 relevance logit".to_owned(),
                actual: format!("{} logits", logits.len()),
            });
        }
        let logit = f64::from(logits[0]);
        if !logit.is_finite() {
            return Err(FamilyError::Numerical(
                "relevance head produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logit)
    }
}

/// Loaded pinned schema-scorer engine on the MLX backend.
pub struct SchemaScorerMlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: SchemaScorerRenderer,
    model: SchemaScorerMlxModel,
    backend_id: &'static str,
}

impl SchemaScorerPassSource for Inner {
    fn renderer(&self) -> &SchemaScorerRenderer {
        &self.renderer
    }

    fn relevance_logit(&self, pair_ids: &[u32]) -> Result<f64, FamilyError> {
        self.model.relevance_logit(pair_ids)
    }
}

/// Filesystem configuration for the pinned schema-scorer profile on the MLX
/// backend.
#[derive(Debug, Clone)]
pub struct SchemaScorerMlxEngineConfig {
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// and `tokenizer.json` at the pinned revision. Artifacts are verified
    /// in place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl SchemaScorerMlxEngineConfig {
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

impl SchemaScorerMlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(
        config: SchemaScorerMlxEngineConfig,
    ) -> Result<BoundedFamilyEngine, SchemaScorerError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = SchemaScorerRenderer::load(&artifacts.tokenizer)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        // The loader reads the checkpoint through its host-side safetensors
        // reader and allocates MLX arrays, so it holds the process-wide
        // execution lock itself.
        let model = runtime.execute(|| SchemaScorerMlxModel::load(&artifacts, &runtime))??;
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

impl FamilyEvaluator for SchemaScorerMlxEngine {
    fn backend_id(&self) -> &str {
        self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} scalar cross-encoder schema scorer ({PROFILE_ID}, {}).",
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
