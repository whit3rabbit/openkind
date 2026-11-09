//! MLX/Metal execution of the pinned Qwen3Guard-Stream checkpoint (feature
//! `mlx`, macOS arm64).
//!
//! The forward reuses the shared dense-Qwen MLX backbone
//! ([`DenseQwenMlxModel`]) at the pinned 0.6B geometry: the digest-verified
//! BF16 checkpoint streams into FP32 arrays and prefills the rendered query
//! prompt as one causal sequence. The Stream head mirrors `super::model` op
//! for op on the returned host rows: the final prompt position's FP32
//! hidden state runs through the host-side no-bias pre projection, the
//! plain RMSNorm, and the no-bias risk head — the same FP32 host math the
//! candle CPU path computes — and yields the three risk-level logits
//! (`Safe` / `Unsafe` / `Controversial`). No verdict string is ever
//! generated.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! the process-wide [`MlxRuntime`] execution lock, per the workspace MLX
//! discipline.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::families::qwen35_survey_mlx::{read_safetensors_filtered, SurveyTensor};
use crate::families::qwen_dense_mlx::{DenseQwenConfig, DenseQwenLoadSpec, DenseQwenMlxModel};
use crate::families::support::FamilyError;
use crate::qwen35::mlx::MlxRuntime;

use super::engine::RiskLogits;
use super::model::VerifiedArtifacts;
use super::{CLASS_CONTROVERSIAL, CLASS_SAFE, CLASS_UNSAFE, MAX_SEQUENCE_TOKENS};

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// single-file BF16 checkpoint widened to FP32 arrays computed on the Metal
/// GPU through the shared dense Qwen MLX backbone.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-qwen3guard";

/// Backend id reported by the MLX engine serving the pinned profile.
pub const MLX_BACKEND_ID: &str = "qwen3guard/mlx-fp32";

/// Pinned backbone geometry (`config.json`, enforced by the loader's
/// contract check before any weights load).
const HIDDEN_SIZE: usize = 1024;
/// Pinned guard inner width (`config.json`: `guard_inner_size`).
const GUARD_INNER_SIZE: usize = 512;
/// Risk-level class count (`config.json`: `num_risk_level`).
const CLASS_COUNT: usize = 3;
/// RMS epsilon of the guard head's `query_risk_level_category_layernorm`.
const HEAD_RMS_EPSILON: f64 = 1e-6;

/// The checkpoint's root-level Stream head tensors, read host-side.
const HEAD_PRE: &str = "query_risk_level_category_pre.weight";
const HEAD_PRE_NORM: &str = "query_risk_level_category_layernorm.weight";
const HEAD_RISK: &str = "query_risk_level_head.weight";

/// The pinned guard model on the MLX backbone plus the host-side FP32 head.
pub(super) struct Qwen3GuardMlxModel {
    backbone: DenseQwenMlxModel,
    /// Row-major `[GUARD_INNER_SIZE, HIDDEN_SIZE]` no-bias projection.
    pre_weight: Vec<f32>,
    /// RMSNorm weight of the guard inner width.
    pre_norm_weight: Vec<f32>,
    /// Row-major `[CLASS_COUNT, GUARD_INNER_SIZE]` no-bias projection.
    risk_weight: Vec<f32>,
}

/// Read one host FP32 tensor of the expected shape out of the head map,
/// failing closed on a missing tensor or a drifted shape.
fn head_values(
    tensors: &mut BTreeMap<String, SurveyTensor>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, FamilyError> {
    let entry = tensors.remove(name).ok_or_else(|| {
        FamilyError::InvalidInput(format!("head tensor `{name}` missing from checkpoint"))
    })?;
    if entry.shape.as_slice() != shape {
        return Err(FamilyError::InvalidInput(format!(
            "head tensor `{name}` has shape {:?}, expected {shape:?}",
            entry.shape
        )));
    }
    Ok(entry.values)
}

/// Row-major no-bias matvec: `[out, in] @ row` in FP32, the arithmetic the
/// CPU head's candle `linear_no_bias` computes.
fn matvec_no_bias(weight: &[f32], row: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(row.len())
        .map(|weights| weights.iter().zip(row).map(|(w, v)| w * v).sum::<f32>())
        .collect()
}

/// Plain RMSNorm over one row (`x · w · rsqrt(mean(x²) + eps)`), matching
/// the CPU head's candle `RmsNorm` FP32 math.
fn rms_norm_row(row: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
    let scale = 1.0 / (mean_square + eps as f32).sqrt();
    row.iter()
        .zip(weight)
        .map(|(value, norm)| value * scale * norm)
        .collect()
}

impl Qwen3GuardMlxModel {
    /// Load the digest-verified checkpoint into FP32 MLX arrays plus the
    /// host-side head vectors.
    pub(super) fn load(
        artifacts: &VerifiedArtifacts,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        let config = DenseQwenConfig {
            vocab_size: 151_936,
            hidden_size: HIDDEN_SIZE,
            head_dim: 128,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            num_hidden_layers: 28,
            intermediate_size: 3072,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            qk_norm: true,
            max_sequence_tokens: MAX_SEQUENCE_TOKENS,
        };
        let spec = DenseQwenLoadSpec {
            shards: std::slice::from_ref(&artifacts.checkpoint),
            embedding_shard: &artifacts.checkpoint,
            adapter: None,
            config,
        };
        // The backbone loader and the head reader both allocate MLX arrays
        // through their host-side safetensors readers, so both run under the
        // process-wide execution lock (the lock is not reentrant).
        let (backbone, pre_weight, pre_norm_weight, risk_weight) = runtime.execute(|| {
            let backbone = DenseQwenMlxModel::load(spec, runtime)?;
            let mut head = read_safetensors_filtered(&artifacts.checkpoint, |name| {
                matches!(name, HEAD_PRE | HEAD_PRE_NORM | HEAD_RISK)
            })?;
            let pre_weight = head_values(&mut head, HEAD_PRE, &[GUARD_INNER_SIZE, HIDDEN_SIZE])?;
            let pre_norm_weight = head_values(&mut head, HEAD_PRE_NORM, &[GUARD_INNER_SIZE])?;
            let risk_weight = head_values(&mut head, HEAD_RISK, &[CLASS_COUNT, GUARD_INNER_SIZE])?;
            Ok::<_, FamilyError>((backbone, pre_weight, pre_norm_weight, risk_weight))
        })??;
        Ok(Self {
            backbone,
            pre_weight,
            pre_norm_weight,
            risk_weight,
        })
    }

    /// Query-side risk-level logits `[Safe, Unsafe, Controversial]` at the
    /// final prompt position, mirroring the CPU oracle's head math.
    fn risk_logits(&self, prompt_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        if prompt_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "prompt token ids are empty".to_owned(),
            ));
        }
        let hidden = self
            .backbone
            .runtime()
            .execute(|| self.backbone.forward_hidden(prompt_ids))??;
        if hidden.len() != prompt_ids.len() * HIDDEN_SIZE {
            return Err(FamilyError::Numerical(format!(
                "backbone output width does not match the pinned geometry: {} values for \
                 {} tokens x {} wide",
                hidden.len(),
                prompt_ids.len(),
                HIDDEN_SIZE
            )));
        }
        let start = (prompt_ids.len() - 1) * HIDDEN_SIZE;
        let last = &hidden[start..];
        let pre = matvec_no_bias(&self.pre_weight, last);
        let normed = rms_norm_row(&pre, &self.pre_norm_weight, HEAD_RMS_EPSILON);
        let logits = matvec_no_bias(&self.risk_weight, &normed);
        if logits.len() != CLASS_COUNT {
            return Err(FamilyError::ContractMismatch {
                field: "risk_head.classes",
                expected: "3 risk-level logits".to_owned(),
                actual: format!("{} logits", logits.len()),
            });
        }
        let logits = [
            f64::from(logits[CLASS_SAFE]),
            f64::from(logits[CLASS_UNSAFE]),
            f64::from(logits[CLASS_CONTROVERSIAL]),
        ];
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(FamilyError::Numerical(
                "risk head produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }
}

impl RiskLogits for Qwen3GuardMlxModel {
    fn risk_logits(&self, prompt_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        self.risk_logits(prompt_ids)
    }
}
