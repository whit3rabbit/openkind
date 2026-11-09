//! MLX/Metal execution of the pinned strands-decider serving checkpoint.
//!
//! The forward runs the shared geometry-parameterized causal survey model
//! ([`SurveyMlxModel`]) at the Qwen3.5-2B geometry: the digest-verified base
//! checkpoint streams into FP32 MLX arrays with the pinned PEFT adapter
//! merged host-side exactly like the CPU oracle's merge. The pointer readout
//! mirrors `super::super::model` op for op on the returned host rows: the
//! `<answer>` row is LayerNormed (biased, torch-default epsilon), the query
//! comes from the biased `q` linear, each option's last-token row is
//! LayerNormed and keyed through the biased `k` linear, and the logits are
//! FP64 dots scaled by `1/sqrt(POINTER_DIM)`.

use std::sync::Arc;

use candle_core::DType;

use crate::families::qwen35_survey_mlx::{SurveyLoadSpec, SurveyMlxModel};
use crate::families::strands_decider::model::VerifiedArtifacts;
use crate::families::strands_decider::renderer::RenderedRow;
use crate::families::strands_decider::{
    HIDDEN_SIZE, LAYER_NORM_EPSILON, LORA_RANK, LORA_SCALE, POINTER_DIM, VOCAB_SIZE,
};
use crate::families::support::FamilyError;
use crate::qwen35::mlx::{MlxRuntime, MlxSurveyLora};
use crate::qwen35::Qwen35Geometry;

/// The pinned family model on the MLX survey backbone plus the host-side
/// FP32 pointer head.
pub(super) struct StrandsMlxModel {
    backbone: SurveyMlxModel,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    /// Row-major `[POINTER_DIM, HIDDEN_SIZE]` float32 projection weights and
    /// their biases.
    head_q_weight: Vec<f32>,
    head_q_bias: Vec<f32>,
    head_k_weight: Vec<f32>,
    head_k_bias: Vec<f32>,
}

/// Read one flattened FP32 tensor of the expected length from a verified
/// safetensors head file (host-side, through candle's mmap).
fn head_vector(path: &std::path::Path, name: &str, len: usize) -> Result<Vec<f32>, FamilyError> {
    // SAFETY: the family verified the immutable size and SHA-256 of the
    // read-only head file before this mapper opens it.
    let variables = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&path.to_path_buf()),
            DType::F32,
            &candle_core::Device::Cpu,
        )
    }
    .map_err(|error| FamilyError::InvalidInput(format!("head mmap failed: {error}")))?;
    let tensor = variables
        .get(len, name)
        .map_err(|error| FamilyError::InvalidInput(format!("head tensor `{name}`: {error}")))?;
    let flattened = tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| FamilyError::InvalidInput(format!("head tensor `{name}`: {error}")))?;
    Ok(flattened)
}

/// Apply the head's LayerNorm to one hidden row (torch `nn.LayerNorm`:
/// biased variance, torch-default epsilon) — identical to the CPU path's
/// host math.
fn layer_norm(row: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let width = row.len();
    let mean = row.iter().sum::<f32>() / width as f32;
    let variance = row
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / width as f32;
    let denominator = (variance + LAYER_NORM_EPSILON).sqrt();
    row.iter()
        .enumerate()
        .map(|(index, value)| (value - mean) / denominator * weight[index] + bias[index])
        .collect()
}

/// Row-major biased matvec: `[POINTER_DIM, HIDDEN] @ row + bias` in FP32,
/// the arithmetic the CPU head's candle linear computes.
fn biased_matvec(weight: &[f32], bias: &[f32], row: &[f32]) -> Vec<f32> {
    weight
        .as_chunks::<HIDDEN_SIZE>()
        .0
        .iter()
        .zip(bias)
        .map(|(weights, bias)| {
            weights
                .iter()
                .zip(row)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                + bias
        })
        .collect()
}

impl StrandsMlxModel {
    /// Load the digest-verified base checkpoint, merge the verified adapter,
    /// and build the FP32 MLX backbone plus the pointer head.
    pub(super) fn load(
        artifacts: &VerifiedArtifacts,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        let adapter = MlxSurveyLora {
            adapter_path: artifacts.adapter.clone(),
            rank: LORA_RANK,
            scale: LORA_SCALE,
        };
        let spec = SurveyLoadSpec {
            shards: std::slice::from_ref(&artifacts.base_checkpoint),
            embedding_shard: &artifacts.base_checkpoint,
            embedding_tensor: "model.language_model.embed_tokens.weight",
            vocab_size: VOCAB_SIZE,
            geometry: Qwen35Geometry::BASE_2B,
            adapter: Some(&adapter),
        };
        // The survey loader reads tensors through its own host-side safetensors
        // reader, which must run under the process-wide execution lock like
        // every other MLX allocation.
        let backbone = runtime.execute(|| SurveyMlxModel::load(spec, runtime))??;
        let norm_weight = head_vector(&artifacts.head, "norm.weight", HIDDEN_SIZE)?;
        let norm_bias = head_vector(&artifacts.head, "norm.bias", HIDDEN_SIZE)?;
        let head_q_weight = head_vector(&artifacts.head, "q.weight", POINTER_DIM * HIDDEN_SIZE)?;
        let head_q_bias = head_vector(&artifacts.head, "q.bias", POINTER_DIM)?;
        let head_k_weight = head_vector(&artifacts.head, "k.weight", POINTER_DIM * HIDDEN_SIZE)?;
        let head_k_bias = head_vector(&artifacts.head, "k.bias", POINTER_DIM)?;
        Ok(Self {
            backbone,
            norm_weight,
            norm_bias,
            head_q_weight,
            head_q_bias,
            head_k_weight,
            head_k_bias,
        })
    }

    /// Pointer logits for one question row, mirroring the CPU oracle's
    /// `rows_logits` op for op on the survey backbone's host rows.
    pub(super) fn rows_logits(&self, row: &RenderedRow) -> Result<Vec<f64>, FamilyError> {
        let hidden = self
            .backbone
            .runtime()
            .execute(|| self.backbone.forward_hidden(&row.ids))??;
        if hidden.len() != row.ids.len() * HIDDEN_SIZE {
            return Err(FamilyError::InvalidInput(format!(
                "backbone output width does not match the pinned geometry: {} values for \
                 {} tokens",
                hidden.len(),
                row.ids.len()
            )));
        }
        let slot = |position: usize| &hidden[position * HIDDEN_SIZE..(position + 1) * HIDDEN_SIZE];

        let pooled = layer_norm(slot(row.answer), &self.norm_weight, &self.norm_bias);
        let query = biased_matvec(&self.head_q_weight, &self.head_q_bias, &pooled);

        let scale = 1.0 / (POINTER_DIM as f64).sqrt();
        let logits: Vec<f64> = row
            .opts
            .iter()
            .map(|position| {
                let key = biased_matvec(&self.head_k_weight, &self.head_k_bias, slot(*position));
                key.iter()
                    .zip(&query)
                    .map(|(key_value, query_value)| f64::from(*key_value) * f64::from(*query_value))
                    .sum::<f64>()
                    * scale
            })
            .collect();
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(FamilyError::Numerical(
                "pointer readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }
}
