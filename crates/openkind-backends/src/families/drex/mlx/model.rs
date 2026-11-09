//! MLX/Metal execution of the pinned Drex v1.5 checkpoint.
//!
//! The forward runs the shared geometry-parameterized causal survey model
//! ([`SurveyMlxModel`]) at the 9B hybrid geometry: the digest-verified BF16
//! shards widen to FP32 MLX arrays exactly (the same values the CPU oracle's
//! BF16-weight FP32-compute path reads). The Kev pointer readout mirrors
//! `super::super::model` op for op on the returned host rows: the
//! `<|fim_suffix|>` decide state is the query, each option's `<|box_end|`
//! state a key, both through the biased FP32 `q`/`k` linears of `head.pt`,
//! and the logits are FP64 dots scaled by `1/sqrt(POINTER_DIM)`.

use std::sync::Arc;

use crate::families::kev::model::QuestionRow;
use crate::families::qwen35_survey_mlx::{SurveyLoadSpec, SurveyMlxModel};
use crate::families::support::FamilyError;
use crate::qwen35::mlx::MlxRuntime;

use super::super::model::{VerifiedArtifacts, HEAD_STATE_KEY};
use super::super::{GEOMETRY, POINTER_DIM, VOCAB_SIZE};

/// The pinned family model on the MLX survey backbone plus the host-side
/// FP32 pointer head.
pub(super) struct DrexMlxModel {
    backbone: SurveyMlxModel,
    /// Row-major `[POINTER_DIM, hidden]` float32 projection weights and
    /// their biases, read from the verified `head.pt` pickle.
    head_q_weight: Vec<f32>,
    head_q_bias: Vec<f32>,
    head_k_weight: Vec<f32>,
    head_k_bias: Vec<f32>,
}

/// The `hidden` width the head projections contract (the 9B geometry's
/// residual width).
fn hidden_size() -> usize {
    GEOMETRY.hidden_size
}

/// Read one flattened FP32 tensor of the expected shape from the verified
/// pickle (host-side, through candle's opcode interpreter — no execution).
fn head_vector(
    tensors: &candle_core::pickle::PthTensors,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, FamilyError> {
    let tensor = tensors
        .get(name)
        .map_err(|error| FamilyError::InvalidInput(format!("head tensor `{name}`: {error}")))?;
    let tensor =
        tensor.ok_or_else(|| FamilyError::InvalidInput(format!("head tensor `{name}` missing")))?;
    if tensor.dtype() != candle_core::DType::F32 || tensor.dims() != shape {
        return Err(FamilyError::ContractMismatch {
            field: "head.tensors",
            expected: format!("{name} f32 {shape:?}"),
            actual: format!("{:?} {:?}", tensor.dtype(), tensor.dims()),
        });
    }
    let flattened = tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| FamilyError::InvalidInput(format!("head tensor `{name}`: {error}")))?;
    Ok(flattened)
}

/// Row-major biased matvec: `[POINTER_DIM, hidden] @ row + bias` in FP32,
/// the arithmetic the CPU head's candle linear computes.
fn biased_matvec(weight: &[f32], bias: &[f32], row: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(hidden_size())
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

impl DrexMlxModel {
    /// Load the digest-verified shards into FP32 MLX arrays and read the
    /// pointer head from the verified pickle.
    pub(super) fn load(
        artifacts: &VerifiedArtifacts,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        let spec = SurveyLoadSpec {
            shards: &artifacts.shards,
            embedding_shard: &artifacts.embedding_shard,
            embedding_tensor: "model.language_model.embed_tokens.weight",
            vocab_size: VOCAB_SIZE,
            geometry: GEOMETRY,
            adapter: None,
        };
        // The survey loader reads tensors through its own host-side
        // safetensors reader, which must run under the process-wide
        // execution lock like every other MLX allocation.
        let backbone = runtime.execute(|| SurveyMlxModel::load(spec, runtime))??;
        // The digest was verified by the caller, so these are known bytes.
        // candle's reader interprets the pickle opcodes into plain values
        // and never imports or calls the globals the stream names.
        let tensors =
            candle_core::pickle::PthTensors::new(&artifacts.head, Some(HEAD_STATE_KEY))
                .map_err(|error| FamilyError::InvalidInput(format!("head pickle: {error}")))?;
        let hidden = hidden_size();
        let head_q_weight = head_vector(&tensors, "q.weight", &[POINTER_DIM, hidden])?;
        let head_q_bias = head_vector(&tensors, "q.bias", &[POINTER_DIM])?;
        let head_k_weight = head_vector(&tensors, "k.weight", &[POINTER_DIM, hidden])?;
        let head_k_bias = head_vector(&tensors, "k.bias", &[POINTER_DIM])?;
        Ok(Self {
            backbone,
            head_q_weight,
            head_q_bias,
            head_k_weight,
            head_k_bias,
        })
    }

    /// Pointer logits for one question row, mirroring the CPU oracle's
    /// `row_logits` op for op on the survey backbone's host rows.
    pub(super) fn row_logits(&self, row: &QuestionRow) -> Result<Vec<f64>, FamilyError> {
        let hidden_values = self
            .backbone
            .runtime()
            .execute(|| self.backbone.forward_hidden(&row.ids))??;
        let hidden = hidden_size();
        if hidden_values.len() != row.ids.len() * hidden {
            return Err(FamilyError::InvalidInput(format!(
                "backbone output width does not match the pinned geometry: {} values for \
                 {} tokens",
                hidden_values.len(),
                row.ids.len()
            )));
        }
        let slot = |position: usize| &hidden_values[position * hidden..(position + 1) * hidden];

        let query = biased_matvec(&self.head_q_weight, &self.head_q_bias, slot(row.decide));
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
