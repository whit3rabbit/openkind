//! MLX/Metal execution of the pinned kev serving checkpoint.
//!
//! The forward runs the shared dense-Qwen MLX backbone
//! ([`DenseQwenMlxModel`]) at the pinned Qwen3-0.6B-Base geometry: the
//! digest-verified BF16 base checkpoint streams into FP32 arrays with the
//! verified PEFT adapter merged host-side exactly like the CPU oracle's
//! merge (`base + (B @ A) · scale`). The pointer readout mirrors
//! `super::super::model` op for op on the returned host rows: the decide
//! row comes from the biased `q` linear, each option's `</opt>` boundary
//! row comes from the biased `k` linear, and the logits are FP64 dots
//! scaled by `1/sqrt(POINTER_DIM)`. Nothing is sampled and no continuation
//! state exists.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::families::kev::{LORA_RANK, LORA_SCALE, MAX_ROW_TOKENS, POINTER_DIM};
use crate::families::qwen35_survey_mlx::{read_safetensors_filtered, SurveyTensor};
use crate::families::qwen_dense_mlx::{DenseQwenConfig, DenseQwenLoadSpec, DenseQwenMlxModel};
use crate::families::support::FamilyError;
use crate::qwen35::mlx::MlxRuntime;

use crate::families::kev::model::VerifiedArtifacts;

/// Pinned backbone geometry (the base checkpoint's `config.json`, enforced
/// by the loader's contract check before any weights load).
const HIDDEN_SIZE: usize = 1024;

/// The pinned family model on the MLX backbone plus the host-side FP32
/// pointer head.
pub(super) struct KevMlxModel {
    backbone: DenseQwenMlxModel,
    /// Row-major `[POINTER_DIM, HIDDEN_SIZE]` float32 projection weights and
    /// their biases.
    head_q_weight: Vec<f32>,
    head_q_bias: Vec<f32>,
    head_k_weight: Vec<f32>,
    head_k_bias: Vec<f32>,
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

impl KevMlxModel {
    /// Load the digest-verified base checkpoint, merge the verified adapter,
    /// and build the FP32 MLX backbone plus the pointer head.
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
            // The CPU oracle builds its RoPE tables for the same frozen row
            // cap the renderer enforces.
            max_sequence_tokens: MAX_ROW_TOKENS,
        };
        let spec = DenseQwenLoadSpec {
            shards: std::slice::from_ref(&artifacts.base_checkpoint),
            embedding_shard: &artifacts.base_checkpoint,
            adapter: Some((&artifacts.adapter, LORA_RANK, LORA_SCALE)),
            config,
        };
        // The backbone loader and the head reader both allocate MLX arrays
        // through their host-side safetensors readers, so both run under the
        // process-wide execution lock (the lock is not reentrant).
        let (backbone, head_q_weight, head_q_bias, head_k_weight, head_k_bias) =
            runtime.execute(|| {
                let backbone = DenseQwenMlxModel::load(spec, runtime)?;
                let mut head = read_safetensors_filtered(&artifacts.head, |name| {
                    matches!(name, "q.weight" | "q.bias" | "k.weight" | "k.bias")
                })?;
                let head_q_weight =
                    head_values(&mut head, "q.weight", &[POINTER_DIM, HIDDEN_SIZE])?;
                let head_q_bias = head_values(&mut head, "q.bias", &[POINTER_DIM])?;
                let head_k_weight =
                    head_values(&mut head, "k.weight", &[POINTER_DIM, HIDDEN_SIZE])?;
                let head_k_bias = head_values(&mut head, "k.bias", &[POINTER_DIM])?;
                Ok::<_, FamilyError>((
                    backbone,
                    head_q_weight,
                    head_q_bias,
                    head_k_weight,
                    head_k_bias,
                ))
            })??;
        Ok(Self {
            backbone,
            head_q_weight,
            head_q_bias,
            head_k_weight,
            head_k_bias,
        })
    }

    /// Pointer logits for one chunk of question rows, mirroring the CPU
    /// oracle's `rows_logits` op for op.
    ///
    /// `rows` is the padded row-major id buffer, `row_lens` the real length
    /// of each row, and `readouts` the per-row `(decide, opts)` offsets.
    /// Every row is an independent causal sequence; each is prefilled as its
    /// own forward. Returns one logit vector per row in chunk order.
    pub(super) fn rows_logits(
        &self,
        rows: &[u32],
        row_lens: &[usize],
        readouts: &[(usize, Vec<usize>)],
    ) -> Result<Vec<Vec<f64>>, FamilyError> {
        if readouts.len() != row_lens.len() {
            return Err(FamilyError::InvalidInput(format!(
                "{} readouts for {} rows",
                readouts.len(),
                row_lens.len()
            )));
        }
        let seq_len = rows.len() / row_lens.len().max(1);
        // Every row forward takes the process-wide execution lock for the
        // whole chunk; no outer `runtime.execute` wraps this call (the lock
        // is not reentrant).
        let hidden_rows = self.backbone.runtime().execute(|| {
            let mut hidden_rows = Vec::with_capacity(row_lens.len());
            for (row_index, row_len) in row_lens.iter().enumerate() {
                let start = row_index * seq_len;
                let ids = rows.get(start..start + *row_len).ok_or_else(|| {
                    FamilyError::InvalidInput(format!(
                        "row {row_index} length {row_len} exceeds the padded id buffer"
                    ))
                })?;
                let hidden = self.backbone.forward_hidden(ids)?;
                if hidden.len() != ids.len() * HIDDEN_SIZE {
                    return Err(FamilyError::Numerical(format!(
                        "backbone output width does not match the pinned geometry: {} values \
                         for {} tokens x {} wide",
                        hidden.len(),
                        ids.len(),
                        HIDDEN_SIZE
                    )));
                }
                hidden_rows.push(hidden);
            }
            Ok(hidden_rows)
        })??;

        let scale = 1.0 / (POINTER_DIM as f64).sqrt();
        let mut out = Vec::with_capacity(readouts.len());
        for (row_index, (decide, opts)) in readouts.iter().enumerate() {
            let hidden = &hidden_rows[row_index];
            let row_len = row_lens[row_index];
            let slot = |position: usize| -> Result<&[f32], FamilyError> {
                if position >= row_len {
                    return Err(FamilyError::InvalidInput(format!(
                        "readout position {position} outside row {row_index} of length \
                         {row_len}"
                    )));
                }
                Ok(&hidden[position * HIDDEN_SIZE..(position + 1) * HIDDEN_SIZE])
            };
            let query = biased_matvec(&self.head_q_weight, &self.head_q_bias, slot(*decide)?);
            let logits: Vec<f64> = opts
                .iter()
                .map(|position| {
                    let key =
                        biased_matvec(&self.head_k_weight, &self.head_k_bias, slot(*position)?);
                    Ok(key
                        .iter()
                        .zip(&query)
                        .map(|(key_value, query_value)| {
                            f64::from(*key_value) * f64::from(*query_value)
                        })
                        .sum::<f64>()
                        * scale)
                })
                .collect::<Result<Vec<f64>, FamilyError>>()?;
            if logits.iter().any(|logit| !logit.is_finite()) {
                return Err(FamilyError::Numerical(
                    "pointer readout produced a non-finite logit".to_owned(),
                ));
            }
            out.push(logits);
        }
        Ok(out)
    }
}
