//! MLX Qwen3.5 backbone: prefill, continuation, and per-layer states.
//!
//! Executes the same 32-layer graph as the Candle CPU oracle over MLX
//! arrays through [`MlxDecoderLayer`](super::layers::MlxDecoderLayer). The
//! tied embedding table stays host-resident (verified seek-based reads, one
//! row per token, widened exactly), so MLX memory holds only decoder weights
//! and per-request state. Continuation states are fully materialized
//! (evaluated) at this executor boundary.

use mlx_rs::fast;
use mlx_rs::Array;
use opendecision_runtime::branch::{StateIdentity, StateLineage};

use crate::qwen35::{Qwen35Embedding, PROFILE_ID, STATE_FIRST_RENDERER_ID};

use super::layers::{MlxDecoderLayer, MlxLayerState, HIDDEN_SIZE};
use super::runtime::SharedMlxRuntime;
use super::weights::{MlxCheckpointFormat, MlxWeightLoadReport, MlxWeightStore};
use super::{MlxError, MlxPrecision};

/// Full backbone output materialized to host FP32 at the executor boundary.
#[derive(Debug, Clone)]
pub struct MlxBackboneOutput {
    /// Number of tokens in the executed suffix.
    pub token_count: usize,
    /// Host-read embedding row of the final token (pre-layer).
    pub embedding_last_token: Vec<f32>,
    /// Final-token hidden vector after each of the 32 layers, in order.
    pub layer_last_tokens: Vec<Vec<f32>>,
    /// Final RMSNorm output for every token, row-major `[token_count, 2560]`.
    pub final_values: Vec<f32>,
}

impl MlxBackboneOutput {
    /// Final-token feature vector consumed by the score-summary head.
    #[must_use]
    pub fn feature(&self) -> &[f32] {
        &self.final_values[(self.token_count - 1) * HIDDEN_SIZE..]
    }
}

/// Complete hybrid continuation state (attention KV, DeltaNet recurrent,
/// convolution, position) as MLX arrays.
pub struct MlxBackboneState {
    /// Owning runtime: array evaluation is only legal under its execution
    /// lock, which is what makes this state `Send + Sync` sound.
    pub(crate) runtime: SharedMlxRuntime,
    /// Pinned profile/model/tokenizer/arithmetic identity. Continuation is
    /// rejected unless this matches the executor's identity exactly.
    pub identity: StateIdentity,
    /// Process-local fork lineage (fresh root per prefill).
    pub lineage: StateLineage,
    /// Absolute next-token position.
    pub position: usize,
    /// 32 per-layer states in model order.
    pub(crate) layers: Vec<MlxLayerState>,
}

impl Clone for MlxBackboneState {
    fn clone(&self) -> Self {
        self.runtime
            .execute(|| Self {
                runtime: std::sync::Arc::clone(&self.runtime),
                identity: self.identity.clone(),
                lineage: self.lineage,
                position: self.position,
                layers: self.layers.clone(),
            })
            .unwrap_or_else(|error| panic!("failed to clone MLX continuation state: {error}"))
    }
}

impl MlxBackboneState {
    /// Logical tensor payload bytes (attention KV + recurrent + conv), the
    /// same quantity the runtime's `TensorStorageBreakdown` tracks. MLX
    /// allocator memory (active/peak/cache) is reported separately.
    #[must_use]
    pub fn tensor_storage_bytes(&self) -> usize {
        self.tensor_storage_breakdown().tensor_storage_bytes()
    }
}

/// MLX-backed Qwen3.5 backbone for the pinned profile.
pub struct MlxQwen35Backbone {
    runtime: SharedMlxRuntime,
    layers: Vec<MlxDecoderLayer>,
    final_norm: Array,
    embedding: Qwen35Embedding,
    precision: MlxPrecision,
    identity: StateIdentity,
    checkpoint_format: MlxCheckpointFormat,
    /// Memory evidence captured while the weights streamed in.
    pub load_report: MlxWeightLoadReport,
}

// SAFETY: every array evaluation in this backbone's methods runs under the
// process-wide `MlxRuntime` execution mutex (see the `branch_state` module
// docs); outside the lock, array handles are immutable refcounted values whose
// handle-only operations (clone, shape, dtype) are safe concurrently. This is
// the same soundness argument as `unsafe impl Sync for MlxBackboneState`, and
// it is what lets the engine share one loaded backbone across worker threads.
unsafe impl Sync for MlxQwen35Backbone {}

impl MlxQwen35Backbone {
    /// Verify the pinned checkpoint and load it into MLX arrays at the
    /// requested precision.
    pub fn load(
        checkpoint_root: impl AsRef<std::path::Path>,
        runtime: SharedMlxRuntime,
        precision: MlxPrecision,
    ) -> Result<Self, MlxError> {
        if precision == MlxPrecision::NativeBf16 && !runtime.bf16_qualified() {
            return Err(MlxError::InvalidState(
                "native BF16 model load requires runtime.qualify_bf16() to pass first".to_owned(),
            ));
        }
        let store = MlxWeightStore::load(checkpoint_root, &runtime, precision)?;
        let load_report = store.load_report;
        let checkpoint_identity = store.checkpoint_identity();
        let (tensors, embedding) = store.into_tensors();

        let identity = StateIdentity::new(
            PROFILE_ID,
            checkpoint_identity.backbone_id,
            checkpoint_identity.backbone_revision,
            STATE_FIRST_RENDERER_ID,
            checkpoint_identity.tokenizer_digest,
            arithmetic_id(precision, &runtime, checkpoint_identity.format),
        )
        .map_err(|error| MlxError::InvalidState(error.to_string()))?;

        let (layers, final_norm) = runtime.execute(|| -> Result<_, MlxError> {
            let mut layers = Vec::with_capacity(32);
            for layer_index in 0..32 {
                layers.push(MlxDecoderLayer::load(&tensors, layer_index, precision)?);
            }
            let final_norm = folded_final_norm(&tensors, precision)?;
            // Weight arrays must leave the loading thread fully materialized:
            // lazy graphs record this thread's stream, which worker threads
            // (engine blocking-pool forwards) cannot resolve.
            for layer in &layers {
                layer.materialize()?;
            }
            final_norm.eval().map_err(|error| MlxError::Operation {
                operation: "final norm materialization",
                message: error.to_string(),
            })?;
            Ok((layers, final_norm))
        })??;
        Ok(Self {
            runtime,
            layers,
            final_norm,
            embedding,
            precision,
            identity,
            checkpoint_format: checkpoint_identity.format,
            load_report,
        })
    }

    /// Arithmetic identity of this backbone's execution path.
    #[must_use]
    pub fn arithmetic_id(&self) -> &str {
        self.identity.arithmetic_id()
    }

    /// Pinned state identity used by continuation states.
    #[must_use]
    pub fn identity(&self) -> &StateIdentity {
        &self.identity
    }

    /// Verified checkpoint layout used by this executor.
    #[must_use]
    pub const fn checkpoint_format(&self) -> MlxCheckpointFormat {
        self.checkpoint_format
    }

    /// Expected logical continuation-state bytes for a position count.
    ///
    /// This is an architecture-derived contract, independent of the current
    /// cache handles. Keeping it separate lets parity detect missing or
    /// over-sized state tensors instead of merely reporting the size it was
    /// handed.
    #[must_use]
    pub fn expected_tensor_storage_bytes(&self, position: usize) -> usize {
        let element_bytes = match self.precision {
            MlxPrecision::Fp32 => 4,
            MlxPrecision::NativeBf16 => 2,
        };
        let linear_layers = self.layers.len() - self.layers.len() / 4;
        let full_layers = self.layers.len() / 4;
        let linear_elements = linear_layers
            * (super::layers::QKV_SIZE * super::layers::CONV_KERNEL
                + super::layers::VALUE_HEADS * super::layers::HEAD_DIM * super::layers::HEAD_DIM);
        let full_elements = full_layers
            * position
            * 2
            * super::layers::KV_HEADS
            * super::layers::ATTENTION_HEAD_DIM;
        (linear_elements + full_elements) * element_bytes
    }

    /// Prefill a fresh root sequence.
    pub fn prefill(
        &self,
        input_ids: &[u32],
    ) -> Result<(MlxBackboneOutput, MlxBackboneState), MlxError> {
        self.execute(input_ids, None)
    }

    /// Read the final host embedding row without executing decoder layers.
    ///
    /// This keeps the embedding-only parity diagnostic bounded to tokenizer
    /// and embedding work. Weight loading remains complete because the
    /// verified loader must validate both checkpoint shards before use.
    pub fn embedding_last_token(&self, input_ids: &[u32]) -> Result<Vec<f32>, MlxError> {
        if input_ids.is_empty() {
            return Err(MlxError::InvalidState(
                "embedding input must contain at least one token".to_owned(),
            ));
        }
        self.embedding
            .embed(input_ids)
            .map(|embedding| embedding.last_token().to_vec())
            .map_err(MlxError::from_qwen)
    }

    /// Continue `state` with a suffix without mutating it.
    pub fn continue_from(
        &self,
        state: &MlxBackboneState,
        suffix_ids: &[u32],
    ) -> Result<(MlxBackboneOutput, MlxBackboneState), MlxError> {
        if state.identity != self.identity {
            return Err(MlxError::InvalidState(format!(
                "continuation state arithmetic identity `{}` does not match this executor `{}`",
                state.identity.arithmetic_id(),
                self.identity.arithmetic_id()
            )));
        }
        if state.layers.len() != self.layers.len() {
            return Err(MlxError::InvalidState(format!(
                "continuation state has {} layers, expected {}",
                state.layers.len(),
                self.layers.len()
            )));
        }
        if !std::sync::Arc::ptr_eq(&state.runtime, &self.runtime) {
            return Err(MlxError::InvalidState(
                "continuation state belongs to a different MLX runtime".to_owned(),
            ));
        }
        self.validate_state(state)?;
        self.execute(suffix_ids, Some(state))
    }

    fn validate_state(&self, state: &MlxBackboneState) -> Result<(), MlxError> {
        let expected_dtype = match self.precision {
            MlxPrecision::Fp32 => mlx_rs::Dtype::Float32,
            MlxPrecision::NativeBf16 => mlx_rs::Dtype::Bfloat16,
        };
        let position = i32::try_from(state.position).map_err(|_| {
            MlxError::InvalidState(
                "continuation position does not fit MLX shape dimensions".to_owned(),
            )
        })?;
        state.runtime.execute(|| {
            for (layer_index, layer) in state.layers.iter().enumerate() {
                match (layer_index % 4 == 3, layer) {
                    (false, MlxLayerState::Linear(layer)) => {
                        if layer.conv.shape() != [super::layers::QKV_SIZE as i32, 4]
                            || layer.recurrent.shape()
                                != [
                                    super::layers::VALUE_HEADS as i32,
                                    super::layers::HEAD_DIM as i32,
                                    super::layers::HEAD_DIM as i32,
                                ]
                            || layer.conv.dtype() != expected_dtype
                            || layer.recurrent.dtype() != expected_dtype
                        {
                            return Err(MlxError::InvalidState(format!(
                                "invalid linear cache shape or dtype at layer {layer_index}"
                            )));
                        }
                    }
                    (true, MlxLayerState::Full(layer)) => {
                        let expected_shape = [
                            position,
                            super::layers::KV_HEADS as i32,
                            super::layers::ATTENTION_HEAD_DIM as i32,
                        ];
                        if layer.keys.shape() != expected_shape
                            || layer.values.shape() != expected_shape
                            || layer.keys.dtype() != expected_dtype
                            || layer.values.dtype() != expected_dtype
                        {
                            return Err(MlxError::InvalidState(format!(
                                "invalid attention cache shape or dtype at layer {layer_index}"
                            )));
                        }
                    }
                    _ => {
                        return Err(MlxError::InvalidState(format!(
                            "cache kind does not match layer {layer_index}"
                        )));
                    }
                }
            }
            Ok(())
        })??;
        Ok(())
    }

    fn execute(
        &self,
        input_ids: &[u32],
        previous_state: Option<&MlxBackboneState>,
    ) -> Result<(MlxBackboneOutput, MlxBackboneState), MlxError> {
        if input_ids.is_empty() {
            return Err(MlxError::InvalidState(
                "execution input must contain at least one token".to_owned(),
            ));
        }
        let token_count = input_ids.len();
        let position_start = previous_state.map_or(0, |state| state.position);
        let lineage = previous_state.map_or_else(StateLineage::new_root, |state| state.lineage);

        let embedding = self
            .embedding
            .embed(input_ids)
            .map_err(MlxError::from_qwen)?;
        let embedding_last_token = embedding.last_token().to_vec();

        let (output, next_layers) =
            self.runtime.execute(
                || -> Result<(MlxBackboneOutput, Vec<MlxLayerState>), MlxError> {
                    let mut hidden = embed_array(embedding.values(), self.precision);
                    let mut layer_last_tokens = Vec::with_capacity(self.layers.len());
                    let mut next_layers = Vec::with_capacity(self.layers.len());
                    for (layer_index, layer) in self.layers.iter().enumerate() {
                        let previous = previous_state.map(|state| &state.layers[layer_index]);
                        let (next_hidden, next_state) =
                            layer.forward(&hidden, token_count, position_start, previous)?;
                        hidden = next_hidden;
                        layer_last_tokens.push(last_row_to_host(&hidden)?);
                        next_layers.push(next_state);
                    }
                    let final_array = fast::rms_norm(&hidden, Some(&self.final_norm), 1e-6)
                        .map_err(|error| MlxError::Operation {
                            operation: "final rms norm",
                            message: error.to_string(),
                        })?;
                    let output = MlxBackboneOutput {
                        token_count,
                        embedding_last_token: embedding_last_token.clone(),
                        layer_last_tokens,
                        final_values: array_to_host(&final_array)?,
                    };
                    materialize_state(&next_layers)?;
                    Ok((output, next_layers))
                },
            )??;

        Ok((
            output,
            MlxBackboneState {
                runtime: super::runtime::SharedMlxRuntime::clone(&self.runtime),
                identity: self.identity.clone(),
                lineage,
                position: position_start + token_count,
                layers: next_layers,
            },
        ))
    }
}

/// Arithmetic identity including precision, kernel family, and the linked
/// Xcode/Metal toolchain that affects native-BF16 behavior.
fn arithmetic_id(
    precision: MlxPrecision,
    runtime: &super::runtime::MlxRuntime,
    checkpoint_format: MlxCheckpointFormat,
) -> String {
    let base = match precision {
        MlxPrecision::Fp32 => super::MLX_ARITHMETIC_ID_FP32_REFERENCE,
        MlxPrecision::NativeBf16 => super::MLX_ARITHMETIC_ID_BF16_REFERENCE,
    };
    format!(
        "{base};checkpoint={};toolchain={}",
        checkpoint_format.as_str(),
        runtime.toolchain_identity()
    )
}

/// Build `[rows, 2560]` embedding array from host-widened rows.
fn embed_array(values: &[f32], precision: MlxPrecision) -> Array {
    match precision {
        MlxPrecision::Fp32 => Array::from_slice(
            values,
            &[(values.len() / HIDDEN_SIZE) as i32, HIDDEN_SIZE as i32],
        ),
        MlxPrecision::NativeBf16 => Array::from_slice(
            &values
                .iter()
                .map(|value| half::bf16::from_f32(*value))
                .collect::<Vec<_>>(),
            &[(values.len() / HIDDEN_SIZE) as i32, HIDDEN_SIZE as i32],
        ),
    }
}

/// Materialize one array to host FP32 (eval first).
fn array_to_host(array: &Array) -> Result<Vec<f32>, MlxError> {
    array.eval().map_err(|error| MlxError::Operation {
        operation: "boundary eval",
        message: error.to_string(),
    })?;
    let values = array
        .to_vec_cast::<f32>()
        .map_err(|error| MlxError::Operation {
            operation: "boundary read",
            message: error.to_string(),
        })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MlxError::Operation {
            operation: "boundary finite check",
            message: "MLX produced a non-finite value".to_owned(),
        });
    }
    Ok(values)
}

/// Host FP32 of the final row of a `[rows, width]` array.
fn last_row_to_host(array: &Array) -> Result<Vec<f32>, MlxError> {
    let rows = array.shape()[0] as usize;
    let all = array_to_host(array)?;
    let width = all.len() / rows;
    Ok(all[(rows - 1) * width..].to_vec())
}

/// Materialize every continuation tensor before it leaves the runtime lock.
fn materialize_state(layers: &[MlxLayerState]) -> Result<(), MlxError> {
    for layer in layers {
        let arrays: [&Array; 2] = match layer {
            MlxLayerState::Linear(state) => [&state.conv, &state.recurrent],
            MlxLayerState::Full(state) => [&state.keys, &state.values],
        };
        for array in arrays {
            array.eval().map_err(|error| MlxError::Operation {
                operation: "continuation state eval",
                message: error.to_string(),
            })?;
        }
    }
    Ok(())
}

/// Final RMSNorm weight, folded `(1 + w)` like every other norm.
fn folded_final_norm(
    tensors: &std::collections::BTreeMap<String, Array>,
    precision: MlxPrecision,
) -> Result<Array, MlxError> {
    let raw = tensors
        .get("model.language_model.norm.weight")
        .ok_or_else(|| MlxError::InvalidState("final norm weight was not loaded".to_owned()))?;
    let values = raw
        .to_vec_cast::<f32>()
        .map_err(|error| MlxError::Operation {
            operation: "final norm read",
            message: error.to_string(),
        })?;
    if values.len() != HIDDEN_SIZE {
        return Err(MlxError::InvalidState(format!(
            "final norm weight has {} values, expected {HIDDEN_SIZE}",
            values.len()
        )));
    }
    let folded: Vec<f32> = values.iter().map(|value| 1.0 + value).collect();
    Ok(match precision {
        MlxPrecision::Fp32 => Array::from_slice(&folded, &[HIDDEN_SIZE as i32]),
        MlxPrecision::NativeBf16 => Array::from_slice(
            &folded
                .iter()
                .map(|value| half::bf16::from_f32(*value))
                .collect::<Vec<_>>(),
            &[HIDDEN_SIZE as i32],
        ),
    })
}
