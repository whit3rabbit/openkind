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
use openkind_runtime::branch::{StateIdentity, StateLineage};

use crate::qwen35::{Qwen35Embedding, PROFILE_ID, STATE_FIRST_RENDERER_ID};

use super::layers::{
    gather_last_real_token, lane_of, prefix_axis0, row_of, stack_lanes, tail_rows,
    LayerOperationArrays, MlxDecoderLayer, MlxFullState, MlxLayerState, MlxLinearState,
    HIDDEN_SIZE,
};
use super::runtime::SharedMlxRuntime;
use super::weights::{MlxCheckpointFormat, MlxWeightLoadReport, MlxWeightStore};
use super::{MlxError, MlxPrecision};

/// Production backbone output materialized to host FP32 at the executor boundary.
#[derive(Debug, Clone)]
pub struct MlxBackboneOutput {
    /// Number of tokens in the executed suffix.
    pub token_count: usize,
    /// Final RMSNorm output for the last token only.
    final_feature: Vec<f32>,
}

impl MlxBackboneOutput {
    /// Final-token feature vector consumed by the score-summary head.
    #[must_use]
    pub fn feature(&self) -> &[f32] {
        &self.final_feature
    }
}

/// Opt-in diagnostic trace for frozen-stage parity localization.
///
/// Production scoring deliberately does not construct this value: collecting
/// it inserts one GPU-to-host readback after every decoder layer.
#[derive(Debug, Clone)]
pub struct MlxBackboneTrace {
    /// Production result from the same forward.
    pub output: MlxBackboneOutput,
    /// Host-read embedding row of the final token (pre-layer).
    pub embedding_last_token: Vec<f32>,
    /// Final-token hidden vector after each of the 32 layers, in order.
    pub layer_last_tokens: Vec<Vec<f32>>,
    /// MLX dtype of each layer's full hidden tensor, in model order.
    pub layer_output_dtypes: Vec<String>,
    /// Decoder layer selected for detailed operation tracing, if any.
    pub operation_trace_layer: Option<usize>,
    /// Opt-in detailed operation arrays restricted to the requested tail rows.
    pub operation_trace: Vec<MlxOperationTrace>,
}

/// Materialized output of one operation in a selected decoder layer.
#[derive(Debug, Clone)]
pub struct MlxOperationTrace {
    /// Layer operation name.
    pub operation: String,
    /// MLX dtype of the selected suffix array.
    pub dtype: String,
    /// Shape after selecting the diagnostic suffix rows.
    pub shape: Vec<i32>,
    /// Row-major FP32 host copy used for numerical comparison.
    pub values: Vec<f32>,
}

type TraceVectors = (
    Vec<f32>,
    Vec<Vec<f32>>,
    Vec<String>,
    Option<usize>,
    Vec<MlxOperationTrace>,
);
type LayerTrace = (
    Vec<Vec<f32>>,
    Vec<String>,
    Option<usize>,
    Vec<MlxOperationTrace>,
);

/// Maximum lane count supported by the current vectorized FP32 path.
pub const MAX_VECTOR_BATCH_LANES: usize = 8;

struct BackboneExecution {
    output: MlxBackboneOutput,
    state: MlxBackboneState,
    trace: Option<TraceVectors>,
}

struct DeviceExecution {
    output: MlxBackboneOutput,
    layers: Vec<MlxLayerState>,
    layer_trace: Option<LayerTrace>,
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
    /// Arithmetic precision used by this loaded backbone.
    #[must_use]
    pub const fn precision(&self) -> MlxPrecision {
        self.precision
    }

    /// Whether this instance can execute the current vectorized continuation
    /// implementation without changing its arithmetic profile.
    #[must_use]
    pub(crate) fn supports_vectorized_batch(&self) -> bool {
        self.precision == MlxPrecision::Fp32
            && self
                .layers
                .iter()
                .all(MlxDecoderLayer::supports_reference_batch)
    }

    pub(crate) fn can_vectorize_batch(
        &self,
        states: &[&MlxBackboneState],
        suffix_ids: &[&[u32]],
    ) -> bool {
        if !self.supports_vectorized_batch()
            || !(2..=MAX_VECTOR_BATCH_LANES).contains(&states.len())
            || states.len() != suffix_ids.len()
            || suffix_ids.iter().any(|suffix| suffix.is_empty())
        {
            return false;
        }
        let position = states[0].position;
        states.iter().all(|state| state.position == position)
    }

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
        let gated_delta_kernel =
            effective_gated_delta_kernel(precision, runtime.config().gated_delta_kernel);

        let identity = StateIdentity::new(
            PROFILE_ID,
            checkpoint_identity.backbone_id,
            checkpoint_identity.backbone_revision,
            STATE_FIRST_RENDERER_ID,
            checkpoint_identity.tokenizer_digest,
            arithmetic_id(
                precision,
                gated_delta_kernel,
                &runtime,
                checkpoint_identity.format,
            ),
        )
        .map_err(|error| MlxError::InvalidState(error.to_string()))?;

        let (layers, final_norm) = runtime.execute(|| -> Result<_, MlxError> {
            let mut layers = Vec::with_capacity(32);
            for layer_index in 0..32 {
                layers.push(MlxDecoderLayer::load(
                    &tensors,
                    layer_index,
                    precision,
                    gated_delta_kernel,
                )?);
            }
            let final_norm = folded_final_norm(&tensors, precision)?;
            // Weight arrays must leave the loading thread fully materialized:
            // lazy graphs record this thread's stream, which worker threads
            // (engine blocking-pool forwards) cannot resolve.
            for (layer_index, layer) in layers.iter().enumerate() {
                layer.materialize(layer_index)?;
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
        let linear_elements = linear_layers.saturating_mul(
            super::layers::QKV_SIZE
                .saturating_mul(super::layers::CONV_KERNEL)
                .saturating_add(
                    super::layers::VALUE_HEADS
                        .saturating_mul(super::layers::HEAD_DIM)
                        .saturating_mul(super::layers::HEAD_DIM),
                ),
        );
        let full_elements = full_layers
            .saturating_mul(position)
            .saturating_mul(2)
            .saturating_mul(super::layers::KV_HEADS)
            .saturating_mul(super::layers::ATTENTION_HEAD_DIM);
        linear_elements
            .saturating_add(full_elements)
            .saturating_mul(element_bytes)
    }

    /// Prefill a fresh root sequence.
    pub fn prefill(
        &self,
        input_ids: &[u32],
    ) -> Result<(MlxBackboneOutput, MlxBackboneState), MlxError> {
        let execution = self.execute(input_ids, None, false, None, None)?;
        Ok((execution.output, execution.state))
    }

    /// Prefill while collecting per-layer host vectors for parity diagnostics.
    ///
    /// This is intentionally separate from [`Self::prefill`] so benchmark and
    /// serving paths cannot accidentally pay for 32 diagnostic readbacks.
    pub fn prefill_trace(
        &self,
        input_ids: &[u32],
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
        let execution = self.execute(input_ids, None, true, None, None)?;
        let (
            embedding_last_token,
            layer_last_tokens,
            layer_output_dtypes,
            operation_trace_layer,
            operation_trace,
        ) = execution.trace.ok_or_else(|| {
            MlxError::InvalidState("diagnostic trace was not collected".to_owned())
        })?;
        Ok((
            MlxBackboneTrace {
                output: execution.output,
                embedding_last_token,
                layer_last_tokens,
                layer_output_dtypes,
                operation_trace_layer,
                operation_trace,
            },
            execution.state,
        ))
    }

    /// Prefill while retaining detailed operation arrays for a suffix of the
    /// full input. This is intended for one-off cache/full diagnostics.
    pub fn prefill_trace_tail(
        &self,
        input_ids: &[u32],
        trace_tail_rows: usize,
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
        self.prefill_trace_layer_tail(input_ids, trace_tail_rows, 1)
    }

    /// Prefill while tracing a selected layer over the final input rows.
    pub fn prefill_trace_layer_tail(
        &self,
        input_ids: &[u32],
        trace_tail_rows: usize,
        operation_trace_layer: usize,
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
        if trace_tail_rows == 0 || trace_tail_rows > input_ids.len() {
            return Err(MlxError::InvalidState(format!(
                "diagnostic tail length {trace_tail_rows} is invalid for {} input rows",
                input_ids.len()
            )));
        }
        if operation_trace_layer >= self.layers.len() {
            return Err(MlxError::InvalidState(format!(
                "diagnostic layer {operation_trace_layer} is out of range for {} layers",
                self.layers.len()
            )));
        }
        let execution = self.execute(
            input_ids,
            None,
            true,
            Some(trace_tail_rows),
            Some(operation_trace_layer),
        )?;
        let (
            embedding_last_token,
            layer_last_tokens,
            layer_output_dtypes,
            operation_trace_layer,
            operation_trace,
        ) = execution.trace.ok_or_else(|| {
            MlxError::InvalidState("diagnostic trace was not collected".to_owned())
        })?;
        Ok((
            MlxBackboneTrace {
                output: execution.output,
                embedding_last_token,
                layer_last_tokens,
                layer_output_dtypes,
                operation_trace_layer,
                operation_trace,
            },
            execution.state,
        ))
    }

    /// Continue a state while collecting the final hidden vector and dtype
    /// after each layer. This is for targeted parity localization only.
    pub fn continue_from_trace(
        &self,
        state: &MlxBackboneState,
        suffix_ids: &[u32],
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
        self.continue_from_trace_layer(state, suffix_ids, 1)
    }

    /// Continue a state while tracing one selected decoder layer.
    pub fn continue_from_trace_layer(
        &self,
        state: &MlxBackboneState,
        suffix_ids: &[u32],
        operation_trace_layer: usize,
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
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
        if operation_trace_layer >= self.layers.len() {
            return Err(MlxError::InvalidState(format!(
                "diagnostic layer {operation_trace_layer} is out of range for {} layers",
                self.layers.len()
            )));
        }
        let execution = self.execute(
            suffix_ids,
            Some(state),
            true,
            Some(suffix_ids.len()),
            Some(operation_trace_layer),
        )?;
        let (
            embedding_last_token,
            layer_last_tokens,
            layer_output_dtypes,
            operation_trace_layer,
            operation_trace,
        ) = execution.trace.ok_or_else(|| {
            MlxError::InvalidState("diagnostic trace was not collected".to_owned())
        })?;
        Ok((
            MlxBackboneTrace {
                output: execution.output,
                embedding_last_token,
                layer_last_tokens,
                layer_output_dtypes,
                operation_trace_layer,
                operation_trace,
            },
            execution.state,
        ))
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
        let execution = self.execute(suffix_ids, Some(state), false, None, None)?;
        Ok((execution.output, execution.state))
    }

    /// Continue multiple equal-position states through one FP32 reference-ops
    /// graph with right-padded suffixes. The returned states preserve each
    /// lane's true position and lineage.
    pub fn continue_batch_vectorized(
        &self,
        states: &[&MlxBackboneState],
        suffix_ids: &[&[u32]],
    ) -> Result<Vec<(MlxBackboneOutput, MlxBackboneState)>, MlxError> {
        if !self.supports_vectorized_batch() {
            return Err(MlxError::InvalidState(
                "vectorized continuation currently supports only FP32 reference-ops".to_owned(),
            ));
        }
        let lengths = suffix_ids
            .iter()
            .map(|suffix| suffix.len())
            .collect::<Vec<_>>();
        let positions = states
            .iter()
            .map(|state| state.position)
            .collect::<Vec<_>>();
        let (position_start, rows) =
            validate_vectorized_batch_inputs(states.len(), &lengths, &positions)?;
        for (lane, state) in states.iter().enumerate() {
            if state.identity != self.identity {
                return Err(MlxError::InvalidState(format!(
                    "continuation lane {lane} has a different arithmetic identity"
                )));
            }
            if !std::sync::Arc::ptr_eq(&state.runtime, &self.runtime) {
                return Err(MlxError::InvalidState(format!(
                    "continuation lane {lane} belongs to a different MLX runtime"
                )));
            }
            self.validate_state(state)?;
        }
        let lanes = states.len();
        let next_position = position_start.checked_add(rows).ok_or_else(|| {
            MlxError::InvalidState("vectorized continuation position overflow".to_owned())
        })?;
        i32::try_from(next_position).map_err(|_| {
            MlxError::InvalidState(
                "vectorized continuation position does not fit MLX dimensions".to_owned(),
            )
        })?;

        let mut padded_embeddings = vec![0.0_f32; lanes * rows * HIDDEN_SIZE];
        for (lane, suffix) in suffix_ids.iter().enumerate() {
            let embedding = self.embedding.embed(suffix).map_err(MlxError::from_qwen)?;
            let start = lane * rows * HIDDEN_SIZE;
            let end = start + suffix.len() * HIDDEN_SIZE;
            padded_embeddings[start..end].copy_from_slice(embedding.values());
        }

        self.runtime.execute(|| -> Result<_, MlxError> {
            let mut hidden = Array::from_slice(
                &padded_embeddings,
                &[lanes as i32, rows as i32, HIDDEN_SIZE as i32],
            );
            let mut next_layers = Vec::with_capacity(self.layers.len());
            for (layer_index, layer) in self.layers.iter().enumerate() {
                let previous = stack_previous_layer(states, layer_index)?;
                let (next_hidden, next_state) =
                    layer.forward_batched(&hidden, rows, &lengths, position_start, &previous)?;
                hidden = next_hidden;
                hidden.eval().map_err(|error| MlxError::Operation {
                    operation: "batched decoder layer eval",
                    message: format!("layer {layer_index}: {error}"),
                })?;
                next_layers.push(next_state);
            }
            let final_array =
                fast::rms_norm(&hidden, Some(&self.final_norm), 1e-6).map_err(|error| {
                    MlxError::Operation {
                        operation: "batched final rms norm",
                        message: error.to_string(),
                    }
                })?;
            materialize_state(&next_layers)?;

            let mut outputs = Vec::with_capacity(lanes);
            for (lane, state) in states.iter().enumerate() {
                let feature =
                    array_to_host(&gather_last_real_token(&final_array, lane, lengths[lane])?)?;
                let lane_layers =
                    split_batch_layer_states(&next_layers, lane, position_start, lengths[lane])?;
                outputs.push((
                    MlxBackboneOutput {
                        token_count: lengths[lane],
                        final_feature: feature,
                    },
                    MlxBackboneState {
                        runtime: std::sync::Arc::clone(&self.runtime),
                        identity: self.identity.clone(),
                        lineage: state.lineage,
                        position: position_start + lengths[lane],
                        layers: lane_layers,
                    },
                ));
            }
            Ok(outputs)
        })?
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
                                "invalid linear cache at layer {layer_index}: conv shape/dtype {:?}/{:?}, recurrent shape/dtype {:?}/{:?}, expected conv [{}, 4], recurrent [{}, {}, {}], dtype {:?}",
                                layer.conv.shape(),
                                layer.conv.dtype(),
                                layer.recurrent.shape(),
                                layer.recurrent.dtype(),
                                super::layers::QKV_SIZE,
                                super::layers::VALUE_HEADS,
                                super::layers::HEAD_DIM,
                                super::layers::HEAD_DIM,
                                expected_dtype,
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
        collect_trace: bool,
        trace_tail_rows: Option<usize>,
        operation_trace_layer: Option<usize>,
    ) -> Result<BackboneExecution, MlxError> {
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
        let embedding_last_token = collect_trace.then(|| embedding.last_token().to_vec());

        let device = self
            .runtime
            .execute(|| -> Result<DeviceExecution, MlxError> {
                let mut hidden = embed_array(embedding.values(), self.precision);
                let mut layer_last_tokens =
                    collect_trace.then(|| Vec::with_capacity(self.layers.len()));
                let mut layer_output_dtypes =
                    collect_trace.then(|| Vec::with_capacity(self.layers.len()));
                let mut operation_traces = collect_trace.then(Vec::new);
                let mut next_layers = Vec::with_capacity(self.layers.len());
                for (layer_index, layer) in self.layers.iter().enumerate() {
                    let previous = previous_state.map(|state| &state.layers[layer_index]);
                    let mut operation_arrays = LayerOperationArrays::new();
                    let collect_layer_operations = collect_trace
                        && trace_tail_rows.is_some()
                        && operation_trace_layer == Some(layer_index);
                    let operation_trace = collect_layer_operations.then_some(&mut operation_arrays);
                    let (next_hidden, next_state) = layer.forward_traced(
                        &hidden,
                        token_count,
                        position_start,
                        previous,
                        operation_trace,
                    )?;
                    hidden = next_hidden;
                    if let Some(trace) = &mut layer_last_tokens {
                        trace.push(last_row_to_host(&hidden)?);
                        layer_output_dtypes
                            .as_mut()
                            .expect("layer dtype trace accompanies vector trace")
                            .push(format!("{:?}", hidden.dtype()));
                        if collect_layer_operations {
                            let tail_length = trace_tail_rows.ok_or_else(|| {
                                MlxError::InvalidState(
                                    "layer operation trace has no requested tail length".to_owned(),
                                )
                            })?;
                            let mut layer_operations = Vec::with_capacity(operation_arrays.len());
                            for (operation, array) in operation_arrays {
                                let tail = tail_rows(&array, tail_length)?;
                                layer_operations.push(MlxOperationTrace {
                                    operation,
                                    dtype: format!("{:?}", tail.dtype()),
                                    shape: tail.shape().to_vec(),
                                    values: array_to_host(&tail)?,
                                });
                            }
                            *operation_traces
                                .as_mut()
                                .expect("layer operation trace accompanies vector trace") =
                                layer_operations;
                        }
                    } else {
                        // Bound the lazy graph without a host readback.
                        hidden.eval().map_err(|error| MlxError::Operation {
                            operation: "decoder layer eval",
                            message: format!("layer {layer_index}: {error}"),
                        })?;
                    }
                    next_layers.push(next_state);
                }
                let final_array =
                    fast::rms_norm(&hidden, Some(&self.final_norm), 1e-6).map_err(|error| {
                        MlxError::Operation {
                            operation: "final rms norm",
                            message: error.to_string(),
                        }
                    })?;
                let final_row = row_of(&final_array, token_count - 1)?;
                let output = MlxBackboneOutput {
                    token_count,
                    final_feature: array_to_host(&final_row)?,
                };
                materialize_state(&next_layers)?;
                Ok(DeviceExecution {
                    output,
                    layers: next_layers,
                    layer_trace: layer_last_tokens
                        .zip(layer_output_dtypes)
                        .zip(operation_traces)
                        .map(|((layers, dtypes), operations)| {
                            (layers, dtypes, operation_trace_layer, operations)
                        }),
                })
            })??;

        Ok(BackboneExecution {
            output: device.output,
            state: MlxBackboneState {
                runtime: super::runtime::SharedMlxRuntime::clone(&self.runtime),
                identity: self.identity.clone(),
                lineage,
                position: position_start + token_count,
                layers: device.layers,
            },
            trace: device.layer_trace.map(|layers| {
                (
                    embedding_last_token.expect("trace embedding accompanies trace layers"),
                    layers.0,
                    layers.1,
                    layers.2,
                    layers.3,
                )
            }),
        })
    }
}

/// Arithmetic identity including precision, kernel family, and the linked
/// Xcode/Metal toolchain that affects native-BF16 behavior.
fn arithmetic_id(
    precision: MlxPrecision,
    gated_delta_kernel: super::MlxGatedDeltaKernel,
    runtime: &super::runtime::MlxRuntime,
    checkpoint_format: MlxCheckpointFormat,
) -> String {
    let base = match (precision, gated_delta_kernel) {
        (MlxPrecision::Fp32, super::MlxGatedDeltaKernel::ReferenceOps) => {
            super::MLX_ARITHMETIC_ID_FP32_REFERENCE
        }
        (MlxPrecision::NativeBf16, super::MlxGatedDeltaKernel::ReferenceOps) => {
            super::MLX_ARITHMETIC_ID_BF16_REFERENCE
        }
        (MlxPrecision::Fp32, super::MlxGatedDeltaKernel::MetalTree) => {
            super::MLX_ARITHMETIC_ID_FP32_METAL_TREE
        }
        (MlxPrecision::NativeBf16, super::MlxGatedDeltaKernel::MetalTree) => {
            super::MLX_ARITHMETIC_ID_BF16_METAL_TREE
        }
    };
    format!(
        "{base};checkpoint={};toolchain={}",
        checkpoint_format.as_str(),
        runtime.toolchain_identity()
    )
}

fn effective_gated_delta_kernel(
    precision: MlxPrecision,
    requested: super::MlxGatedDeltaKernel,
) -> super::MlxGatedDeltaKernel {
    match (precision, requested) {
        // The generic BF16 Metal kernel is directly tested, but its current
        // model-backed result exceeds the frozen probability gate. Keep the
        // candidate implementation out of serving/benchmark dispatch until a
        // distinct arithmetic profile qualifies it.
        (MlxPrecision::NativeBf16, super::MlxGatedDeltaKernel::MetalTree) => {
            super::MlxGatedDeltaKernel::ReferenceOps
        }
        _ => requested,
    }
}

fn validate_vectorized_batch_inputs(
    lanes: usize,
    suffix_lengths: &[usize],
    positions: &[usize],
) -> Result<(usize, usize), MlxError> {
    if !(2..=MAX_VECTOR_BATCH_LANES).contains(&lanes)
        || suffix_lengths.len() != lanes
        || positions.len() != lanes
    {
        return Err(MlxError::InvalidState(format!(
            "vectorized continuation supports 2..={MAX_VECTOR_BATCH_LANES} matching lanes, found {lanes} states, {} suffixes, and {} positions",
            suffix_lengths.len(),
            positions.len()
        )));
    }
    if suffix_lengths.contains(&0) {
        return Err(MlxError::InvalidState(
            "vectorized continuation contains an empty suffix".to_owned(),
        ));
    }
    let position_start = positions[0];
    if positions.iter().any(|position| *position != position_start) {
        return Err(MlxError::InvalidState(
            "vectorized continuation requires equal starting positions".to_owned(),
        ));
    }
    let rows = suffix_lengths.iter().copied().max().unwrap_or(0);
    Ok((position_start, rows))
}

fn stack_previous_layer(
    states: &[&MlxBackboneState],
    layer_index: usize,
) -> Result<MlxLayerState, MlxError> {
    match states[0].layers.get(layer_index) {
        Some(MlxLayerState::Linear(_)) => {
            let mut conv = Vec::with_capacity(states.len());
            let mut recurrent = Vec::with_capacity(states.len());
            for state in states {
                match state.layers.get(layer_index) {
                    Some(MlxLayerState::Linear(layer)) => {
                        conv.push(&layer.conv);
                        recurrent.push(&layer.recurrent);
                    }
                    _ => {
                        return Err(MlxError::InvalidState(format!(
                            "continuation cache kind differs at layer {layer_index}"
                        )));
                    }
                }
            }
            Ok(MlxLayerState::Linear(MlxLinearState {
                conv: stack_lanes(&conv)?,
                recurrent: stack_lanes(&recurrent)?,
            }))
        }
        Some(MlxLayerState::Full(_)) => {
            let mut keys = Vec::with_capacity(states.len());
            let mut values = Vec::with_capacity(states.len());
            for state in states {
                match state.layers.get(layer_index) {
                    Some(MlxLayerState::Full(layer)) => {
                        keys.push(&layer.keys);
                        values.push(&layer.values);
                    }
                    _ => {
                        return Err(MlxError::InvalidState(format!(
                            "continuation cache kind differs at layer {layer_index}"
                        )));
                    }
                }
            }
            Ok(MlxLayerState::Full(MlxFullState {
                keys: stack_lanes(&keys)?,
                values: stack_lanes(&values)?,
            }))
        }
        None => Err(MlxError::InvalidState(format!(
            "continuation state has no cache for layer {layer_index}"
        ))),
    }
}

fn split_batch_layer_states(
    batched: &[MlxLayerState],
    lane: usize,
    position_start: usize,
    suffix_length: usize,
) -> Result<Vec<MlxLayerState>, MlxError> {
    batched
        .iter()
        .enumerate()
        .map(|(layer_index, state)| match state {
            MlxLayerState::Linear(layer) => Ok(MlxLayerState::Linear(MlxLinearState {
                conv: lane_of(&layer.conv, lane)?,
                recurrent: lane_of(&layer.recurrent, lane)?,
            })),
            MlxLayerState::Full(layer) => {
                let valid_position =
                    position_start.checked_add(suffix_length).ok_or_else(|| {
                        MlxError::InvalidState(format!(
                            "attention position overflow at layer {layer_index}"
                        ))
                    })?;
                let keys = prefix_axis0(&lane_of(&layer.keys, lane)?, valid_position)?;
                let values = prefix_axis0(&lane_of(&layer.values, lane)?, valid_position)?;
                Ok(MlxLayerState::Full(MlxFullState { keys, values }))
            }
        })
        .collect()
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
    let row = row_of(array, rows - 1)?;
    array_to_host(&row)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::mlx::MlxGatedDeltaKernel;

    #[test]
    fn fused_kernel_promotion_is_precision_scoped() {
        assert_eq!(
            effective_gated_delta_kernel(MlxPrecision::Fp32, MlxGatedDeltaKernel::MetalTree),
            MlxGatedDeltaKernel::MetalTree,
        );
        assert_eq!(
            effective_gated_delta_kernel(MlxPrecision::NativeBf16, MlxGatedDeltaKernel::MetalTree,),
            MlxGatedDeltaKernel::ReferenceOps,
        );
    }

    #[test]
    fn vectorized_batch_accepts_right_padded_lengths_and_enforces_lane_limit() {
        assert_eq!(
            validate_vectorized_batch_inputs(3, &[2, 7, 4], &[19, 19, 19]).unwrap(),
            (19, 7)
        );
        assert!(validate_vectorized_batch_inputs(1, &[2], &[19]).is_err());
        assert!(validate_vectorized_batch_inputs(9, &[1; 9], &[19; 9]).is_err());
        assert!(validate_vectorized_batch_inputs(2, &[1], &[19, 19]).is_err());
        assert!(validate_vectorized_batch_inputs(2, &[1, 2], &[19, 20]).is_err());
        assert!(validate_vectorized_batch_inputs(2, &[1, 0], &[19, 19]).is_err());
    }
}
