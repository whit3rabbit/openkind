//! Public and internal output/trace types for the MLX Qwen3.5 backbone.

use crate::qwen35::mlx::layers::MlxLayerState;

use super::state::MlxBackboneState;

/// Production backbone output materialized to host FP32 at the executor boundary.
#[derive(Debug, Clone)]
pub struct MlxBackboneOutput {
    /// Number of tokens in the executed suffix.
    pub token_count: usize,
    /// Final RMSNorm output for the last token only.
    pub(super) final_feature: Vec<f32>,
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

pub(super) type TraceVectors = (
    Vec<f32>,
    Vec<Vec<f32>>,
    Vec<String>,
    Option<usize>,
    Vec<MlxOperationTrace>,
);

pub(super) type LayerTrace = (
    Vec<Vec<f32>>,
    Vec<String>,
    Option<usize>,
    Vec<MlxOperationTrace>,
);

/// Maximum lane count supported by the current vectorized FP32 path.
pub const MAX_VECTOR_BATCH_LANES: usize = 8;

pub(super) struct BackboneExecution {
    pub(super) output: MlxBackboneOutput,
    pub(super) state: MlxBackboneState,
    pub(super) trace: Option<TraceVectors>,
}

pub(super) struct DeviceExecution {
    pub(super) output: MlxBackboneOutput,
    pub(super) layers: Vec<MlxLayerState>,
    pub(super) layer_trace: Option<LayerTrace>,
}
