//! Continuation state container and validation for the MLX Qwen3.5 backbone.

use openkind_runtime::branch::{StateIdentity, StateLineage};

use crate::qwen35::mlx::layers::{
    MlxLayerState, ATTENTION_HEAD_DIM, HEAD_DIM, KV_HEADS, QKV_SIZE, VALUE_HEADS,
};
use crate::qwen35::mlx::runtime::SharedMlxRuntime;
use crate::qwen35::mlx::{MlxError, MlxPrecision};

use super::MlxQwen35Backbone;

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

impl MlxQwen35Backbone {
    pub(super) fn validate_state(&self, state: &MlxBackboneState) -> Result<(), MlxError> {
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
                        if layer.conv.shape() != [QKV_SIZE as i32, 4]
                            || layer.recurrent.shape()
                                != [
                                    VALUE_HEADS as i32,
                                    HEAD_DIM as i32,
                                    HEAD_DIM as i32,
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
                                QKV_SIZE,
                                VALUE_HEADS,
                                HEAD_DIM,
                                HEAD_DIM,
                                expected_dtype,
                            )));
                        }
                    }
                    (true, MlxLayerState::Full(layer)) => {
                        let expected_shape = [
                            position,
                            KV_HEADS as i32,
                            ATTENTION_HEAD_DIM as i32,
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
}
