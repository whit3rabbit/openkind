//! MLX Qwen3.5 backbone: prefill, continuation, and per-layer states.
//!
//! Executes the same 32-layer graph as the Candle CPU oracle over MLX
//! arrays through [`MlxDecoderLayer`](super::layers::MlxDecoderLayer). The
//! tied embedding table stays host-resident (verified seek-based reads, one
//! row per token, widened exactly), so MLX memory holds only decoder weights
//! and per-request state. Continuation states are fully materialized
//! (evaluated) at this executor boundary.

use std::path::Path;

use mlx_rs::Array;
use openkind_runtime::branch::StateIdentity;

use crate::qwen35::mlx::layers::{
    MlxDecoderLayer, ATTENTION_HEAD_DIM, CONV_KERNEL, HEAD_DIM, KV_HEADS, QKV_SIZE, VALUE_HEADS,
};
use crate::qwen35::mlx::runtime::SharedMlxRuntime;
use crate::qwen35::mlx::weights::{
    MlxCheckpointFormat, MlxSurveyCheckpoint, MlxWeightLoadReport, MlxWeightStore,
};
use crate::qwen35::mlx::{MlxError, MlxPrecision};
use crate::qwen35::{Qwen35Embedding, PROFILE_ID, STATE_FIRST_RENDERER_ID};

mod batched;
mod execution;
mod helpers;
mod state;
mod trace;
mod types;

#[cfg(test)]
mod tests;

pub use state::MlxBackboneState;
pub use types::{MlxBackboneOutput, MlxBackboneTrace, MlxOperationTrace, MAX_VECTOR_BATCH_LANES};

use helpers::{effective_gated_delta_kernel, folded_final_norm};

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
    pub(crate) fn embedding_rows(&self, ids: &[u32]) -> Result<Vec<f32>, MlxError> {
        self.embedding
            .embed(ids)
            .map(|output| output.values().to_vec())
            .map_err(MlxError::from_qwen)
    }

    /// Arithmetic precision used by this loaded backbone.
    #[must_use]
    pub const fn precision(&self) -> MlxPrecision {
        self.precision
    }

    /// Verify the pinned checkpoint and load it into MLX arrays at the
    /// requested precision.
    pub fn load(
        checkpoint_root: impl AsRef<Path>,
        runtime: SharedMlxRuntime,
        precision: MlxPrecision,
    ) -> Result<Self, MlxError> {
        if precision == MlxPrecision::NativeBf16 && !runtime.bf16_qualified() {
            return Err(MlxError::InvalidState(
                "native BF16 model load requires runtime.qualify_bf16() to pass first".to_owned(),
            ));
        }
        let store = MlxWeightStore::load(checkpoint_root, &runtime, precision)?;
        Self::from_store(
            store,
            runtime,
            precision,
            PROFILE_ID,
            STATE_FIRST_RENDERER_ID,
        )
    }

    /// Load a caller-verified survey checkpoint into MLX arrays at the
    /// requested precision.
    ///
    /// The survey family owns its pinned digests and verifies every artifact
    /// before building the descriptor; the backbone identity binds the
    /// survey profile's profile/renderer identifiers so any state produced by
    /// a prefill cannot be confused with a pinned-profile state.
    pub fn load_survey(
        checkpoint: &MlxSurveyCheckpoint,
        runtime: SharedMlxRuntime,
        precision: MlxPrecision,
        profile_id: &str,
        renderer_id: &str,
    ) -> Result<Self, MlxError> {
        if precision == MlxPrecision::NativeBf16 && !runtime.bf16_qualified() {
            return Err(MlxError::InvalidState(
                "native BF16 model load requires runtime.qualify_bf16() to pass first".to_owned(),
            ));
        }
        let store = MlxWeightStore::load_survey(checkpoint, &runtime, precision)?;
        Self::from_store(store, runtime, precision, profile_id, renderer_id)
    }

    /// Build the backbone from a verified weight store (layer construction
    /// and identity binding are shared by every checkpoint format).
    fn from_store(
        store: MlxWeightStore,
        runtime: SharedMlxRuntime,
        precision: MlxPrecision,
        profile_id: &str,
        renderer_id: &str,
    ) -> Result<Self, MlxError> {
        let load_report = store.load_report;
        let checkpoint_identity = store.checkpoint_identity();
        let (tensors, embedding) = store.into_tensors();
        let gated_delta_kernel =
            effective_gated_delta_kernel(precision, runtime.config().gated_delta_kernel);

        let identity = StateIdentity::new(
            profile_id,
            checkpoint_identity.backbone_id,
            checkpoint_identity.backbone_revision,
            renderer_id,
            checkpoint_identity.tokenizer_digest,
            helpers::arithmetic_id(
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
            QKV_SIZE.saturating_mul(CONV_KERNEL).saturating_add(
                VALUE_HEADS
                    .saturating_mul(HEAD_DIM)
                    .saturating_mul(HEAD_DIM),
            ),
        );
        let full_elements = full_layers
            .saturating_mul(position)
            .saturating_mul(2)
            .saturating_mul(KV_HEADS)
            .saturating_mul(ATTENTION_HEAD_DIM);
        linear_elements
            .saturating_add(full_elements)
            .saturating_mul(element_bytes)
    }
}
