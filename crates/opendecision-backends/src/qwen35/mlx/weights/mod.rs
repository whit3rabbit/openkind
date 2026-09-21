//! Streamed verified-checkpoint loading into MLX arrays (Phase 3M).
//!
//! Accepted checkpoint shards are size- and SHA-256-verified before any tensor
//! is read, then tensors are loaded one at a time: read BF16/FP32 bytes →
//! widen on host (or keep BF16 for
//! the candidate profile) → construct the MLX array → drop the staging
//! buffer. The loader never retains a second full copy of the weights in
//! host memory, and it reports peak MLX allocation, process-lifetime peak RSS
//! sampled after loading, and inactive-cache size.
//!
//! RMSNorm offset folding: the Candle oracle multiplies by `(1 + w)` per
//! element at execution time; this loader folds that offset once, in FP32 on
//! the host, before the value reaches MLX — the same values the CPU path
//! computes, matching mlx-lm's folded offset-RMSNorm handling of Qwen
//! checkpoints. Only the *layernorm* and q/k-norm weights fold; the DeltaNet
//! `norm.weight` is used raw (the oracle does not offset it).

mod checkpoint;
mod shard;

use std::collections::BTreeMap;
use std::path::Path;

use mlx_rs::Array;

pub use self::checkpoint::MlxCheckpointFormat;
use self::checkpoint::{canonical_tensor_name, load_checkpoint, MlxCheckpointIdentity};
pub(crate) use self::shard::shape_i32;
use self::shard::{read_f32, widen_bf16};
use super::runtime::MlxRuntime;
use super::{MlxError, MlxPrecision};
use crate::qwen35::{Qwen35Embedding, Qwen35Error};

/// Memory evidence captured while the weights were loaded.
#[derive(Debug, Clone, Copy, Default)]
pub struct MlxWeightLoadReport {
    /// Peak active MLX allocation observed during loading.
    pub peak_active_mlx_bytes: Option<usize>,
    /// Process RSS high-water mark sampled after loading. This is process
    /// lifetime telemetry, not a load-scoped peak, because `getrusage` does
    /// not provide a resettable RSS high-water mark.
    pub peak_process_bytes: Option<usize>,
    /// Inactive MLX allocator cache after loading.
    pub inactive_cache_bytes: Option<usize>,
    /// Wall time of the streamed tensor load in seconds.
    pub load_seconds: f64,
    /// Total checkpoint tensor bytes streamed (both shards, embedding table
    /// excluded).
    pub loaded_tensor_bytes: u64,
}

/// Verified, streamed MLX weight store for an accepted Qwen3.5 checkpoint
/// layout.
pub struct MlxWeightStore {
    tensors: BTreeMap<String, Array>,
    embedding: Qwen35Embedding,
    checkpoint: MlxCheckpointIdentity,
    /// Memory evidence captured while the tensors streamed in.
    pub load_report: MlxWeightLoadReport,
}

impl MlxWeightStore {
    /// Verify both shards and stream every tensor named in the index into
    /// MLX arrays at the requested precision.
    pub fn load(
        checkpoint_root: impl AsRef<Path>,
        runtime: &MlxRuntime,
        precision: MlxPrecision,
    ) -> Result<Self, MlxError> {
        let checkpoint = load_checkpoint(checkpoint_root.as_ref())?;
        let format = checkpoint.format;
        let indexes = checkpoint.indexes;

        let started = std::time::Instant::now();
        runtime.reset_peak_memory().ok();
        let mut tensors = BTreeMap::new();
        let mut loaded_bytes = 0_u64;
        for index in &indexes {
            for source_tensor_name in index.tensors.keys() {
                let Some(tensor_name) = canonical_tensor_name(format, source_tensor_name) else {
                    continue;
                };
                // The tied embedding table stays on the host: evaluation
                // reads only the rows named by the request (see the model's
                // embed step), so its 1.3 GiB never becomes GPU-resident.
                if tensor_name.starts_with("model.language_model.embed_tokens") {
                    continue;
                }
                let (bytes, shape, is_f32) = index
                    .read_bytes(source_tensor_name)
                    .map_err(MlxError::from_qwen)?;
                loaded_bytes = loaded_bytes.saturating_add(bytes.len() as u64);
                let values = if is_f32 {
                    read_f32(&bytes).ok_or_else(|| {
                        MlxError::InvalidState(format!(
                            "FP32 tensor `{tensor_name}` has a non-integral byte count"
                        ))
                    })?
                } else {
                    widen_bf16(&bytes).ok_or_else(|| {
                        MlxError::InvalidState(format!(
                            "BF16 tensor `{tensor_name}` has a non-integral byte count"
                        ))
                    })?
                };
                let array = runtime.execute(|| -> Result<Array, MlxError> {
                    let array = match precision {
                        MlxPrecision::Fp32 => Array::from_slice(&values, &shape_i32(shape)),
                        MlxPrecision::NativeBf16 => {
                            let bf16_values: Vec<half::bf16> =
                                values.iter().map(|v| half::bf16::from_f32(*v)).collect();
                            Array::from_slice(&bf16_values, &shape_i32(shape))
                        }
                    };
                    array.eval().map_err(|e| MlxError::Operation {
                        operation: "weight eval",
                        message: e.to_string(),
                    })?;
                    Ok(array)
                })??;
                if tensors.insert(tensor_name.clone(), array).is_some() {
                    return Err(MlxError::InvalidState(format!(
                        "checkpoint adapter mapped multiple tensors to `{tensor_name}`"
                    )));
                }
                // Staging buffers drop here, before the next tensor is read.
            }
        }
        let report = MlxWeightLoadReport {
            load_seconds: started.elapsed().as_secs_f64(),
            peak_active_mlx_bytes: runtime.peak_memory_bytes().ok(),
            peak_process_bytes: opendecision_runtime::peak_resident_bytes().ok(),
            inactive_cache_bytes: runtime.cache_memory_bytes().ok(),
            loaded_tensor_bytes: loaded_bytes,
        };
        Ok(Self {
            tensors,
            embedding: checkpoint.embedding,
            checkpoint: checkpoint.identity,
            load_report: report,
        })
    }

    /// Checkpoint identity selected by the verified adapter.
    #[must_use]
    pub(crate) fn checkpoint_identity(&self) -> MlxCheckpointIdentity {
        self.checkpoint
    }

    /// Take ownership of the loaded tensor map (layer construction).
    #[must_use]
    pub fn into_tensors(self) -> (BTreeMap<String, Array>, Qwen35Embedding) {
        (self.tensors, self.embedding)
    }
}

impl MlxError {
    pub(crate) fn from_qwen(error: Qwen35Error) -> Self {
        MlxError::InvalidState(error.to_string())
    }
}
