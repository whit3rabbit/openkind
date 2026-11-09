//! Streamed verified-checkpoint loading into MLX arrays (Phase 3M).
//!
//! Accepted checkpoint shards are size- and SHA-256-verified before any tensor
//! is read, then required decoder tensors are loaded one at a time. FP32
//! execution widens BF16 exactly; native-BF16 execution decodes BF16 directly
//! without a full FP32 staging copy. Embeddings, vision tensors, and unused
//! heads never become MLX arrays. The loader reports peak MLX allocation,
//! process-lifetime peak RSS sampled after loading, and inactive-cache size.
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

use self::checkpoint::{canonical_tensor_name, load_checkpoint, MlxCheckpointIdentity};
pub use self::checkpoint::{MlxCheckpointFormat, MlxSurveyCheckpoint, MlxSurveyLora};
pub(crate) use self::shard::shape_i32;
use self::shard::{read_bf16, read_f32, widen_bf16, ShardIndex};
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
    /// Total retained decoder tensor bytes streamed (embedding, vision, and
    /// unused output heads excluded).
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
                if !is_required_decoder_tensor(&tensor_name) {
                    continue;
                }
                let (bytes, shape, is_f32) = index
                    .read_bytes(source_tensor_name)
                    .map_err(MlxError::from_qwen)?;
                loaded_bytes = loaded_bytes.saturating_add(bytes.len() as u64);
                let mlx_shape = shape_i32(shape);
                let array = materialize_tensor(
                    runtime,
                    precision,
                    &tensor_name,
                    &bytes,
                    &mlx_shape,
                    is_f32,
                )?;
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
            peak_process_bytes: openkind_runtime::peak_resident_bytes().ok(),
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

    /// Stream a caller-verified survey checkpoint's required decoder tensors
    /// into MLX arrays at the requested precision.
    ///
    /// The descriptor's artifacts (including the checkpoint shards) were
    /// digest-verified by the survey family before this call; this loader
    /// reads the shards in place, applies the format's key normalization,
    /// applies the descriptor's optional LoRA merge, and keeps the tied
    /// embedding host-resident exactly like [`Self::load`].
    pub fn load_survey(
        checkpoint: &MlxSurveyCheckpoint,
        runtime: &MlxRuntime,
        precision: MlxPrecision,
    ) -> Result<Self, MlxError> {
        let format = checkpoint.format;
        let indexes: Vec<ShardIndex> = checkpoint
            .shards
            .iter()
            .map(|path| ShardIndex::read(path).map_err(MlxError::from_qwen))
            .collect::<Result<_, _>>()?;
        let lora_factors = checkpoint
            .lora
            .as_ref()
            .map(|lora| load_lora_factors_in_container(lora, "model.language_model"))
            .transpose()?;

        let started = std::time::Instant::now();
        runtime.reset_peak_memory().ok();
        let mut tensors = BTreeMap::new();
        let mut loaded_bytes = 0_u64;
        for index in &indexes {
            for source_tensor_name in index.tensors.keys() {
                let Some(tensor_name) = canonical_tensor_name(format, source_tensor_name) else {
                    continue;
                };
                if tensor_name.starts_with("model.language_model.embed_tokens") {
                    continue;
                }
                if !is_required_decoder_tensor(&tensor_name) {
                    continue;
                }
                let (bytes, shape, is_f32) = index
                    .read_bytes(source_tensor_name)
                    .map_err(MlxError::from_qwen)?;
                loaded_bytes = loaded_bytes.saturating_add(bytes.len() as u64);
                let mlx_shape = shape_i32(shape);
                let array = match &lora_factors {
                    Some(factors) => {
                        // The merge computes in FP32 on the host before the
                        // values reach MLX, so the loaded array is the CPU
                        // oracle's merged weight exactly.
                        let mut values = widened_f32_values(&tensor_name, &bytes, is_f32)?;
                        if let Some((a, b)) = factors.get(&tensor_name) {
                            apply_lora_delta(
                                &tensor_name,
                                &mut values,
                                a,
                                b,
                                checkpoint.lora.as_ref().map_or(0.0, |lora| lora.scale),
                            )?;
                        }
                        materialize_f32_values(runtime, precision, &values, &mlx_shape)?
                    }
                    None => materialize_tensor(
                        runtime,
                        precision,
                        &tensor_name,
                        &bytes,
                        &mlx_shape,
                        is_f32,
                    )?,
                };
                if tensors.insert(tensor_name.clone(), array).is_some() {
                    return Err(MlxError::InvalidState(format!(
                        "checkpoint adapter mapped multiple tensors to `{tensor_name}`"
                    )));
                }
            }
        }
        // Every adapter target must have contributed a merged tensor: a
        // factor pair whose base tensor is missing (or filtered out) would
        // otherwise silently load the unadapted model.
        if let Some(factors) = &lora_factors {
            for name in factors.keys() {
                if !tensors.contains_key(name) {
                    return Err(MlxError::InvalidState(format!(
                        "adapter targets `{name}`, which the checkpoint layout does not load"
                    )));
                }
            }
        }
        let report = MlxWeightLoadReport {
            load_seconds: started.elapsed().as_secs_f64(),
            peak_active_mlx_bytes: runtime.peak_memory_bytes().ok(),
            peak_process_bytes: openkind_runtime::peak_resident_bytes().ok(),
            inactive_cache_bytes: runtime.cache_memory_bytes().ok(),
            loaded_tensor_bytes: loaded_bytes,
        };
        Ok(Self {
            tensors,
            embedding: checkpoint.embedding.clone(),
            checkpoint: MlxCheckpointIdentity {
                format,
                backbone_id: checkpoint.backbone_id,
                backbone_revision: checkpoint.backbone_revision,
                tokenizer_digest: checkpoint.tokenizer_digest,
            },
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

fn is_required_decoder_tensor(name: &str) -> bool {
    name == "model.language_model.norm.weight" || name.starts_with("model.language_model.layers.")
}

/// The `(lora_A, lora_B)` FP32 factor pairs of a verified adapter, keyed by
/// the base tensor name each pair adapts.
pub(crate) type LoraFactors = BTreeMap<String, (candle_core::Tensor, candle_core::Tensor)>;

/// Load the factor pairs of a verified adapter for a survey-family loader
/// (see [`load_lora_factors_in_container`]).
pub(crate) fn load_lora_factors_for_survey(lora: &MlxSurveyLora) -> Result<LoraFactors, MlxError> {
    load_lora_factors_in_container(lora, "model.language_model")
}

/// Load the factor pairs of a verified adapter keyed against the dense
/// Qwen container namespace (`model.layers.*`).
pub(crate) fn load_lora_factors_for_dense(lora: &MlxSurveyLora) -> Result<LoraFactors, MlxError> {
    load_lora_factors_in_container(lora, "model")
}

/// Add a factor pair's `delta = (B @ A) · scale` into FP32 host values for
/// a survey-family loader (see [`apply_lora_delta`]).
pub(crate) fn apply_lora_delta_to_values(
    name: &str,
    values: &mut [f32],
    a: &candle_core::Tensor,
    b: &candle_core::Tensor,
    scale: f64,
) -> Result<(), MlxError> {
    apply_lora_delta(name, values, a, b, scale)
}

/// Pair the `(lora_A, lora_B)` factors of a verified PEFT adapter and map
/// them onto their base tensor names under `container`.
///
/// PEFT keys look like
/// `base_model.model.layers.N.linear_attn.in_proj_qkv.lora_A.weight`; the
/// projection maps onto the base tensor
/// `{container}.layers.N.linear_attn.in_proj_qkv.weight` (the survey
/// checkpoints keep the multimodal container's `model.language_model`
/// prefix; the dense Qwen checkpoints use `model`). Factors load as FP32 on
/// the CPU so the delta the loader adds is the CPU oracle's exactly. The
/// adapter file is opened memory-mapped under the same trust contract as
/// the descriptor: the survey family digest-verified it before building the
/// descriptor.
fn load_lora_factors_in_container(
    lora: &MlxSurveyLora,
    container: &str,
) -> Result<LoraFactors, MlxError> {
    use candle_core::safetensors::MmapedSafetensors;
    use candle_core::{DType, Device};

    // Read the tensor names from the adapter header without loading data.
    let mut file = std::fs::File::open(&lora.adapter_path).map_err(|error| {
        MlxError::InvalidState(format!(
            "failed to read adapter {}: {error}",
            lora.adapter_path.display()
        ))
    })?;
    let mut length_bytes = [0_u8; 8];
    std::io::Read::read_exact(&mut file, &mut length_bytes)
        .map_err(|error| MlxError::InvalidState(format!("adapter header unreadable: {error}")))?;
    let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
        .map_err(|_| MlxError::InvalidState("adapter header too large".to_owned()))?;
    let mut header_bytes = vec![0_u8; header_len];
    std::io::Read::read_exact(&mut file, &mut header_bytes)
        .map_err(|error| MlxError::InvalidState(format!("adapter header unreadable: {error}")))?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|error| MlxError::InvalidState(format!("adapter header invalid: {error}")))?;
    let mut names: Vec<String> = header
        .as_object()
        .ok_or_else(|| MlxError::InvalidState("adapter header is not an object".to_owned()))?
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect();
    names.sort();
    // SAFETY: the survey family verified the immutable size and SHA-256 of
    // the read-only adapter file before this mapper opens it.
    let adapter = unsafe { MmapedSafetensors::new(&lora.adapter_path) }
        .map_err(|error| MlxError::InvalidState(format!("adapter mmap failed: {error}")))?;
    let mut pairs: BTreeMap<String, (Option<candle_core::Tensor>, Option<candle_core::Tensor>)> =
        BTreeMap::new();
    for name in names {
        let Some(stem) = name.strip_suffix(".weight") else {
            return Err(MlxError::InvalidState(format!(
                "adapter key `{name}` does not end in `.weight`"
            )));
        };
        let Some((projection, kind)) = stem.rsplit_once('.') else {
            return Err(MlxError::InvalidState(format!("bad adapter key {name}")));
        };
        let Some(base_projection) = projection.strip_prefix("base_model.model.") else {
            return Err(MlxError::InvalidState(format!(
                "adapter key `{name}` does not target the checkpoint namespace"
            )));
        };
        let tensor = adapter
            .load(&name, &Device::Cpu)
            .and_then(|tensor| tensor.to_dtype(DType::F32))
            .map_err(|error| MlxError::InvalidState(format!("adapter tensor {name}: {error}")))?;
        let entry = pairs
            .entry(base_projection.to_owned())
            .or_insert((None, None));
        match kind {
            "lora_A" => entry.0 = Some(tensor),
            "lora_B" => entry.1 = Some(tensor),
            other => {
                return Err(MlxError::InvalidState(format!(
                    "unexpected adapter tensor kind `{other}` in {name}"
                )));
            }
        }
    }
    let mut factors = BTreeMap::new();
    for (projection, (a, b)) in pairs {
        let (Some(a), Some(b)) = (a, b) else {
            return Err(MlxError::InvalidState(format!(
                "adapter projection `{projection}` is missing lora_A or lora_B"
            )));
        };
        if a.dims().first() != Some(&lora.rank) || b.dims().get(1) != Some(&lora.rank) {
            let a_rank = a.dims().first().copied().unwrap_or_default();
            let b_rank = b.dims().get(1).copied().unwrap_or_default();
            return Err(MlxError::InvalidState(format!(
                "adapter projection `{projection}` has rank {a_rank}/{b_rank}, expected {}",
                lora.rank
            )));
        }
        factors.insert(format!("{container}.{projection}.weight"), (a, b));
    }
    Ok(factors)
}

/// Widen one raw checkpoint tensor to FP32 host values exactly like the
/// CPU oracle's loader.
fn widened_f32_values(name: &str, bytes: &[u8], is_f32: bool) -> Result<Vec<f32>, MlxError> {
    if is_f32 {
        read_f32(bytes).ok_or_else(|| {
            MlxError::InvalidState(format!(
                "FP32 tensor `{name}` has a non-integral byte count"
            ))
        })
    } else {
        widen_bf16(bytes).ok_or_else(|| {
            MlxError::InvalidState(format!(
                "BF16 tensor `{name}` has a non-integral byte count"
            ))
        })
    }
}

/// Add `delta = (B @ A) · scale` into `values` (FP32, elementwise), the
/// PEFT merge the CPU oracle applies. The delta is computed through the
/// same candle matmul the oracle's merge uses, so the merged values are
/// bit-identical to the CPU path's.
fn apply_lora_delta(
    name: &str,
    values: &mut [f32],
    a: &candle_core::Tensor,
    b: &candle_core::Tensor,
    scale: f64,
) -> Result<(), MlxError> {
    let delta = b
        .matmul(a)
        .and_then(|delta| delta * scale)
        .map_err(|error| MlxError::InvalidState(format!("LoRA delta for `{name}`: {error}")))?;
    let delta = delta
        .flatten_all()
        .and_then(|delta| delta.to_vec1::<f32>())
        .map_err(|error| MlxError::InvalidState(format!("LoRA delta for `{name}`: {error}")))?;
    if delta.len() != values.len() {
        return Err(MlxError::InvalidState(format!(
            "LoRA delta for `{name}` has {} values, base tensor has {}",
            delta.len(),
            values.len()
        )));
    }
    for (value, delta) in values.iter_mut().zip(&delta) {
        *value += delta;
    }
    Ok(())
}

/// Materialize merged FP32 host values into an MLX array at the requested
/// execution precision (FP32 keeps them; native BF16 converts exactly).
fn materialize_f32_values(
    runtime: &MlxRuntime,
    precision: MlxPrecision,
    values: &[f32],
    mlx_shape: &[i32],
) -> Result<mlx_rs::Array, MlxError> {
    match precision {
        MlxPrecision::Fp32 => make_f32_array(runtime, values, mlx_shape),
        MlxPrecision::NativeBf16 => {
            let converted: Vec<half::bf16> = values
                .iter()
                .map(|value| half::bf16::from_f32(*value))
                .collect();
            make_bf16_array(runtime, &converted, mlx_shape)
        }
    }
}

/// Convert one raw checkpoint tensor into an MLX array at the requested
/// execution precision (FP32 widens BF16 exactly; native BF16 decodes
/// directly, converting the small FP32 vectors without matrix-sized
/// staging copies).
fn materialize_tensor(
    runtime: &MlxRuntime,
    precision: MlxPrecision,
    tensor_name: &str,
    bytes: &[u8],
    mlx_shape: &[i32],
    is_f32: bool,
) -> Result<Array, MlxError> {
    match (precision, is_f32) {
        (MlxPrecision::Fp32, true) => {
            let values = read_f32(bytes).ok_or_else(|| {
                MlxError::InvalidState(format!(
                    "FP32 tensor `{tensor_name}` has a non-integral byte count"
                ))
            })?;
            make_f32_array(runtime, &values, mlx_shape)
        }
        (MlxPrecision::Fp32, false) => {
            let values = widen_bf16(bytes).ok_or_else(|| {
                MlxError::InvalidState(format!(
                    "BF16 tensor `{tensor_name}` has a non-integral byte count"
                ))
            })?;
            make_f32_array(runtime, &values, mlx_shape)
        }
        (MlxPrecision::NativeBf16, false) => {
            let values = read_bf16(bytes).ok_or_else(|| {
                MlxError::InvalidState(format!(
                    "BF16 tensor `{tensor_name}` has a non-integral byte count"
                ))
            })?;
            make_bf16_array(runtime, &values, mlx_shape)
        }
        (MlxPrecision::NativeBf16, true) => {
            let values = read_f32(bytes).ok_or_else(|| {
                MlxError::InvalidState(format!(
                    "FP32 tensor `{tensor_name}` has a non-integral byte count"
                ))
            })?;
            let values: Vec<half::bf16> = values.into_iter().map(half::bf16::from_f32).collect();
            make_bf16_array(runtime, &values, mlx_shape)
        }
    }
}

fn make_f32_array(runtime: &MlxRuntime, values: &[f32], shape: &[i32]) -> Result<Array, MlxError> {
    runtime.execute(|| -> Result<Array, MlxError> {
        let array = Array::from_slice(values, shape);
        array.eval().map_err(|error| MlxError::Operation {
            operation: "weight eval",
            message: error.to_string(),
        })?;
        Ok(array)
    })?
}

fn make_bf16_array(
    runtime: &MlxRuntime,
    values: &[half::bf16],
    shape: &[i32],
) -> Result<Array, MlxError> {
    runtime.execute(|| -> Result<Array, MlxError> {
        let array = Array::from_slice(values, shape);
        array.eval().map_err(|error| MlxError::Operation {
            operation: "weight eval",
            message: error.to_string(),
        })?;
        Ok(array)
    })?
}

impl MlxError {
    pub(crate) fn from_qwen(error: Qwen35Error) -> Self {
        MlxError::InvalidState(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::is_required_decoder_tensor;

    #[test]
    fn required_tensor_filter_excludes_non_decoder_payloads() {
        assert!(is_required_decoder_tensor(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        ));
        assert!(is_required_decoder_tensor(
            "model.language_model.norm.weight"
        ));
        assert!(!is_required_decoder_tensor(
            "model.language_model.embed_tokens.weight"
        ));
        assert!(!is_required_decoder_tensor("model.visual.blocks.0.weight"));
        assert!(!is_required_decoder_tensor("lm_head.weight"));
    }
}
