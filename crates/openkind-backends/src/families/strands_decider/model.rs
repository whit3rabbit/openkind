//! Pinned strands-decider checkpoint loading: base Qwen3.5-2B + LoRA merge +
//! pointer head.
//!
//! Loading verifies every artifact in place (cheap contracts first, the
//! multi-gigabyte base checkpoint last) and never copies or writes weights.
//! The adapter is merged lazily through a [`MergedLoRABackend`]: a candle
//! `SimpleBackend` view over the memory-mapped base checkpoint that adds
//! `(B @ A) · alpha/r` to each targeted weight as the shared Qwen3.5 layer
//! kernels read it, so the parity-verified forward runs untouched.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, Linear, Module};

use crate::families::decoder_logit_qwen35::model::resolve_config_field;
use crate::families::support::{verify_digest, FamilyControl, FamilyError};
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding, TextBackbone};

use super::{
    pinned_adapter_config, pinned_base_config, StrandsDeciderProfile, BASE_CHECKPOINT_SHA256,
    BASE_CONFIG_SHA256, HIDDEN_SIZE, HOBSON_V19, HOBSON_V21, LAYER_NORM_EPSILON, LORA_RANK,
    LORA_SCALE, POINTER_DIM, VOCAB_SIZE,
};

/// Digest-verified artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified base checkpoint (single shard).
    pub base_checkpoint: PathBuf,
    /// Path to the verified LoRA adapter.
    pub adapter: PathBuf,
    /// Path to the verified pointer head.
    pub head: PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: PathBuf,
    /// The verified profile (Hobson v19 or Hobson v21).
    pub profile: &'static StrandsDeciderProfile,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place with an optional explicit profile.
    ///
    /// Cheap contract checks run first so a drifted config, adapter, head,
    /// or tokenizer digest fails before the multi-gigabyte base checkpoint
    /// is streamed for its digest.
    pub fn verify_with_profile(
        model_root: &Path,
        base_root: &Path,
        profile: Option<&'static StrandsDeciderProfile>,
    ) -> Result<Self, FamilyError> {
        let base_checkpoint = base_root.join("model.safetensors");
        let base_config = base_root.join("config.json");
        let adapter = model_root.join("adapter_model.safetensors");
        let adapter_config = model_root.join("adapter_config.json");
        let head = model_root.join("head.safetensors");
        let tokenizer = model_root.join("tokenizer.json");
        let hobson_config = if model_root.join("hobson_config.json").is_file() {
            model_root.join("hobson_config.json")
        } else if model_root.join("strands_decider_config.json").is_file() {
            model_root.join("strands_decider_config.json")
        } else {
            model_root.join("hobson_config.json")
        };
        // Cheap artifacts first: existence, digest, and contract checks fail
        // fast before the multi-gigabyte checkpoint is even opened.
        for path in [
            &base_config,
            &adapter,
            &adapter_config,
            &head,
            &tokenizer,
            &hobson_config,
        ] {
            if !path.is_file() {
                return Err(FamilyError::Io {
                    path: path.to_path_buf(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "required pinned artifact is missing",
                    ),
                });
            }
        }

        let profile = match profile {
            Some(profile) => profile,
            None => {
                let adapter_config_hash = crate::families::support::sha256_file(&adapter_config)?;
                if adapter_config_hash == HOBSON_V21.adapter_config_sha256 {
                    &HOBSON_V21
                } else {
                    &HOBSON_V19
                }
            }
        };

        verify_digest(&hobson_config, profile.hobson_config_sha256)?;
        verify_digest(&adapter_config, profile.adapter_config_sha256)?;
        verify_digest(&tokenizer, profile.tokenizer_json_sha256)?;
        verify_digest(&head, profile.head_sha256)?;
        verify_digest(&adapter, profile.adapter_sha256)?;
        verify_digest(&base_config, BASE_CONFIG_SHA256)?;

        let base_json: serde_json::Value = crate::families::support::read_json(&base_config)?;
        for (field, expected) in pinned_base_config() {
            let actual = resolve_config_field(&base_json, field).ok_or_else(|| {
                FamilyError::ContractMismatch {
                    field,
                    expected: expected.to_string(),
                    actual: "missing".to_owned(),
                }
            })?;
            if actual != expected {
                return Err(FamilyError::contract(field, expected, actual));
            }
        }
        let hobson_json: serde_json::Value = crate::families::support::read_json(&hobson_config)?;
        for (field, expected) in profile.pinned_hobson_config() {
            let actual = resolve_config_field(&hobson_json, field).ok_or_else(|| {
                FamilyError::ContractMismatch {
                    field,
                    expected: expected.to_string(),
                    actual: "missing".to_owned(),
                }
            })?;
            if actual != expected {
                return Err(FamilyError::contract(field, expected, actual));
            }
        }
        let adapter_json: serde_json::Value = crate::families::support::read_json(&adapter_config)?;
        for (field, expected) in pinned_adapter_config() {
            let actual = resolve_config_field(&adapter_json, field).ok_or_else(|| {
                FamilyError::ContractMismatch {
                    field,
                    expected: expected.to_string(),
                    actual: "missing".to_owned(),
                }
            })?;
            if actual != expected {
                return Err(FamilyError::contract(field, expected, actual));
            }
        }

        if !base_checkpoint.is_file() {
            return Err(FamilyError::Io {
                path: base_checkpoint,
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "required pinned artifact is missing",
                ),
            });
        }
        verify_digest(&base_checkpoint, BASE_CHECKPOINT_SHA256)?;
        Ok(Self {
            base_checkpoint,
            adapter,
            head,
            tokenizer,
            profile,
        })
    }
}

/// Read the tensor names from a safetensors header without loading data.
fn safetensors_names(path: &Path) -> Result<Vec<String>, FamilyError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|source| FamilyError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let length = u64::from_le_bytes(length_bytes) as usize;
    let mut header_bytes = vec![0u8; length];
    file.read_exact(&mut header_bytes)
        .map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header: serde_json::Value =
        serde_json::from_slice(&header_bytes).map_err(|error| FamilyError::Json {
            path: path.to_path_buf(),
            source: error,
        })?;
    let mut names: Vec<String> = header
        .as_object()
        .ok_or_else(|| FamilyError::InvalidInput("safetensors header is not an object".into()))?
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect();
    names.sort();
    Ok(names)
}

/// A LoRA-merged view over the memory-mapped base checkpoint.
///
/// Targeted weights return `base + (B @ A) · scale` (FP32 math, widened to
/// the requested dtype); every other tensor passes through untouched. The
/// merge is computed once per load and re-applied on every read, so no
/// merged checkpoint is ever materialized or written.
struct MergedLoRABackend {
    inner: MmapedSafetensors,
    deltas: HashMap<String, Tensor>,
    names: HashSet<String>,
}

impl MergedLoRABackend {
    /// Pair the verified adapter with the verified base checkpoint.
    ///
    /// PEFT keys look like
    /// `base_model.model.layers.N.self_attn.q_proj.lora_A.weight`; the
    /// projection maps onto the base tensor
    /// `model.language_model.layers.N.self_attn.q_proj.weight` (the
    /// checkpoint keeps the multimodal container's `language_model` prefix).
    fn build(artifacts: &VerifiedArtifacts, device: &Device) -> Result<Self, FamilyError> {
        let names: HashSet<String> = safetensors_names(&artifacts.base_checkpoint)?
            .into_iter()
            .collect();
        let inner = unsafe { MmapedSafetensors::new(&artifacts.base_checkpoint)? };
        let adapter = unsafe { MmapedSafetensors::new(&artifacts.adapter)? };
        let adapter_names = safetensors_names(&artifacts.adapter)?;
        let mut pairs: BTreeMap<String, (Option<Tensor>, Option<Tensor>)> = BTreeMap::new();
        for name in adapter_names {
            let stem = name
                .strip_suffix(".weight")
                .ok_or_else(|| FamilyError::InvalidInput(format!("bad adapter key {name}")))?;
            let (projection, kind) = stem
                .rsplit_once('.')
                .ok_or_else(|| FamilyError::InvalidInput(format!("bad adapter key {name}")))?;
            let base_projection = projection
                .strip_prefix("base_model.model.")
                .ok_or_else(|| FamilyError::InvalidInput(format!("bad adapter key {name}")))?;
            let entry = pairs.entry(base_projection.to_owned()).or_default();
            let tensor = adapter.load(&name, device)?.to_dtype(DType::F32)?;
            match kind {
                "lora_A" => entry.0 = Some(tensor),
                "lora_B" => entry.1 = Some(tensor),
                other => {
                    return Err(FamilyError::InvalidInput(format!(
                        "unexpected adapter tensor kind `{other}` in {name}"
                    )));
                }
            }
        }
        let mut deltas = HashMap::with_capacity(pairs.len());
        for (projection, (a, b)) in pairs {
            let (Some(a), Some(b)) = (a, b) else {
                return Err(FamilyError::InvalidInput(format!(
                    "adapter projection `{projection}` is missing lora_A or lora_B"
                )));
            };
            // PEFT stores lora_A as (rank, in) and lora_B as (out, rank);
            // B @ A is (out, in), matching the checkpoint's weight layout.
            if a.dim(0)? != LORA_RANK || b.dim(1)? != LORA_RANK {
                return Err(FamilyError::contract(
                    "adapter.rank",
                    format!("rank {LORA_RANK}"),
                    format!("a {:?}, b {:?}", a.shape(), b.shape()),
                ));
            }
            let key = format!("model.language_model.{projection}.weight");
            let expected_shape = match names.get(&key) {
                Some(_) => inner.load(&key, device)?.shape().clone(),
                None => {
                    return Err(FamilyError::ContractMismatch {
                        field: "adapter.target",
                        expected: format!("base tensor `{key}` present"),
                        actual: "missing".to_owned(),
                    });
                }
            };
            let delta = (b.matmul(&a)? * LORA_SCALE)?;
            if delta.shape() != &expected_shape {
                return Err(FamilyError::contract(
                    "adapter.delta.shape",
                    format!("{expected_shape:?}"),
                    format!("{:?}", delta.shape()),
                ));
            }
            deltas.insert(key, delta);
        }
        Ok(Self {
            inner,
            deltas,
            names,
        })
    }
}

impl SimpleBackend for MergedLoRABackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _hints: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.inner.load(name, dev)?.to_dtype(dtype)?;
        let tensor = match self.deltas.get(name) {
            Some(delta) => Tensor::add(&tensor, &delta.to_device(dev)?.to_dtype(dtype)?)?,
            None => tensor,
        };
        if tensor.shape() != &s {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}

/// The pinned strands-decider model: merged Qwen3.5-2B backbone plus the
/// FP32 pointer head.
pub struct StrandsDeciderModel {
    backbone: TextBackbone,
    norm_weight: Tensor,
    norm_bias: Tensor,
    head_q: Linear,
    head_k: Linear,
    device: Device,
}

/// Apply the head's LayerNorm to one hidden row (torch `nn.LayerNorm`:
/// biased variance, torch-default epsilon).
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

impl StrandsDeciderModel {
    /// Load the verified base checkpoint, merge the adapter, and build the
    /// FP32 CPU model.
    pub fn load(artifacts: &VerifiedArtifacts) -> Result<Self, FamilyError> {
        Self::load_with_device(artifacts, Device::Cpu)
    }

    /// Load the verified base checkpoint and adapter onto `device`.
    ///
    /// The pointer-head forward executes on `device`; CUDA requires the
    /// `cuda` feature and fails closed when unavailable.
    pub fn load_with_device(
        artifacts: &VerifiedArtifacts,
        device: Device,
    ) -> Result<Self, FamilyError> {
        let layout = EmbeddingLayout::read(&artifacts.base_checkpoint, VOCAB_SIZE, HIDDEN_SIZE)
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        let embedding = Qwen35Embedding::from_layout(artifacts.base_checkpoint.clone(), layout);
        let merged = MergedLoRABackend::build(artifacts, &device)?;
        let backbone = TextBackbone::new_with_adapter_backend(
            embedding,
            Arc::new(merged),
            crate::qwen35::Qwen35Geometry::BASE_2B,
            device.clone(),
        );

        let head_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&artifacts.head),
                DType::F32,
                &device,
            )
        }?;
        let norm_weight = head_vb.get(HIDDEN_SIZE, "norm.weight")?.flatten_all()?;
        let norm_bias = head_vb.get(HIDDEN_SIZE, "norm.bias")?.flatten_all()?;
        let head_q = candle_nn::linear(HIDDEN_SIZE, POINTER_DIM, head_vb.pp("q"))?;
        let head_k = candle_nn::linear(HIDDEN_SIZE, POINTER_DIM, head_vb.pp("k"))?;
        Ok(Self {
            backbone,
            norm_weight,
            norm_bias,
            head_q,
            head_k,
            device,
        })
    }

    /// Pointer logits for one question row.
    ///
    /// The pooled position is the row's final `<answer>` token; each option
    /// scores from its own last-token hidden state. The head computes in
    /// FP32; the checkpoint's per-type temperature is applied by the caller.
    pub fn rows_logits(
        &self,
        row: &super::renderer::RenderedRow,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        control.check()?;
        let hidden = self
            .backbone
            .forward_hidden_with_check(&row.ids, || control.check())?;
        control.check()?;
        let norm_weight = self.norm_weight.to_vec1::<f32>()?;
        let norm_bias = self.norm_bias.to_vec1::<f32>()?;

        let answer_slot = &hidden[row.answer * HIDDEN_SIZE..(row.answer + 1) * HIDDEN_SIZE];
        let pooled = Tensor::new(
            layer_norm(answer_slot, &norm_weight, &norm_bias).as_slice(),
            &self.device,
        )?;
        let query = self.head_q.forward(&pooled.unsqueeze(0)?)?.squeeze(0)?;

        let mut option_rows = Vec::with_capacity(row.opts.len() * HIDDEN_SIZE);
        for &position in &row.opts {
            let slot = &hidden[position * HIDDEN_SIZE..(position + 1) * HIDDEN_SIZE];
            option_rows.extend_from_slice(&layer_norm(slot, &norm_weight, &norm_bias));
        }
        let option_count = row.opts.len();
        let keys = self.head_k.forward(
            &Tensor::new(option_rows.as_slice(), &self.device)?
                .reshape((option_count, HIDDEN_SIZE))?,
        )?;

        let query = query.to_vec1::<f32>()?;
        let keys = keys.to_vec2::<f32>()?;
        let scale = 1.0 / (POINTER_DIM as f64).sqrt();
        let logits: Vec<f64> = keys
            .iter()
            .map(|key| {
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
