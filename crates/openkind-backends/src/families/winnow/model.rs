//! Pinned winnow checkpoint loading: base Qwen2.5 + LoRA merge.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};

use crate::families::support::{verify_digest, FamilyError};

use super::{ADAPTER_SHA256, LORA_RANK, LORA_SCALE, TOKENIZER_JSON_SHA256};

/// The pinned winnow model: base weights with the LoRA delta merged at load,
/// FP32 on CPU. Merging once at load keeps the forward path identical to the
/// un-LoRA'd decoder-letter family.
pub struct WinnowModel {
    // `ModelForCausalLM` accumulates KV cache inside `forward(&mut self)`;
    // evaluation is serialized through this mutex and the cache is cleared
    // after every routing pass.
    model: Mutex<ModelForCausalLM>,
}

/// Digest-verified artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified base checkpoint.
    pub checkpoint: std::path::PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: std::path::PathBuf,
    /// Path to the verified LoRA adapter.
    pub adapter: std::path::PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place.
    ///
    /// The base checkpoint digest is the same pinned checkpoint the
    /// decoder-logit-letter family pins: both profiles share the base model.
    pub fn verify(
        model_root: &Path,
        adapter_path: &Path,
        checkpoint_sha256: &str,
    ) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join("model.safetensors");
        let tokenizer = model_root.join("tokenizer.json");
        for path in [&checkpoint, &tokenizer, adapter_path] {
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
        verify_digest(&tokenizer, TOKENIZER_JSON_SHA256)?;
        verify_digest(&checkpoint, checkpoint_sha256)?;
        verify_digest(adapter_path, ADAPTER_SHA256)?;
        Ok(Self {
            checkpoint,
            tokenizer,
            adapter: adapter_path.to_path_buf(),
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

/// Load every base tensor into memory as FP32.
fn base_tensors(
    checkpoint: &Path,
    device: &Device,
) -> Result<std::collections::HashMap<String, Tensor>, FamilyError> {
    let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(checkpoint)? };
    let mut tensors = std::collections::HashMap::new();
    for name in safetensors_names(checkpoint)? {
        let tensor = mmap.load(&name, device)?.to_dtype(DType::F32)?;
        tensors.insert(name, tensor);
    }
    Ok(tensors)
}

impl WinnowModel {
    /// Load the verified base checkpoint, merge the adapter, and build the
    /// FP32 CPU model.
    pub fn load(artifacts: &VerifiedArtifacts, config: &Config) -> Result<Self, FamilyError> {
        let device = Device::Cpu;
        let mut tensors = base_tensors(&artifacts.checkpoint, &device)?;

        // Collect the adapter lora_a/lora_b pairs.
        let adapter =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(&artifacts.adapter)? };
        let adapter_names = safetensors_names(&artifacts.adapter)?;
        let mut pairs: std::collections::BTreeMap<String, (Option<Tensor>, Option<Tensor>)> =
            std::collections::BTreeMap::new();
        for name in adapter_names {
            let tensor = adapter.load(&name, &device)?.to_dtype(DType::F32)?;
            let (tensor_name, kind) = name
                .rsplit_once('.')
                .ok_or_else(|| FamilyError::InvalidInput(format!("bad adapter key {name}")))?;
            let entry = pairs.entry(tensor_name.to_owned()).or_default();
            match kind {
                "lora_a" => entry.0 = Some(tensor),
                "lora_b" => entry.1 = Some(tensor),
                other => {
                    return Err(FamilyError::InvalidInput(format!(
                        "unexpected adapter tensor kind `{other}` in {name}"
                    )));
                }
            }
        }

        // Merge each delta into the base tensors.
        for (projection, (a, b)) in pairs {
            let (Some(a), Some(b)) = (a, b) else {
                return Err(FamilyError::InvalidInput(format!(
                    "adapter projection `{projection}` is missing lora_a or lora_b"
                )));
            };
            if a.dim(1)? != LORA_RANK || b.dim(0)? != LORA_RANK {
                return Err(FamilyError::contract(
                    "adapter.rank",
                    format!("rank {LORA_RANK}"),
                    format!("a {:?}, b {:?}", a.shape(), b.shape()),
                ));
            }
            // lora_a (in, rank) @ lora_b (rank, out) = (in, out); the candle
            // Linear layout is (out, in), so the delta is transposed.
            let delta = (a.matmul(&b)? * LORA_SCALE)?.t()?;
            let key = format!("{projection}.weight");
            let base_weight =
                tensors
                    .get_mut(&key)
                    .ok_or_else(|| FamilyError::ContractMismatch {
                        field: "adapter.target",
                        expected: format!("base tensor `{key}` present"),
                        actual: "missing".to_owned(),
                    })?;
            if delta.shape() != base_weight.shape() {
                return Err(FamilyError::contract(
                    "adapter.delta.shape",
                    format!("{:?}", base_weight.shape()),
                    format!("{:?}", delta.shape()),
                ));
            }
            let updated: Tensor = Tensor::add(base_weight, &delta)?;
            *base_weight = updated;
        }

        let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = ModelForCausalLM::new(config, vb)?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    /// Forward the routing prompt once and return the next-token logits at
    /// the letter slots. Nothing is sampled and no continuation exists.
    pub fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<Vec<f64>, FamilyError> {
        let input = Tensor::new(prompt_ids, &Device::Cpu)?.unsqueeze(0)?;
        let logits = {
            let mut model = self
                .model
                .lock()
                .map_err(|_| FamilyError::InvalidInput("winnow model lock poisoned".to_owned()))?;
            let logits = model.forward(&input, 0)?;
            model.clear_kv_cache();
            logits
        };
        let logits = logits.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        Ok(letter_ids
            .iter()
            .map(|token| f64::from(logits[*token as usize]))
            .collect())
    }
}
