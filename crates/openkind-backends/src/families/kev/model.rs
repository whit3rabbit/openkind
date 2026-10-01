//! Pinned kev checkpoint loading: base Qwen3 + LoRA merge + pointer head.

use std::collections::BTreeMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{linear, Linear, Module};

use crate::families::support::{verify_digest, FamilyError};

use super::arch::{config_from_pinned, Qwen3Model};
use super::{
    pinned_base_config, ADAPTER_SHA256, BASE_CHECKPOINT_SHA256, BASE_CONFIG_SHA256,
    HEAD_PT_SOURCE_SHA256, HEAD_SHA256, LORA_RANK, LORA_SCALE, POINTER_DIM, TOKENIZER_JSON_SHA256,
};

/// One encoded question row plus its readout offsets (relative to the row
/// start): the `<|fim_suffix|>` decide position and one `</opt>` position
/// per option.
#[derive(Debug, Clone)]
pub struct QuestionRow {
    pub ids: Vec<u32>,
    pub decide: usize,
    pub opts: Vec<usize>,
}

/// The pinned kev model: merged Qwen3 backbone plus the FP32 pointer head.
pub struct KevModel {
    body: Qwen3Model,
    head_q: Linear,
    head_k: Linear,
    device: Device,
}

/// Digest-verified artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified base checkpoint.
    pub base_checkpoint: std::path::PathBuf,
    /// Path to the verified adapter.
    pub adapter: std::path::PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: std::path::PathBuf,
    /// Path to the verified pointer head.
    pub head: std::path::PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place.
    pub fn verify(model_root: &Path, base_root: &Path) -> Result<Self, FamilyError> {
        let base_checkpoint = base_root.join("model.safetensors");
        let base_config = base_root.join("config.json");
        let adapter = model_root.join("adapter_model.safetensors");
        let tokenizer = model_root.join("tokenizer.json");
        let head_safetensors = model_root.join("head.safetensors");
        let head_pt = model_root.join("head.pt");
        for path in [&base_checkpoint, &base_config, &adapter, &tokenizer] {
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
        let head = if head_safetensors.is_file() {
            verify_digest(&head_safetensors, HEAD_SHA256)?;
            head_safetensors
        } else if head_pt.is_file() {
            // The original torch-pickle head: the loader cannot execute a
            // pickle, so its presence with the pinned digest documents
            // provenance but the converted head.safetensors is required.
            verify_digest(&head_pt, HEAD_PT_SOURCE_SHA256)?;
            return Err(FamilyError::InvalidInput(
                "model root holds the original `head.pt`; convert it to \
                 `head.safetensors` once with safetensors.torch.save_file \
                 (see the family module docs) and reload"
                    .to_owned(),
            ));
        } else {
            return Err(FamilyError::Io {
                path: head_safetensors,
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "required pinned artifact is missing",
                ),
            });
        };
        // Cheap contract checks first; large checkpoints stream last.
        verify_digest(&base_config, BASE_CONFIG_SHA256)?;
        verify_digest(&tokenizer, TOKENIZER_JSON_SHA256)?;
        verify_digest(&adapter, ADAPTER_SHA256)?;
        verify_digest(&base_checkpoint, BASE_CHECKPOINT_SHA256)?;
        // Contract-check the pinned base config values.
        let config_json: serde_json::Value = crate::families::support::read_json(&base_config)?;
        for (field, expected) in pinned_base_config() {
            let actual = config_json
                .get(field)
                .ok_or_else(|| FamilyError::ContractMismatch {
                    field,
                    expected: expected.to_string(),
                    actual: "missing".to_owned(),
                })?;
            if actual != &expected {
                return Err(FamilyError::contract(field, expected, actual));
            }
        }
        Ok(Self {
            base_checkpoint,
            adapter,
            tokenizer,
            head,
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

impl KevModel {
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
        let mmap = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(&artifacts.base_checkpoint)?
        };
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        for name in safetensors_names(&artifacts.base_checkpoint)? {
            // The Qwen3 base stores bf16; the family execution path is FP32.
            let tensor = mmap.load(&name, &device)?.to_dtype(DType::F32)?;
            tensors.insert(name, tensor);
        }

        // Collect the adapter lora_A/lora_B pairs. PEFT keys look like
        // `base_model.model.layers.N.self_attn.q_proj.lora_A.weight`; the
        // projection maps onto the base tensor
        // `model.layers.N.self_attn.q_proj.weight`.
        let adapter =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(&artifacts.adapter)? };
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
            let tensor = adapter.load(&name, &device)?.to_dtype(DType::F32)?;
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

        // Merge each delta into the base tensors: W += (B @ A) * scale,
        // transposed into the candle Linear layout.
        for (projection, (a, b)) in pairs {
            let (Some(a), Some(b)) = (a, b) else {
                return Err(FamilyError::InvalidInput(format!(
                    "adapter projection `{projection}` is missing lora_A or lora_B"
                )));
            };
            // PEFT stores lora_A as (rank, in) and lora_B as (out, rank);
            // B @ A is (out, in), already the candle Linear orientation.
            if a.dim(0)? != LORA_RANK || b.dim(1)? != LORA_RANK {
                return Err(FamilyError::contract(
                    "adapter.rank",
                    format!("rank {LORA_RANK}"),
                    format!("a {:?}, b {:?}", a.shape(), b.shape()),
                ));
            }
            let delta = (b.matmul(&a)? * LORA_SCALE)?;
            let key = format!("model.{projection}.weight");
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
        let body = Qwen3Model::load(
            &config_from_pinned(crate::families::kev::MAX_ROW_TOKENS),
            vb.pp("model"),
        )?;

        let head_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&artifacts.head),
                DType::F32,
                &device,
            )
        }?;
        let head_q = linear(1024, POINTER_DIM, head_vb.pp("q"))?;
        let head_k = linear(1024, POINTER_DIM, head_vb.pp("k"))?;
        Ok(Self {
            body,
            head_q,
            head_k,
            device,
        })
    }

    /// Pointer logits for one chunk of question rows.
    ///
    /// `rows` is the padded row-major id buffer, `row_lens` the real length
    /// of each row, and `readouts` the per-row `(decide, opts)` offsets.
    /// Returns one logit vector per row in chunk order. The checkpoint's
    /// temperature is applied by the caller.
    pub fn rows_logits(
        &self,
        rows: &[u32],
        row_lens: &[usize],
        readouts: &[(usize, Vec<usize>)],
    ) -> Result<Vec<Vec<f64>>, FamilyError> {
        let hidden = self
            .body
            .forward_rows(rows, row_lens, &self.device)?
            .to_dtype(DType::F32)?;
        let scale = 1.0 / (POINTER_DIM as f64).sqrt();
        let mut out = Vec::with_capacity(row_lens.len());
        for (row_index, (decide, opts)) in readouts.iter().enumerate() {
            let h_decide = hidden.i((row_index, *decide))?.unsqueeze(0)?;
            let q_decide = self.head_q.forward(&h_decide)?;
            let opt_indices = Tensor::from_slice(
                &(opts.iter().map(|&p| p as u32).collect::<Vec<u32>>()),
                opts.len(),
                &self.device,
            )?;
            let h_opts = hidden.i(row_index)?.index_select(&opt_indices, 0)?;
            let k_opts = self.head_k.forward(&h_opts)?;
            // k(h_opts) · q(h_decide): one dot product per option.
            let q_row = q_decide.broadcast_as(k_opts.shape())?;
            let logits = k_opts.broadcast_mul(&q_row)?.sum(candle_core::D::Minus1)?;
            let logits = logits.to_vec1::<f32>()?;
            out.push(
                logits
                    .into_iter()
                    .map(|value| f64::from(value) * scale)
                    .collect(),
            );
        }
        Ok(out)
    }
}
