//! Pinned Drex checkpoint loading: merged Qwen3.5 9B hybrid backbone (BF16
//! weights, FP32 compute) plus the Kev pointer head read from `head.pt`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use serde::Deserialize;

use crate::families::kev::model::QuestionRow;
use crate::families::support::{
    read_json, verify_digest, verify_sized_digest, FamilyControl, FamilyError,
};
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding, TextBackbone};

use super::{
    pinned_config, CHECKPOINT_SHARDS, CONFIG_SHA256, GEOMETRY, HEAD_PT_BYTES, HEAD_PT_SHA256,
    INDEX_SHA256, LICENSE_SHA256, POINTER_DIM, TOKENIZER_JSON_SHA256, VOCAB_SIZE,
};

const EMBED_TOKENS_TENSOR: &str = "model.language_model.embed_tokens.weight";
/// Top-level `head.pt` key holding the pointer-head state dict.
const HEAD_STATE_KEY: &str = "head";

#[derive(Debug, Deserialize)]
struct CheckpointIndex {
    weight_map: BTreeMap<String, String>,
}

/// Digest-verified artifacts required by the loader.
#[derive(Debug, Clone)]
pub struct VerifiedArtifacts {
    /// Verified shard paths in pinned order.
    pub shards: Vec<PathBuf>,
    /// Shard holding the embedding table.
    pub embedding_shard: PathBuf,
    /// Verified tokenizer.
    pub tokenizer: PathBuf,
    /// Verified pointer head (torch pickle, read without execution).
    pub head: PathBuf,
}

impl VerifiedArtifacts {
    /// Verify every pinned artifact under `model_root` in place and enforce
    /// the config contract. Small files are checked first so a wrong root
    /// fails before any multi-gigabyte shard is streamed.
    pub fn verify(model_root: &Path) -> Result<Self, FamilyError> {
        let config = model_root.join("config.json");
        let index = model_root.join("model.safetensors.index.json");
        let tokenizer = model_root.join("tokenizer.json");
        let license = model_root.join("LICENSE");
        let head = model_root.join("head.pt");
        verify_digest(&config, CONFIG_SHA256)?;
        verify_digest(&index, INDEX_SHA256)?;
        verify_digest(&tokenizer, TOKENIZER_JSON_SHA256)?;
        verify_digest(&license, LICENSE_SHA256)?;
        verify_sized_digest(&head, "head.pt", HEAD_PT_BYTES, HEAD_PT_SHA256, "head_size")?;

        let config_json: serde_json::Value = read_json(&config)?;
        for (field, expected) in pinned_config() {
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

        // Every tensor must live in a pinned shard; an index naming any
        // other file would load unverified bytes.
        let index: CheckpointIndex = read_json(&index)?;
        let pinned: Vec<&str> = CHECKPOINT_SHARDS.iter().map(|(name, _, _)| *name).collect();
        if let Some((tensor, shard)) = index
            .weight_map
            .iter()
            .find(|(_, shard)| !pinned.contains(&shard.as_str()))
        {
            return Err(FamilyError::ContractMismatch {
                field: "index.weight_map",
                expected: "every tensor in a pinned shard".to_owned(),
                actual: format!("{tensor} in {shard}"),
            });
        }
        let embedding_shard = index
            .weight_map
            .get(EMBED_TOKENS_TENSOR)
            .map(|shard| model_root.join(shard))
            .ok_or_else(|| {
                FamilyError::InvalidInput(format!(
                    "{EMBED_TOKENS_TENSOR} missing from checkpoint index"
                ))
            })?;

        let mut shards = Vec::with_capacity(CHECKPOINT_SHARDS.len());
        for (name, digest, bytes) in CHECKPOINT_SHARDS {
            let path = model_root.join(name);
            verify_sized_digest(&path, name, bytes, digest, "checkpoint_size")?;
            shards.push(path);
        }
        Ok(Self {
            shards,
            embedding_shard,
            tokenizer,
            head,
        })
    }
}

/// The pinned Drex model: BF16-stored backbone and FP32 pointer head.
pub struct DrexModel {
    backbone: TextBackbone,
    head_q: Linear,
    head_k: Linear,
    device: Device,
}

/// Read one FP32 head tensor of the expected shape from the verified pickle.
fn head_tensor(
    tensors: &candle_core::pickle::PthTensors,
    name: &str,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor, FamilyError> {
    let tensor = tensors
        .get(name)?
        .ok_or_else(|| FamilyError::ContractMismatch {
            field: "head.tensors",
            expected: format!("{name} present"),
            actual: "missing".to_owned(),
        })?;
    if tensor.dtype() != DType::F32 || tensor.dims() != shape {
        return Err(FamilyError::ContractMismatch {
            field: "head.tensors",
            expected: format!("{name} f32 {shape:?}"),
            actual: format!("{:?} {:?}", tensor.dtype(), tensor.dims()),
        });
    }
    Ok(tensor.to_device(device)?)
}

impl DrexModel {
    /// Build the CPU model from verified artifacts.
    pub fn load(artifacts: &VerifiedArtifacts) -> Result<Self, FamilyError> {
        let device = Device::Cpu;
        let hidden = GEOMETRY.hidden_size;
        let layout = EmbeddingLayout::read(&artifacts.embedding_shard, VOCAB_SIZE, hidden)
            .map_err(FamilyError::from)?;
        let embedding = Qwen35Embedding::from_layout(artifacts.embedding_shard.clone(), layout);
        let backbone = TextBackbone::new_with_dtype(
            embedding,
            artifacts.shards.clone(),
            GEOMETRY,
            DType::BF16,
            device.clone(),
        );

        // The digest was verified above, so these are known bytes. candle's
        // reader interprets the pickle opcodes into plain values and never
        // imports or calls the globals the stream names.
        let tensors = candle_core::pickle::PthTensors::new(&artifacts.head, Some(HEAD_STATE_KEY))?;
        if tensors.tensor_infos().len() != 4 {
            return Err(FamilyError::ContractMismatch {
                field: "head.tensors",
                expected: "4 tensors (q/k weight and bias)".to_owned(),
                actual: format!("{} tensors", tensors.tensor_infos().len()),
            });
        }
        let linear = |prefix: &str| -> Result<Linear, FamilyError> {
            let weight = head_tensor(
                &tensors,
                &format!("{prefix}.weight"),
                &[POINTER_DIM, hidden],
                &device,
            )?;
            let bias = head_tensor(&tensors, &format!("{prefix}.bias"), &[POINTER_DIM], &device)?;
            Ok(Linear::new(weight, Some(bias)))
        };
        let head_q = linear("q")?;
        let head_k = linear("k")?;
        Ok(Self {
            backbone,
            head_q,
            head_k,
            device,
        })
    }

    /// Pointer logits for one question row: the `<|fim_suffix|>` decide
    /// state is the query, each option's `<|box_end|>` state a key. The head
    /// runs in FP32 and the dot products accumulate in FP64.
    pub fn row_logits(
        &self,
        row: &QuestionRow,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        control.check()?;
        let hidden = GEOMETRY.hidden_size;
        let states = self
            .backbone
            .forward_hidden_with_check(&row.ids, || control.check())?;
        if states.len() != row.ids.len() * hidden {
            return Err(FamilyError::InvalidInput(
                "backbone output width does not match the pinned geometry".to_owned(),
            ));
        }
        control.check()?;
        let slot = |position: usize| &states[position * hidden..(position + 1) * hidden];

        let decide = Tensor::new(slot(row.decide), &self.device)?.unsqueeze(0)?;
        let query = self.head_q.forward(&decide)?.squeeze(0)?.to_vec1::<f32>()?;
        let mut option_rows = Vec::with_capacity(row.opts.len() * hidden);
        for &position in &row.opts {
            option_rows.extend_from_slice(slot(position));
        }
        let keys = self
            .head_k
            .forward(
                &Tensor::new(option_rows.as_slice(), &self.device)?
                    .reshape((row.opts.len(), hidden))?,
            )?
            .to_vec2::<f32>()?;

        let scale = 1.0 / (POINTER_DIM as f64).sqrt();
        let logits: Vec<f64> = keys
            .iter()
            .map(|key| {
                key.iter()
                    .zip(&query)
                    .map(|(key, query)| f64::from(*key) * f64::from(*query))
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
