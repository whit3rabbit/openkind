//! Digest-verified artifact loading and the candle BF16 execution model for
//! the Clef family.
//!
//! Large checkpoint shards are verified in the read-only source directory
//! and memory-mapped in place; nothing is staged. The BF16 profile maps its
//! shards in native width and widens weights per use inside the shared
//! Qwen3.5 layer kernels, so peak resident memory stays close to one layer
//! of widened weights plus activations.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::DType;
use serde::Deserialize;

use crate::families::support::{read_json, verify_digest, FamilyControl, FamilyError};
use crate::qwen35::Qwen35Embedding;

use super::head::{JointSchemaHead, LexicalLookup};
use super::renderer::EncodedRecord;
use super::ClefProfile;

/// Execution backend of one loaded Clef model.
///
/// Every backend answers the same contract: one full-sequence forward over
/// the encoded record, returning one FP64 logit vector per question.
pub trait ClefExecutionModel: Send + Sync {
    /// Run the backbone and joint head for one encoded record.
    fn evaluate_record(
        &self,
        encoded: &EncodedRecord,
        control: &FamilyControl,
    ) -> Result<Vec<Vec<f64>>, FamilyError>;
}

impl ClefExecutionModel for ClefModel {
    fn evaluate_record(
        &self,
        encoded: &EncodedRecord,
        control: &FamilyControl,
    ) -> Result<Vec<Vec<f64>>, FamilyError> {
        Self::evaluate_record(self, encoded, control)
    }
}

/// Digest-verified, contract-checked artifacts required by the loader.
#[derive(Debug, Clone)]
pub struct VerifiedArtifacts {
    /// Verified model root.
    pub model_root: PathBuf,
    profile: &'static ClefProfile,
}

/// Subset of the pinned `config.json` the loader enforces before load.
#[derive(Debug, Deserialize)]
struct PinnedTextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    full_attention_interval: usize,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct PinnedConfig {
    text_config: PinnedTextConfig,
}

impl VerifiedArtifacts {
    /// Verify every pinned artifact of `profile` under `model_root` in
    /// place, and enforce the config contract.
    pub fn verify(
        model_root: impl AsRef<Path>,
        profile: &'static ClefProfile,
    ) -> Result<Self, FamilyError> {
        let model_root = model_root.as_ref().to_path_buf();
        for (name, digest, bytes) in profile.checkpoint_shards {
            let path = model_root.join(name);
            let length = std::fs::metadata(&path)
                .map_err(|source| FamilyError::Io {
                    path: path.clone(),
                    source,
                })?
                .len();
            if length != *bytes {
                return Err(FamilyError::ContractMismatch {
                    field: "checkpoint_size",
                    expected: format!("{name} is {bytes} bytes"),
                    actual: format!("{length} bytes"),
                });
            }
            verify_digest(&path, digest)?;
        }
        let (head_name, head_digest, head_bytes) = profile.joint_head;
        let head_path = model_root.join(head_name);
        let head_length = std::fs::metadata(&head_path)
            .map_err(|source| FamilyError::Io {
                path: head_path.clone(),
                source,
            })?
            .len();
        if head_length != head_bytes {
            return Err(FamilyError::ContractMismatch {
                field: "joint_head_size",
                expected: format!("{head_name} is {head_bytes} bytes"),
                actual: format!("{head_length} bytes"),
            });
        }
        verify_digest(&head_path, head_digest)?;
        verify_digest(
            &model_root.join(profile.tokenizer_path),
            profile.tokenizer_json_sha256,
        )?;
        verify_digest(
            &model_root.join(profile.config_path),
            profile.config_json_sha256,
        )?;
        let artifacts = Self {
            model_root,
            profile,
        };
        artifacts.enforce_config_contract()?;
        Ok(artifacts)
    }

    /// The pinned profile these artifacts verify against.
    pub fn profile(&self) -> &'static ClefProfile {
        self.profile
    }

    /// Reject a checkpoint whose geometry does not match the pinned
    /// profile: loading with mismatched widths would corrupt every readout.
    fn enforce_config_contract(&self) -> Result<(), FamilyError> {
        let config: PinnedConfig = read_json(&self.model_root.join(self.profile.config_path))?;
        let geometry = self.profile.geometry;
        let text = &config.text_config;
        let checks: [(&'static str, String, String); 12] = [
            (
                "hidden_size",
                geometry.hidden_size.to_string(),
                text.hidden_size.to_string(),
            ),
            (
                "num_hidden_layers",
                geometry.layer_count.to_string(),
                text.num_hidden_layers.to_string(),
            ),
            (
                "intermediate_size",
                geometry.intermediate_size.to_string(),
                text.intermediate_size.to_string(),
            ),
            (
                "linear_num_key_heads",
                geometry.key_heads.to_string(),
                text.linear_num_key_heads.to_string(),
            ),
            (
                "linear_num_value_heads",
                geometry.value_heads.to_string(),
                text.linear_num_value_heads.to_string(),
            ),
            (
                "linear_key_head_dim",
                geometry.head_dim.to_string(),
                text.linear_key_head_dim.to_string(),
            ),
            (
                "linear_value_head_dim",
                geometry.head_dim.to_string(),
                text.linear_value_head_dim.to_string(),
            ),
            (
                "linear_conv_kernel_dim",
                geometry.conv_kernel.to_string(),
                text.linear_conv_kernel_dim.to_string(),
            ),
            (
                "num_attention_heads",
                geometry.attention_heads.to_string(),
                text.num_attention_heads.to_string(),
            ),
            (
                "num_key_value_heads",
                geometry.kv_heads.to_string(),
                text.num_key_value_heads.to_string(),
            ),
            (
                "head_dim",
                geometry.attention_head_dim.to_string(),
                text.head_dim.to_string(),
            ),
            (
                "full_attention_interval",
                geometry.full_attention_interval.to_string(),
                text.full_attention_interval.to_string(),
            ),
        ];
        for (field, expected, actual) in checks {
            if expected != actual {
                return Err(FamilyError::ContractMismatch {
                    field,
                    expected,
                    actual,
                });
            }
        }
        // The vocabulary enters through the embedding and output-embedding
        // readers; a mismatch would misread every row.
        let index: CheckpointIndex =
            read_json(&self.model_root.join("model.safetensors.index.json"))?;
        let vocab = config.text_config.vocab_size;
        let hidden = geometry.hidden_size;
        for tensor in [EMBED_TOKENS_TENSOR, LM_HEAD_TENSOR] {
            let shard = index.weight_map.get(tensor).ok_or_else(|| {
                FamilyError::InvalidInput(format!("{tensor} missing from checkpoint index"))
            })?;
            let layout = TensorLayout::read(&self.model_root.join(shard), tensor)?;
            layout.rows(vocab, hidden)?;
        }
        Ok(())
    }

    /// The verified shard paths in pinned order.
    fn shard_paths(&self) -> Vec<PathBuf> {
        self.profile
            .checkpoint_shards
            .iter()
            .map(|(name, _, _)| self.model_root.join(name))
            .collect()
    }
}

const EMBED_TOKENS_TENSOR: &str = "model.language_model.embed_tokens.weight";
const LM_HEAD_TENSOR: &str = "lm_head.weight";

#[derive(Debug, Deserialize)]
struct CheckpointIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl TensorMetadata {
    fn from_value(value: &serde_json::Value, tensor: &str) -> Result<Self, FamilyError> {
        serde_json::from_value(value.clone()).map_err(|error| FamilyError::Json {
            path: std::path::PathBuf::from(tensor),
            source: error,
        })
    }
}

/// Safetensors header reader locating one tensor's data window in a shard.
#[derive(Debug, Clone)]
struct TensorLayout {
    shard_path: PathBuf,
    data_start: u64,
    dtype: String,
    shape: Vec<usize>,
}

impl TensorLayout {
    fn read(shard_path: &Path, tensor: &str) -> Result<Self, FamilyError> {
        let mut file = File::open(shard_path).map_err(|source| FamilyError::Io {
            path: shard_path.to_path_buf(),
            source,
        })?;
        let mut length_bytes = [0_u8; 8];
        file.read_exact(&mut length_bytes)
            .map_err(|source| FamilyError::Io {
                path: shard_path.to_path_buf(),
                source,
            })?;
        let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
            .map_err(|_| FamilyError::InvalidInput("safetensors header too large".into()))?;
        if header_len == 0 || header_len > 16 * 1024 * 1024 {
            return Err(FamilyError::InvalidInput(format!(
                "invalid safetensors header length {header_len} in {}",
                shard_path.display()
            )));
        }
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)
            .map_err(|source| FamilyError::Io {
                path: shard_path.to_path_buf(),
                source,
            })?;
        let tensors: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)
            .map_err(|error| FamilyError::Json {
                path: shard_path.to_path_buf(),
                source: error,
            })?;
        let metadata = TensorMetadata::from_value(
            tensors.get(tensor).ok_or_else(|| {
                FamilyError::InvalidInput(format!(
                    "{tensor} missing from safetensors header of {}",
                    shard_path.display()
                ))
            })?,
            tensor,
        )?;
        Ok(Self {
            shard_path: shard_path.to_path_buf(),
            // The tensor data window starts after the 8-byte length prefix
            // and header, offset by the tensor's own data offset.
            data_start: 8
                + u64::try_from(header_len).unwrap_or(u64::MAX)
                + metadata.data_offsets[0],
            dtype: metadata.dtype.clone(),
            shape: metadata.shape.clone(),
        })
    }

    fn rows(&self, vocab: usize, hidden: usize) -> Result<(), FamilyError> {
        if self.dtype != "BF16" {
            return Err(FamilyError::ContractMismatch {
                field: "embedding_dtype",
                expected: "BF16".into(),
                actual: self.dtype.clone(),
            });
        }
        if self.shape != [vocab, hidden] {
            return Err(FamilyError::ContractMismatch {
                field: "embedding_shape",
                expected: format!("[{vocab}, {hidden}]"),
                actual: format!("{:?}", self.shape),
            });
        }
        Ok(())
    }
}

/// Seek-based BF16 row reader over one `[rows, width]` safetensors tensor.
///
/// Used for the input-embedding table and the untied output-embedding rows
/// the joint head's lexical prior consumes; only the addressed rows leave
/// the page cache.
#[derive(Debug, Clone)]
pub(crate) struct RowReader {
    layout: TensorLayout,
    rows: usize,
    width: usize,
}

impl RowReader {
    fn load(
        model_root: &Path,
        index: &CheckpointIndex,
        tensor: &str,
        rows: usize,
        width: usize,
    ) -> Result<Self, FamilyError> {
        let shard = index.weight_map.get(tensor).ok_or_else(|| {
            FamilyError::InvalidInput(format!("{tensor} missing from checkpoint index"))
        })?;
        let layout = TensorLayout::read(&model_root.join(shard), tensor)?;
        layout.rows(rows, width)?;
        Ok(Self {
            layout,
            rows,
            width,
        })
    }

    /// Read the BF16 rows for `row_ids`, widened to FP32, row-major.
    fn read_rows(&self, row_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let row_bytes = self.width * 2;
        let mut file = File::open(&self.layout.shard_path).map_err(|source| FamilyError::Io {
            path: self.layout.shard_path.clone(),
            source,
        })?;
        let mut raw = vec![0_u8; row_bytes];
        let mut values = Vec::with_capacity(row_ids.len() * self.width);
        for &row_id in row_ids {
            let row = usize::try_from(row_id)
                .map_err(|_| FamilyError::InvalidInput(format!("row id {row_id} overflow")))?;
            if row >= self.rows {
                return Err(FamilyError::InvalidInput(format!(
                    "row id {row} exceeds {} rows",
                    self.rows
                )));
            }
            let offset = self
                .layout
                .data_start
                .checked_add(u64::try_from(row * row_bytes).unwrap_or(u64::MAX))
                .ok_or_else(|| FamilyError::InvalidInput("row offset overflow".into()))?;
            file.seek(SeekFrom::Start(offset))
                .and_then(|_| file.read_exact(&mut raw))
                .map_err(|source| FamilyError::Io {
                    path: self.layout.shard_path.clone(),
                    source,
                })?;
            for index in 0..self.width {
                let byte = index * 2;
                let bits = u32::from(u16::from_le_bytes([raw[byte], raw[byte + 1]])) << 16;
                values.push(f32::from_bits(bits));
            }
        }
        Ok(values)
    }
}

/// The digest-verified Clef model: BF16 candle backbone, joint head, and
/// output-embedding lexical reader.
pub struct ClefModel {
    text: crate::qwen35::TextBackbone,
    lexical: RowReader,
    head: JointSchemaHead,
    geometry: crate::qwen35::Qwen35Geometry,
    device: candle_core::Device,
}

impl ClefModel {
    /// Build the BF16 model from verified artifacts.
    pub(crate) fn load(artifacts: &VerifiedArtifacts) -> Result<Self, FamilyError> {
        Self::load_with_device(artifacts, candle_core::Device::Cpu)
    }

    /// Build the BF16 model on an explicit device.
    pub(crate) fn load_with_device(
        artifacts: &VerifiedArtifacts,
        device: candle_core::Device,
    ) -> Result<Self, FamilyError> {
        let profile = artifacts.profile();
        let geometry = profile.geometry;
        let index: CheckpointIndex =
            read_json(&artifacts.model_root.join("model.safetensors.index.json"))?;
        let vocab = {
            let config: PinnedConfig = read_json(&artifacts.model_root.join(profile.config_path))?;
            config.text_config.vocab_size
        };
        let embedding_shard = index
            .weight_map
            .get(EMBED_TOKENS_TENSOR)
            .ok_or_else(|| {
                FamilyError::InvalidInput(format!(
                    "{EMBED_TOKENS_TENSOR} missing from checkpoint index"
                ))
            })?
            .clone();
        let embedding_layout = crate::qwen35::EmbeddingLayout::read(
            &artifacts.model_root.join(&embedding_shard),
            vocab,
            geometry.hidden_size,
        )
        .map_err(FamilyError::from)?;
        let embedding = Qwen35Embedding::from_layout(
            artifacts.model_root.join(&embedding_shard),
            embedding_layout,
        );
        let lexical = RowReader::load(
            &artifacts.model_root,
            &index,
            LM_HEAD_TENSOR,
            vocab,
            geometry.hidden_size,
        )?;
        let text = crate::qwen35::TextBackbone::new_with_dtype(
            embedding,
            artifacts.shard_paths(),
            geometry,
            DType::BF16,
            device.clone(),
        );
        let head_path = artifacts.model_root.join(profile.joint_head.0);
        let head = JointSchemaHead::load(&profile.joint_head_config, &head_path, &device)?;
        Ok(Self {
            text,
            lexical,
            head,
            geometry,
            device,
        })
    }

    /// Run the backbone and joint head for one encoded record.
    ///
    /// Returns one FP64 logit vector per encoded question, in encoded order.
    pub(crate) fn evaluate_record(
        &self,
        encoded: &EncodedRecord,
        control: &crate::families::support::FamilyControl,
    ) -> Result<Vec<Vec<f64>>, FamilyError> {
        control.check()?;
        let hidden = self
            .text
            .forward_hidden_with_check(&encoded.input_ids, || control.check())?;
        let token_count = encoded.input_ids.len();
        if hidden.len() != token_count * self.geometry.hidden_size {
            return Err(FamilyError::InvalidInput(
                "backbone output width does not match the pinned geometry".to_owned(),
            ));
        }
        control.check()?;
        let result =
            self.head
                .forward(&hidden, token_count, encoded, &self.lexical, &self.device)?;
        control.check()?;
        Ok(result)
    }
}

impl LexicalLookup for RowReader {
    fn rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        self.read_rows(token_ids)
    }
}
