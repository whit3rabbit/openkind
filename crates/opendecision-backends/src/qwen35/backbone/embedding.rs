use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::super::Qwen35Error;

mod layout;

pub(crate) use layout::EmbeddingLayout;
use layout::{file_len, sha256_file, verify_small_artifact, CheckpointIndex};

pub(super) const CONFIG_FILE: &str = "config.json";
pub(super) const MODEL_INDEX_FILE: &str = "model.safetensors.index.json";
pub(super) const EMBEDDING_TENSOR: &str = "model.language_model.embed_tokens.weight";
pub(super) const EMBEDDING_SHARD: &str = "model.safetensors-00001-of-00002.safetensors";
pub(super) const DECODER_SHARD: &str = "model.safetensors-00002-of-00002.safetensors";
pub(crate) const CONFIG_SHA256: &str =
    "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670";
pub(crate) const MODEL_INDEX_SHA256: &str =
    "eae340074abb0a5f31a6621f7ae8e8248a7c1790df04a722c4e4b70c2a6d1dbb";
pub(super) const EMBEDDING_SHARD_SHA256: &str =
    "df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712";
pub(super) const EMBEDDING_SHARD_BYTES: u64 = 5_329_398_712;
pub(super) const DECODER_SHARD_SHA256: &str =
    "590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef";
pub(super) const DECODER_SHARD_BYTES: u64 = 3_990_429_344;
pub(super) const CHECKPOINT_BYTES: u64 = 9_319_737_856;
pub(super) const VOCAB_SIZE: usize = 248_320;
pub(super) const HIDDEN_SIZE: usize = 2_560;

/// Exact FP32 embedding output for one token sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingOutput {
    token_count: usize,
    hidden_size: usize,
    values: Vec<f32>,
}

impl EmbeddingOutput {
    /// Number of embedded input tokens.
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.token_count
    }

    /// Width of every token embedding.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Row-major `[token_count, hidden_size]` FP32 values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// FP32 embedding for the final token.
    #[must_use]
    pub fn last_token(&self) -> &[f32] {
        let start = (self.token_count - 1) * self.hidden_size;
        &self.values[start..]
    }
}

/// Verified seek-based access to the selected Qwen3.5 input embedding table.
///
/// Loading hashes the exact config, sharded index, and 5.3 GB shard containing
/// the tied embedding table. Evaluation then reads only the rows named by the
/// input IDs and widens the checkpoint BF16 values to FP32 exactly.
#[derive(Debug, Clone)]
pub struct Qwen35Embedding {
    shard_path: PathBuf,
    tensor_data_start: u64,
    vocab_size: usize,
    hidden_size: usize,
}

impl Qwen35Embedding {
    /// Load the pinned checkpoint from an already-downloaded local snapshot.
    ///
    /// This method never downloads model assets. The directory must contain
    /// the immutable revision's `config.json`, safetensors index, and first
    /// model shard.
    pub fn load(checkpoint_root: impl AsRef<Path>) -> Result<Self, Qwen35Error> {
        let root = checkpoint_root.as_ref();
        verify_small_artifact(&root.join(CONFIG_FILE), CONFIG_FILE, CONFIG_SHA256)?;
        let index_bytes = verify_small_artifact(
            &root.join(MODEL_INDEX_FILE),
            MODEL_INDEX_FILE,
            MODEL_INDEX_SHA256,
        )?;
        let index: CheckpointIndex =
            serde_json::from_slice(&index_bytes).map_err(|source| Qwen35Error::Json {
                path: root.join(MODEL_INDEX_FILE),
                source,
            })?;
        if index.metadata.total_size != CHECKPOINT_BYTES {
            return Err(Qwen35Error::InvalidInput(format!(
                "checkpoint index expected {CHECKPOINT_BYTES} parameter bytes, found {}",
                index.metadata.total_size
            )));
        }
        let shard_name =
            index
                .weight_map
                .get(EMBEDDING_TENSOR)
                .ok_or_else(|| Qwen35Error::InvalidTensor {
                    name: EMBEDDING_TENSOR.to_owned(),
                    message: "missing from checkpoint index".to_owned(),
                })?;
        if shard_name != EMBEDDING_SHARD {
            return Err(Qwen35Error::InvalidTensor {
                name: EMBEDDING_TENSOR.to_owned(),
                message: format!("expected shard {EMBEDDING_SHARD}, found {shard_name}"),
            });
        }

        let shard_path = root.join(shard_name);
        let file_size = file_len(&shard_path)?;
        if file_size != EMBEDDING_SHARD_BYTES {
            return Err(Qwen35Error::InvalidTensor {
                name: EMBEDDING_TENSOR.to_owned(),
                message: format!("expected shard size {EMBEDDING_SHARD_BYTES}, found {file_size}"),
            });
        }
        let layout = EmbeddingLayout::read(&shard_path, VOCAB_SIZE, HIDDEN_SIZE)?;
        let actual_digest = sha256_file(&shard_path)?;
        if actual_digest != EMBEDDING_SHARD_SHA256 {
            return Err(Qwen35Error::DigestMismatch {
                path: shard_name.clone(),
                expected: EMBEDDING_SHARD_SHA256.to_owned(),
                actual: actual_digest,
            });
        }
        Ok(Self::from_layout(shard_path, layout))
    }

    /// Embed an exact token-ID sequence using FP32 host values.
    pub fn embed(&self, input_ids: &[u32]) -> Result<EmbeddingOutput, Qwen35Error> {
        if input_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(
                "embedding input must contain at least one token".to_owned(),
            ));
        }
        let value_count = input_ids
            .len()
            .checked_mul(self.hidden_size)
            .ok_or_else(|| Qwen35Error::InvalidInput("embedding output is too large".to_owned()))?;
        let row_bytes = self
            .hidden_size
            .checked_mul(2)
            .ok_or_else(|| Qwen35Error::InvalidInput("embedding row is too large".to_owned()))?;
        let mut file = File::open(&self.shard_path).map_err(|source| Qwen35Error::Io {
            path: self.shard_path.clone(),
            source,
        })?;
        let mut row = vec![0_u8; row_bytes];
        let mut values = Vec::with_capacity(value_count);
        for &input_id in input_ids {
            let token = usize::try_from(input_id).map_err(|_| {
                Qwen35Error::InvalidInput(format!("token ID {input_id} cannot address this host"))
            })?;
            if token >= self.vocab_size {
                return Err(Qwen35Error::InvalidInput(format!(
                    "token ID {input_id} exceeds vocabulary size {}",
                    self.vocab_size
                )));
            }
            let byte_offset = token
                .checked_mul(row_bytes)
                .and_then(|offset| u64::try_from(offset).ok())
                .and_then(|offset| self.tensor_data_start.checked_add(offset))
                .ok_or_else(|| {
                    Qwen35Error::InvalidInput(format!("embedding offset overflow for {input_id}"))
                })?;
            file.seek(SeekFrom::Start(byte_offset))
                .and_then(|_| file.read_exact(&mut row))
                .map_err(|source| Qwen35Error::Io {
                    path: self.shard_path.clone(),
                    source,
                })?;
            decode_bf16_row(&row, input_id, &mut values)?;
        }
        Ok(EmbeddingOutput {
            token_count: input_ids.len(),
            hidden_size: self.hidden_size,
            values,
        })
    }

    pub(crate) fn from_layout(shard_path: PathBuf, layout: EmbeddingLayout) -> Self {
        Self {
            shard_path,
            tensor_data_start: layout.tensor_data_start,
            vocab_size: layout.vocab_size,
            hidden_size: layout.hidden_size,
        }
    }

    /// Construct an embedding reader after another checkpoint adapter has
    /// verified the shard and the embedding tensor layout.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn from_verified_parts(
        shard_path: PathBuf,
        tensor_data_start: u64,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            shard_path,
            tensor_data_start,
            vocab_size,
            hidden_size,
        }
    }

    pub(crate) fn verified_shard_path(&self) -> &Path {
        &self.shard_path
    }
}

pub(crate) fn verify_decoder_shard(checkpoint_root: &Path) -> Result<PathBuf, Qwen35Error> {
    let shard_path = checkpoint_root.join(DECODER_SHARD);
    let file_size = file_len(&shard_path)?;
    if file_size != DECODER_SHARD_BYTES {
        return Err(Qwen35Error::InvalidTensor {
            name: "decoder checkpoint shard".to_owned(),
            message: format!("expected shard size {DECODER_SHARD_BYTES}, found {file_size}"),
        });
    }
    let actual_digest = sha256_file(&shard_path)?;
    if actual_digest != DECODER_SHARD_SHA256 {
        return Err(Qwen35Error::DigestMismatch {
            path: DECODER_SHARD.to_owned(),
            expected: DECODER_SHARD_SHA256.to_owned(),
            actual: actual_digest,
        });
    }
    Ok(shard_path)
}

pub(crate) fn decode_bf16_row(
    bytes: &[u8],
    input_id: u32,
    output: &mut Vec<f32>,
) -> Result<(), Qwen35Error> {
    for index in 0..bytes.len() / 2 {
        let offset = index * 2;
        let bits = u32::from(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])) << 16;
        let value = f32::from_bits(bits);
        if !value.is_finite() {
            return Err(Qwen35Error::Numerical(format!(
                "embedding token {input_id} element {index} is not finite: {value}"
            )));
        }
        output.push(value);
    }
    Ok(())
}
