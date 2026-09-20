//! Safetensors layout parsing and file verification for Qwen 3.5 checkpoint shards.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::qwen35::Qwen35Error;

use super::EMBEDDING_TENSOR;

const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub(super) struct CheckpointIndex {
    pub(super) metadata: CheckpointMetadata,
    pub(super) weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct CheckpointMetadata {
    pub(super) total_size: u64,
}

#[derive(Debug, Deserialize)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
pub(crate) struct EmbeddingLayout {
    pub(super) tensor_data_start: u64,
    pub(super) vocab_size: usize,
    pub(super) hidden_size: usize,
}

impl EmbeddingLayout {
    pub(crate) fn read(
        path: &Path,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Self, Qwen35Error> {
        let mut file = File::open(path).map_err(|source| Qwen35Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut length_bytes = [0_u8; 8];
        file.read_exact(&mut length_bytes)
            .map_err(|source| Qwen35Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let header_len = usize::try_from(u64::from_le_bytes(length_bytes)).map_err(|_| {
            Qwen35Error::InvalidTensor {
                name: EMBEDDING_TENSOR.to_owned(),
                message: "safetensors header length cannot address this host".to_owned(),
            }
        })?;
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(Qwen35Error::InvalidTensor {
                name: EMBEDDING_TENSOR.to_owned(),
                message: format!("invalid safetensors header length {header_len}"),
            });
        }
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)
            .map_err(|source| Qwen35Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let tensors: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)
            .map_err(|source| Qwen35Error::Json {
                path: path.to_path_buf(),
                source,
            })?;
        let metadata: TensorMetadata = serde_json::from_value(
            tensors
                .get(EMBEDDING_TENSOR)
                .ok_or_else(|| Qwen35Error::InvalidTensor {
                    name: EMBEDDING_TENSOR.to_owned(),
                    message: "missing from safetensors header".to_owned(),
                })?
                .clone(),
        )
        .map_err(|source| Qwen35Error::Json {
            path: path.to_path_buf(),
            source,
        })?;
        if metadata.dtype != "BF16" || metadata.shape != [vocab_size, hidden_size] {
            return Err(Qwen35Error::InvalidTensor {
                name: EMBEDDING_TENSOR.to_owned(),
                message: format!(
                    "expected BF16 [{vocab_size}, {hidden_size}], found {} {:?}",
                    metadata.dtype, metadata.shape
                ),
            });
        }
        let expected_bytes = vocab_size
            .checked_mul(hidden_size)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Qwen35Error::InvalidTensor {
                name: EMBEDDING_TENSOR.to_owned(),
                message: "embedding tensor byte count overflow".to_owned(),
            })?;
        let [start, end] = metadata.data_offsets;
        if end.checked_sub(start) != Some(expected_bytes) {
            return Err(Qwen35Error::InvalidTensor {
                name: EMBEDDING_TENSOR.to_owned(),
                message: format!(
                    "expected {expected_bytes} data bytes, found offsets [{start}, {end}]"
                ),
            });
        }
        let data_start = 8_u64
            .checked_add(
                u64::try_from(header_len).map_err(|_| Qwen35Error::InvalidTensor {
                    name: EMBEDDING_TENSOR.to_owned(),
                    message: "safetensors header length overflow".to_owned(),
                })?,
            )
            .ok_or_else(|| Qwen35Error::InvalidTensor {
                name: EMBEDDING_TENSOR.to_owned(),
                message: "safetensors data offset overflow".to_owned(),
            })?;
        let tensor_data_start =
            data_start
                .checked_add(start)
                .ok_or_else(|| Qwen35Error::InvalidTensor {
                    name: EMBEDDING_TENSOR.to_owned(),
                    message: "embedding data offset overflow".to_owned(),
                })?;
        let required_len =
            data_start
                .checked_add(end)
                .ok_or_else(|| Qwen35Error::InvalidTensor {
                    name: EMBEDDING_TENSOR.to_owned(),
                    message: "embedding data end overflow".to_owned(),
                })?;
        let actual_len = file_len(path)?;
        if actual_len < required_len {
            return Err(Qwen35Error::InvalidTensor {
                name: EMBEDDING_TENSOR.to_owned(),
                message: format!(
                    "tensor ends at byte {required_len}, but shard has {actual_len} bytes"
                ),
            });
        }
        Ok(Self {
            tensor_data_start,
            vocab_size,
            hidden_size,
        })
    }
}

pub(super) fn verify_small_artifact(
    path: &Path,
    relative: &str,
    expected_digest: &str,
) -> Result<Vec<u8>, Qwen35Error> {
    let bytes = fs::read(path).map_err(|source| Qwen35Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let actual_digest = format!("{:x}", Sha256::digest(&bytes));
    if actual_digest != expected_digest {
        return Err(Qwen35Error::DigestMismatch {
            path: relative.to_owned(),
            expected: expected_digest.to_owned(),
            actual: actual_digest,
        });
    }
    Ok(bytes)
}

pub(super) fn sha256_file(path: &Path) -> Result<String, Qwen35Error> {
    let file = File::open(path).map_err(|source| Qwen35Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer).map_err(|source| Qwen35Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(super) fn file_len(path: &Path) -> Result<u64, Qwen35Error> {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|source| Qwen35Error::Io {
            path: path.to_path_buf(),
            source,
        })
}
