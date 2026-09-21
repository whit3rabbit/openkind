//! Safetensors shard index parsing and raw byte access.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::qwen35::Qwen35Error;

/// One safetensors shard's tensor directory: name → (shape, absolute byte
/// offset of the tensor data, byte length).
pub(crate) struct ShardIndex {
    path: PathBuf,
    pub(crate) tensors: BTreeMap<String, ShardTensor>,
}

#[derive(Debug, Deserialize)]
struct RawTensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

pub(crate) struct ShardTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) offset: u64,
    pub(crate) len: u64,
    /// `true` for FP32 checkpoint tensors (the pinned checkpoint stores the
    /// per-layer `A_log` and `linear_attn.norm.weight` vectors in FP32;
    /// every other tensor is BF16).
    pub(crate) is_f32: bool,
}

impl ShardIndex {
    /// Parse a safetensors header and validate every tensor is BF16 or FP32
    /// and lies inside the file.
    pub(crate) fn read(path: &Path) -> Result<Self, Qwen35Error> {
        let mut file = std::fs::File::open(path).map_err(|source| Qwen35Error::Io {
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
                name: path.to_string_lossy().into_owned(),
                message: "safetensors header length cannot address this host".to_owned(),
            }
        })?;
        if header_len == 0 || header_len > 16 * 1024 * 1024 {
            return Err(Qwen35Error::InvalidTensor {
                name: path.to_string_lossy().into_owned(),
                message: format!("invalid safetensors header length {header_len}"),
            });
        }
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)
            .map_err(|source| Qwen35Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&header).map_err(|source| Qwen35Error::Json {
                path: path.to_path_buf(),
                source,
            })?;
        let data_start = 8_u64
            + u64::try_from(header_len).map_err(|_| Qwen35Error::InvalidTensor {
                name: path.to_string_lossy().into_owned(),
                message: "safetensors header length overflow".to_owned(),
            })?;
        let file_len = std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .map_err(|source| Qwen35Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let mut tensors = BTreeMap::new();
        for (name, raw_metadata) in raw {
            // Safetensors permits a top-level `__metadata__` object alongside
            // tensor entries. It has no dtype/shape and is not a tensor.
            if name == "__metadata__" {
                continue;
            }
            let metadata: RawTensorMetadata =
                serde_json::from_value(raw_metadata).map_err(|source| Qwen35Error::Json {
                    path: path.to_path_buf(),
                    source,
                })?;
            let is_f32 = match metadata.dtype.as_str() {
                "BF16" => false,
                "F32" => true,
                other => {
                    return Err(Qwen35Error::InvalidTensor {
                        name,
                        message: format!(
                            "expected a BF16 or FP32 checkpoint tensor, found {other}"
                        ),
                    });
                }
            };
            let [start, end] = metadata.data_offsets;
            let Some(len) = end.checked_sub(start) else {
                return Err(Qwen35Error::InvalidTensor {
                    name,
                    message: format!("invalid offsets [{start}, {end}]"),
                });
            };
            let element_count = metadata
                .shape
                .iter()
                .try_fold(1_u64, |count, dimension| {
                    count.checked_mul(u64::try_from(*dimension).ok()?)
                })
                .ok_or_else(|| Qwen35Error::InvalidTensor {
                    name: name.clone(),
                    message: "tensor element count overflowed u64".to_owned(),
                })?;
            let element_width = if is_f32 { 4 } else { 2 };
            let expected_len = element_count.checked_mul(element_width).ok_or_else(|| {
                Qwen35Error::InvalidTensor {
                    name: name.clone(),
                    message: "tensor byte count overflowed u64".to_owned(),
                }
            })?;
            if len != expected_len {
                return Err(Qwen35Error::InvalidTensor {
                    name,
                    message: format!(
                        "tensor byte count {len} does not match shape {:?} and dtype width {element_width}",
                        metadata.shape
                    ),
                });
            }
            if metadata
                .shape
                .iter()
                .any(|dimension| i32::try_from(*dimension).is_err())
            {
                return Err(Qwen35Error::InvalidTensor {
                    name,
                    message: "tensor dimension does not fit MLX's i32 shape API".to_owned(),
                });
            }
            let Some(offset) = data_start.checked_add(start) else {
                return Err(Qwen35Error::InvalidTensor {
                    name,
                    message: "tensor data offset overflow".to_owned(),
                });
            };
            if data_start.checked_add(end).is_none_or(|end| end > file_len) {
                return Err(Qwen35Error::InvalidTensor {
                    name,
                    message: format!("tensor data ends past the {file_len}-byte shard"),
                });
            }
            tensors.insert(
                name,
                ShardTensor {
                    shape: metadata.shape,
                    offset,
                    len,
                    is_f32,
                },
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            tensors,
        })
    }

    /// Read one tensor's raw checkpoint bytes.
    pub(crate) fn read_bytes(&self, name: &str) -> Result<(Vec<u8>, &[usize], bool), Qwen35Error> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| Qwen35Error::InvalidTensor {
                name: name.to_owned(),
                message: format!("missing from shard {}", self.path.display()),
            })?;
        let mut bytes = vec![
            0_u8;
            usize::try_from(tensor.len).map_err(|_| {
                Qwen35Error::InvalidTensor {
                    name: name.to_owned(),
                    message: "tensor byte count cannot address this host".to_owned(),
                }
            })?
        ];
        let mut file = std::fs::File::open(&self.path).map_err(|source| Qwen35Error::Io {
            path: self.path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(tensor.offset))
            .and_then(|_| file.read_exact(&mut bytes))
            .map_err(|source| Qwen35Error::Io {
                path: self.path.clone(),
                source,
            })?;
        Ok((bytes, &tensor.shape, tensor.is_f32))
    }
}

/// Exact BF16 → FP32 widening, identical to the Candle oracle's loader.
pub(crate) fn widen_bf16(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    let mut values = Vec::with_capacity(bytes.len() / 2);
    for index in 0..bytes.len() / 2 {
        let offset = index * 2;
        let bits = u32::from(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])) << 16;
        values.push(f32::from_bits(bits));
    }
    Some(values)
}

/// Exact little-endian FP32 read for the checkpoint's FP32 tensors.
pub(crate) fn read_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return None;
    }
    Some(
        chunks
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
    )
}

pub(crate) fn shape_i32(shape: &[usize]) -> Vec<i32> {
    shape
        .iter()
        .map(|dim| i32::try_from(*dim).expect("checkpoint dimension fits i32"))
        .collect()
}
