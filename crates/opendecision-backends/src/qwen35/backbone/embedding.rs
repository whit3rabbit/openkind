use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::super::Qwen35Error;

const CONFIG_FILE: &str = "config.json";
const MODEL_INDEX_FILE: &str = "model.safetensors.index.json";
const EMBEDDING_TENSOR: &str = "model.language_model.embed_tokens.weight";
const EMBEDDING_SHARD: &str = "model.safetensors-00001-of-00002.safetensors";
const DECODER_SHARD: &str = "model.safetensors-00002-of-00002.safetensors";
const CONFIG_SHA256: &str = "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670";
const MODEL_INDEX_SHA256: &str = "eae340074abb0a5f31a6621f7ae8e8248a7c1790df04a722c4e4b70c2a6d1dbb";
const EMBEDDING_SHARD_SHA256: &str =
    "df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712";
const EMBEDDING_SHARD_BYTES: u64 = 5_329_398_712;
const DECODER_SHARD_SHA256: &str =
    "590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef";
const DECODER_SHARD_BYTES: u64 = 3_990_429_344;
const CHECKPOINT_BYTES: u64 = 9_319_737_856;
const VOCAB_SIZE: usize = 248_320;
const HIDDEN_SIZE: usize = 2_560;
const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;

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

    fn from_layout(shard_path: PathBuf, layout: EmbeddingLayout) -> Self {
        Self {
            shard_path,
            tensor_data_start: layout.tensor_data_start,
            vocab_size: layout.vocab_size,
            hidden_size: layout.hidden_size,
        }
    }

    pub(super) fn verified_shard_path(&self) -> &Path {
        &self.shard_path
    }
}

pub(super) fn verify_decoder_shard(checkpoint_root: &Path) -> Result<PathBuf, Qwen35Error> {
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

#[derive(Debug, Deserialize)]
struct CheckpointIndex {
    metadata: CheckpointMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct CheckpointMetadata {
    total_size: u64,
}

#[derive(Debug, Deserialize)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
struct EmbeddingLayout {
    tensor_data_start: u64,
    vocab_size: usize,
    hidden_size: usize,
}

impl EmbeddingLayout {
    fn read(path: &Path, vocab_size: usize, hidden_size: usize) -> Result<Self, Qwen35Error> {
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

fn decode_bf16_row(bytes: &[u8], input_id: u32, output: &mut Vec<f32>) -> Result<(), Qwen35Error> {
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

fn verify_small_artifact(
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

fn sha256_file(path: &Path) -> Result<String, Qwen35Error> {
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

fn file_len(path: &Path) -> Result<u64, Qwen35Error> {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|source| Qwen35Error::Io {
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::qwen35::BackboneReference;

    const TOKEN_25_ROW_SHA256: &str =
        "84ce40703c960b305ad72adbdc0f6fc5b8df6fd279e5b18d5dd194e4764377c7";
    const TOKEN_25_ROW_HEX: &str = include_str!(
        "../../../tests/fixtures/qwen35_backbone_phase3b_a047d6802c3f06f085b8/embedding_token_25.bf16.hex"
    );

    #[test]
    fn seek_lookup_preserves_token_order_and_widens_bf16() {
        let root = temp_dir("lookup");
        let shard_path = root.join("tiny.safetensors");
        write_tiny_shard(&shard_path, &[[1.0, -2.0], [3.5, 4.0], [-0.5, 8.0]]);
        let layout = EmbeddingLayout::read(&shard_path, 3, 2).expect("read tiny layout");
        let embedding = Qwen35Embedding::from_layout(shard_path, layout);

        let output = embedding.embed(&[2, 0]).expect("embed tokens");
        assert_eq!(output.token_count(), 2);
        assert_eq!(output.hidden_size(), 2);
        assert_eq!(output.values(), &[-0.5, 8.0, 1.0, -2.0]);
        assert_eq!(output.last_token(), &[1.0, -2.0]);
        assert!(matches!(
            embedding.embed(&[3]),
            Err(Qwen35Error::InvalidInput(_))
        ));
        assert!(matches!(
            embedding.embed(&[]),
            Err(Qwen35Error::InvalidInput(_))
        ));
        drop(embedding);
        fs::remove_dir_all(root).expect("remove temp directory");
    }

    #[test]
    fn tensor_layout_rejects_wrong_dtype_and_shape() {
        let root = temp_dir("invalid-layout");
        let rows = [[1.0, -2.0], [3.5, 4.0], [-0.5, 8.0]];
        let wrong_dtype = root.join("wrong-dtype.safetensors");
        write_tiny_shard_with_layout(&wrong_dtype, "F32", [3, 2], &rows);
        assert!(matches!(
            EmbeddingLayout::read(&wrong_dtype, 3, 2),
            Err(Qwen35Error::InvalidTensor { .. })
        ));

        let wrong_shape = root.join("wrong-shape.safetensors");
        write_tiny_shard_with_layout(&wrong_shape, "BF16", [2, 3], &rows);
        assert!(matches!(
            EmbeddingLayout::read(&wrong_shape, 3, 2),
            Err(Qwen35Error::InvalidTensor { .. })
        ));
        fs::remove_dir_all(root).expect("remove temp directory");
    }

    #[test]
    fn pinned_checkpoint_row_matches_phase3b_embedding_exactly() {
        let bytes = decode_hex(TOKEN_25_ROW_HEX);
        assert_eq!(bytes.len(), HIDDEN_SIZE * 2);
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), TOKEN_25_ROW_SHA256);
        let mut actual = Vec::with_capacity(HIDDEN_SIZE);
        decode_bf16_row(&bytes, 25, &mut actual).expect("decode pinned BF16 row");

        let phase3b_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z");
        let reference = BackboneReference::load(phase3b_root).expect("load Phase 3B reference");
        let comparison = reference
            .compare("diagnostic.embedding", &actual)
            .expect("compare embedding");
        assert_eq!(comparison.max_abs(), 0.0);
        assert_eq!(comparison.rms(), 0.0);
        assert!((comparison.cosine() - 1.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn non_finite_bf16_is_rejected() {
        let mut output = Vec::new();
        assert!(matches!(
            decode_bf16_row(&0x7f80_u16.to_le_bytes(), 0, &mut output),
            Err(Qwen35Error::Numerical(_))
        ));
    }

    #[test]
    fn decoder_shard_verification_rejects_wrong_size() {
        let root = temp_dir("decoder-shard-size");
        fs::write(root.join(DECODER_SHARD), b"not a model shard").expect("write tiny shard");
        assert!(matches!(
            verify_decoder_shard(&root),
            Err(Qwen35Error::InvalidTensor { .. })
        ));
        fs::remove_dir_all(root).expect("remove temp directory");
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "opendecision-qwen35-embedding-{}-{label}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create temp directory");
        root
    }

    fn write_tiny_shard(path: &Path, rows: &[[f32; 2]; 3]) {
        write_tiny_shard_with_layout(path, "BF16", [3, 2], rows);
    }

    fn write_tiny_shard_with_layout(
        path: &Path,
        dtype: &str,
        shape: [usize; 2],
        rows: &[[f32; 2]; 3],
    ) {
        let mut header = serde_json::to_vec(&serde_json::json!({
            EMBEDDING_TENSOR: {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, 12]
            }
        }))
        .expect("serialize header");
        let padding = (8 - header.len() % 8) % 8;
        header.resize(header.len() + padding, b' ');
        let mut file = File::create(path).expect("create tiny shard");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("write header length");
        file.write_all(&header).expect("write header");
        for row in rows {
            for value in row {
                let bf16 = (value.to_bits() >> 16) as u16;
                file.write_all(&bf16.to_le_bytes()).expect("write BF16");
            }
        }
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        let value = value.trim();
        assert_eq!(value.len() % 2, 0, "hex fixture must have byte pairs");
        let bytes = value.as_bytes();
        (0..bytes.len() / 2)
            .map(|index| {
                let offset = index * 2;
                let pair = std::str::from_utf8(&bytes[offset..offset + 2]).expect("ASCII hex");
                u8::from_str_radix(pair, 16).expect("valid hex")
            })
            .collect()
    }
}
