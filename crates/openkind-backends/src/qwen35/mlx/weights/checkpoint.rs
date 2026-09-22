//! Checkpoint format validation and key normalization for MLX.

use std::collections::BTreeMap;
use std::io::{BufReader, Read};
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::shard::ShardIndex;
use crate::qwen35::backbone::{verify_decoder_shard, CONFIG_SHA256, MODEL_INDEX_SHA256};
use crate::qwen35::mlx::MlxError;
use crate::qwen35::{Qwen35Embedding, BACKBONE_ID, BACKBONE_REVISION, TOKENIZER_JSON_SHA256};

const COMMUNITY_BACKBONE_ID: &str = "mlx-community/Qwen3.5-4B-MLX-bf16";
const COMMUNITY_BACKBONE_REVISION: &str = "475632ded9a95863da4e4b235ab9ccbc5d3cc6bf";
const COMMUNITY_TOKENIZER_JSON_SHA256: &str =
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4";
const COMMUNITY_INDEX_FILE: &str = "model.safetensors.index.json";
const COMMUNITY_INDEX_SHA256: &str =
    "0d718f2e2c433b1913fdeb7e4ebcdaab291e49137ec806980aecb1e88cf94d9d";
const COMMUNITY_TOTAL_SIZE: u64 = 9_078_532_608;
const COMMUNITY_EMBEDDING_TENSOR: &str = "language_model.model.embed_tokens.weight";
const COMMUNITY_EMBEDDING_SHARD: &str = "model-00001-of-00002.safetensors";
const COMMUNITY_EMBEDDING_SHARD_SHA256: &str =
    "bbf84cddf80989b5a34317031a766816e4b5752468c498b60076cc4380b4aa9e";
const COMMUNITY_EMBEDDING_SHARD_BYTES: u64 = 5_366_980_383;
const COMMUNITY_DECODER_SHARD: &str = "model-00002-of-00002.safetensors";
const COMMUNITY_DECODER_SHARD_SHA256: &str =
    "b022545897033ebc19523a3a949e0ebe1837f3e76abe8a991aa2f71e4babe07a";
const COMMUNITY_DECODER_SHARD_BYTES: u64 = 3_711_641_426;
const COMMUNITY_VOCAB_SIZE: usize = 248_320;
const COMMUNITY_HIDDEN_SIZE: usize = 2_560;

/// Checkpoint layouts accepted by the MLX loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxCheckpointFormat {
    /// The pinned Qwen base safetensors layout used by the Phase 3B oracle.
    PinnedQwen35Base,
    /// The verified `mlx-community/Qwen3.5-4B-MLX-bf16` safetensors export.
    MlxCommunityQwen35Bf16,
}

impl MlxCheckpointFormat {
    /// Stable format identifier for evidence and state identity.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PinnedQwen35Base => "qwen35-base-safetensors",
            Self::MlxCommunityQwen35Bf16 => "mlx-community-qwen35-4b-bf16",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MlxCheckpointIdentity {
    pub(crate) format: MlxCheckpointFormat,
    pub(crate) backbone_id: &'static str,
    pub(crate) backbone_revision: &'static str,
    pub(crate) tokenizer_digest: &'static str,
}

#[derive(Debug, Deserialize)]
struct RawCheckpointIndex {
    metadata: RawCheckpointMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawCheckpointMetadata {
    total_size: u64,
}

pub(super) struct VerifiedCheckpoint {
    pub(super) format: MlxCheckpointFormat,
    pub(super) identity: MlxCheckpointIdentity,
    pub(super) embedding: Qwen35Embedding,
    pub(super) indexes: [ShardIndex; 2],
}

pub(super) fn load_checkpoint(checkpoint_root: &Path) -> Result<VerifiedCheckpoint, MlxError> {
    let index_path = checkpoint_root.join("model.safetensors.index.json");
    let index_bytes = std::fs::read(&index_path).map_err(|error| {
        MlxError::InvalidState(format!(
            "failed to read checkpoint index {}: {error}",
            index_path.display()
        ))
    })?;
    let index_digest = format!("{:x}", Sha256::digest(&index_bytes));
    let format = if index_digest == MODEL_INDEX_SHA256 {
        MlxCheckpointFormat::PinnedQwen35Base
    } else if index_digest == COMMUNITY_INDEX_SHA256 {
        MlxCheckpointFormat::MlxCommunityQwen35Bf16
    } else {
        return Err(MlxError::InvalidState(format!(
            "unsupported Qwen3.5 safetensors index digest {index_digest}"
        )));
    };

    match format {
        MlxCheckpointFormat::PinnedQwen35Base => {
            let embedding = Qwen35Embedding::load(checkpoint_root).map_err(MlxError::from_qwen)?;
            let decoder_shard =
                verify_decoder_shard(checkpoint_root).map_err(MlxError::from_qwen)?;
            let embedding_shard = embedding.verified_shard_path().to_path_buf();
            let embedding_index =
                ShardIndex::read(&embedding_shard).map_err(MlxError::from_qwen)?;
            let decoder_index = ShardIndex::read(&decoder_shard).map_err(MlxError::from_qwen)?;
            Ok(VerifiedCheckpoint {
                format,
                identity: MlxCheckpointIdentity {
                    format,
                    backbone_id: BACKBONE_ID,
                    backbone_revision: BACKBONE_REVISION,
                    tokenizer_digest: TOKENIZER_JSON_SHA256,
                },
                embedding,
                indexes: [embedding_index, decoder_index],
            })
        }
        MlxCheckpointFormat::MlxCommunityQwen35Bf16 => {
            load_community_checkpoint(checkpoint_root, &index_bytes)
        }
    }
}

fn load_community_checkpoint(
    checkpoint_root: &Path,
    index_bytes: &[u8],
) -> Result<VerifiedCheckpoint, MlxError> {
    verify_digest(
        &checkpoint_root.join("config.json"),
        "config.json",
        CONFIG_SHA256,
    )?;
    verify_digest(
        &checkpoint_root.join("tokenizer.json"),
        "tokenizer.json",
        COMMUNITY_TOKENIZER_JSON_SHA256,
    )?;
    let index: RawCheckpointIndex = serde_json::from_slice(index_bytes).map_err(|error| {
        MlxError::InvalidState(format!("failed to decode {COMMUNITY_INDEX_FILE}: {error}"))
    })?;
    if index.metadata.total_size != COMMUNITY_TOTAL_SIZE {
        return Err(MlxError::InvalidState(format!(
            "community checkpoint index expected {COMMUNITY_TOTAL_SIZE} parameter bytes, found {}",
            index.metadata.total_size
        )));
    }
    let embedding_shard = index
        .weight_map
        .get(COMMUNITY_EMBEDDING_TENSOR)
        .ok_or_else(|| {
            MlxError::InvalidState(format!(
                "community checkpoint index is missing {COMMUNITY_EMBEDDING_TENSOR}"
            ))
        })?;
    if embedding_shard != COMMUNITY_EMBEDDING_SHARD {
        return Err(MlxError::InvalidState(format!(
            "community embedding is in unexpected shard {embedding_shard}"
        )));
    }
    if index
        .weight_map
        .values()
        .any(|shard| shard != COMMUNITY_EMBEDDING_SHARD && shard != COMMUNITY_DECODER_SHARD)
    {
        return Err(MlxError::InvalidState(
            "community checkpoint index references an unexpected shard".to_owned(),
        ));
    }

    let embedding_path = checkpoint_root.join(COMMUNITY_EMBEDDING_SHARD);
    let decoder_path = checkpoint_root.join(COMMUNITY_DECODER_SHARD);
    verify_file(
        &embedding_path,
        COMMUNITY_EMBEDDING_SHARD_BYTES,
        COMMUNITY_EMBEDDING_SHARD_SHA256,
        COMMUNITY_EMBEDDING_SHARD,
    )?;
    verify_file(
        &decoder_path,
        COMMUNITY_DECODER_SHARD_BYTES,
        COMMUNITY_DECODER_SHARD_SHA256,
        COMMUNITY_DECODER_SHARD,
    )?;

    let embedding_index = ShardIndex::read(&embedding_path).map_err(MlxError::from_qwen)?;
    let decoder_index = ShardIndex::read(&decoder_path).map_err(MlxError::from_qwen)?;
    let embedding_tensor = embedding_index
        .tensors
        .get(COMMUNITY_EMBEDDING_TENSOR)
        .ok_or_else(|| {
            MlxError::InvalidState(format!(
                "community embedding shard is missing {COMMUNITY_EMBEDDING_TENSOR}"
            ))
        })?;
    if embedding_tensor.is_f32
        || embedding_tensor.shape != [COMMUNITY_VOCAB_SIZE, COMMUNITY_HIDDEN_SIZE]
    {
        return Err(MlxError::InvalidState(format!(
            "community embedding has unexpected dtype or shape {:?}",
            embedding_tensor.shape
        )));
    }
    let embedding = Qwen35Embedding::from_verified_parts(
        embedding_path,
        embedding_tensor.offset,
        COMMUNITY_VOCAB_SIZE,
        COMMUNITY_HIDDEN_SIZE,
    );
    let format = MlxCheckpointFormat::MlxCommunityQwen35Bf16;
    Ok(VerifiedCheckpoint {
        format,
        identity: MlxCheckpointIdentity {
            format,
            backbone_id: COMMUNITY_BACKBONE_ID,
            backbone_revision: COMMUNITY_BACKBONE_REVISION,
            tokenizer_digest: COMMUNITY_TOKENIZER_JSON_SHA256,
        },
        embedding,
        indexes: [embedding_index, decoder_index],
    })
}

fn verify_digest(path: &Path, label: &str, expected: &str) -> Result<(), MlxError> {
    let bytes = std::fs::read(path).map_err(|error| {
        MlxError::InvalidState(format!(
            "failed to read {label} at {}: {error}",
            path.display()
        ))
    })?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        return Err(MlxError::InvalidState(format!(
            "SHA-256 mismatch for {label}: expected {expected}, found {actual}"
        )));
    }
    Ok(())
}

fn verify_file(
    path: &Path,
    expected_bytes: u64,
    expected_digest: &str,
    label: &str,
) -> Result<(), MlxError> {
    let actual_bytes = std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| MlxError::InvalidState(format!("failed to stat {label}: {error}")))?;
    if actual_bytes != expected_bytes {
        return Err(MlxError::InvalidState(format!(
            "{label} expected {expected_bytes} bytes, found {actual_bytes}"
        )));
    }
    let file = std::fs::File::open(path)
        .map_err(|error| MlxError::InvalidState(format!("failed to open {label}: {error}")))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| MlxError::InvalidState(format!("failed to hash {label}: {error}")))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual_digest = format!("{:x}", hasher.finalize());
    if actual_digest != expected_digest {
        return Err(MlxError::InvalidState(format!(
            "SHA-256 mismatch for {label}: expected {expected_digest}, found {actual_digest}"
        )));
    }
    Ok(())
}

pub(super) fn canonical_tensor_name(format: MlxCheckpointFormat, name: &str) -> Option<String> {
    match format {
        MlxCheckpointFormat::PinnedQwen35Base => Some(name.to_owned()),
        MlxCheckpointFormat::MlxCommunityQwen35Bf16 => {
            if let Some(rest) = name.strip_prefix("language_model.model.") {
                Some(format!("model.language_model.{rest}"))
            } else {
                name.strip_prefix("vision_tower.")
                    .map(|rest| format!("model.visual.{rest}"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{canonical_tensor_name, MlxCheckpointFormat};

    #[test]
    fn community_text_keys_are_normalized_to_backend_namespace() {
        assert_eq!(
            canonical_tensor_name(
                MlxCheckpointFormat::MlxCommunityQwen35Bf16,
                "language_model.model.layers.0.linear_attn.A_log"
            ),
            Some("model.language_model.layers.0.linear_attn.A_log".to_owned())
        );
    }

    #[test]
    fn community_vision_keys_are_normalized_without_claiming_text_ownership() {
        assert_eq!(
            canonical_tensor_name(
                MlxCheckpointFormat::MlxCommunityQwen35Bf16,
                "vision_tower.blocks.0.attn.qkv.weight"
            ),
            Some("model.visual.blocks.0.attn.qkv.weight".to_owned())
        );
        assert_eq!(
            canonical_tensor_name(
                MlxCheckpointFormat::MlxCommunityQwen35Bf16,
                "language_model.lm_head.weight"
            ),
            None
        );
    }

    #[test]
    fn pinned_keys_are_left_unchanged() {
        let name = "model.language_model.layers.0.mlp.down_proj.weight";
        assert_eq!(
            canonical_tensor_name(MlxCheckpointFormat::PinnedQwen35Base, name),
            Some(name.to_owned())
        );
    }
}
