//! Pinned JevK5 checkpoint loading and forward execution.
//!
//! Loading verifies the small artifacts by digest, streams the 8.4 GB
//! checkpoint shard for its digest in place, and never copies weights. The
//! forward reuses the parity-verified native Qwen3.5 layer implementations
//! through the shared [`TextBackbone`], reading the tied output-embedding
//! rows of the answer letters as the readout projection.

use std::path::{Path, PathBuf};

use crate::families::support::{read_json, verify_digest, FamilyControl, FamilyError};
use crate::qwen35::{EmbeddingLayout, TextBackbone};

use super::{
    pinned_config, CHECKPOINT_SHA256, CONFIG_JSON_SHA256, RUNTIME_CONFIG_SHA256,
    TOKENIZER_JSON_SHA256,
};

/// Vocab size and hidden width the tied embedding table must have.
const VOCAB_SIZE: usize = 248_320;
const HIDDEN_SIZE: usize = 2_560;
/// Frozen decoder-layer count of the pinned architecture.
const LAYER_COUNT: usize = 32;

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified single-file checkpoint.
    pub checkpoint: PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place.
    ///
    /// Cheap contract checks run first so a drifted config, tokenizer, or
    /// runtime-config digest fails before the multi-gigabyte checkpoint is
    /// streamed for its digest.
    pub fn verify(model_root: &Path) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join("model.safetensors");
        let tokenizer = model_root.join("tokenizer.json");
        let config_path = model_root.join("config.json");
        let runtime_config_path = model_root.join("jevk5_config.json");
        // Cheap artifacts first: existence, digest, and contract checks fail
        // fast before the multi-gigabyte checkpoint is even opened.
        for path in [&tokenizer, &config_path, &runtime_config_path] {
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
        verify_digest(&config_path, CONFIG_JSON_SHA256)?;
        verify_digest(&tokenizer, TOKENIZER_JSON_SHA256)?;
        verify_digest(&runtime_config_path, RUNTIME_CONFIG_SHA256)?;

        let config_json: serde_json::Value = read_json(&config_path)?;
        for (field, expected) in pinned_config() {
            let actual = resolve_config_field(&config_json, field).ok_or_else(|| {
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
        if !checkpoint.is_file() {
            return Err(FamilyError::Io {
                path: checkpoint.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "required pinned artifact is missing",
                ),
            });
        }
        verify_digest(&checkpoint, CHECKPOINT_SHA256)?;
        Ok(Self {
            checkpoint,
            tokenizer,
        })
    }
}

/// Resolve a dotted config path such as `rope_parameters.rope_theta` or an
/// `architectures[0]` list index.
fn resolve_config_field(value: &serde_json::Value, field: &str) -> Option<serde_json::Value> {
    let mut current = value;
    for segment in field.split('.') {
        if let Some(name) = segment.strip_suffix("[0]") {
            current = current.get(name).and_then(|items| items.get(0))?;
        } else {
            current = current.get(segment)?;
        }
    }
    Some(current.clone())
}

/// The pinned JevK5 model: verified backbone plus tied-head readout.
pub struct Jevk5Model {
    backbone: TextBackbone,
}

impl Jevk5Model {
    /// Load the verified checkpoint into the shared FP32 CPU backbone.
    ///
    /// The embedding layout is read from the safetensors header after the
    /// whole-file digest passed, so the reader can seek rows in place.
    pub fn load(artifacts: &VerifiedArtifacts) -> Result<Self, FamilyError> {
        let layout = EmbeddingLayout::read(&artifacts.checkpoint, VOCAB_SIZE, HIDDEN_SIZE)
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        let embedding =
            crate::qwen35::Qwen35Embedding::from_layout(artifacts.checkpoint.clone(), layout);
        let backbone =
            TextBackbone::new(embedding, vec![artifacts.checkpoint.clone()], LAYER_COUNT);
        Ok(Self { backbone })
    }

    /// Forward the prompt once and return the next-token logits restricted to
    /// `letter_ids`.
    ///
    /// The answer slot is the final prompt position; the tied output-embedding
    /// rows of the letter tokens are the readout projection. Nothing is
    /// sampled and no continuation token exists.
    pub fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        let hidden = self
            .backbone
            .forward_hidden_with_check(prompt_ids, || control.check())?;
        control.check()?;
        let rows = self.backbone.embedding_rows(letter_ids)?;
        let token_start = (prompt_ids.len() - 1) * HIDDEN_SIZE;
        let final_token = &hidden[token_start..];
        let logits: Vec<f64> = rows
            .as_chunks::<HIDDEN_SIZE>()
            .0
            .iter()
            .map(|row| {
                row.iter()
                    .zip(final_token)
                    .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                    .sum::<f64>()
            })
            .collect();
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(FamilyError::Numerical(
                "letter-logit readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_field_resolution_walks_nested_paths_and_list_indices() {
        let value = json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "rope_parameters": {"rope_theta": 10_000_000},
        });
        assert_eq!(
            resolve_config_field(&value, "architectures[0]"),
            Some(json!("Qwen3_5ForCausalLM"))
        );
        assert_eq!(
            resolve_config_field(&value, "rope_parameters.rope_theta"),
            Some(json!(10_000_000))
        );
        assert_eq!(resolve_config_field(&value, "missing.path"), None);
        assert_eq!(resolve_config_field(&value, "architectures[1]"), None);
    }
}
