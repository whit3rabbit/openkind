//! Pinned decider checkpoint loading and forward execution.
//!
//! Loading verifies the small artifacts (including the pinned
//! `decider_config.json` serving semantics) by digest, streams the 8.4 GB
//! checkpoint shard for its digest in place, and never copies weights. The
//! forward reuses the parity-verified native Qwen3.5 layer implementations
//! through the shared [`TextBackbone`], reading the tied output-embedding
//! rows of the answer labels as the readout projection at the final prompt
//! position — the `Answer: (` slot.

use std::path::{Path, PathBuf};

use candle_core::Device;

use crate::families::decoder_logit_qwen35::model::resolve_config_field;
use crate::families::support::{read_json, verify_digest, FamilyControl, FamilyError};
use crate::qwen35::{EmbeddingLayout, Qwen35Geometry, TextBackbone};

use super::{pinned_config, pinned_decider_config, DeciderProfile};

/// Vocab size and hidden width the tied embedding table must have.
pub(super) const VOCAB_SIZE: usize = 248_320;
pub(super) const HIDDEN_SIZE: usize = 2_560;
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
    /// Verify the pinned artifacts of `profile` in place.
    ///
    /// Cheap contract checks run first so a drifted config, tokenizer, or
    /// `decider_config.json` digest fails before the multi-gigabyte
    /// checkpoint is streamed for its digest.
    pub fn verify(model_root: &Path, profile: &DeciderProfile) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join("model.safetensors");
        let tokenizer = model_root.join("tokenizer.json");
        let config_path = model_root.join("config.json");
        let runtime_config_path = model_root.join("decider_config.json");
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
        verify_digest(&config_path, profile.config_json_sha256)?;
        verify_digest(&tokenizer, profile.tokenizer_json_sha256)?;
        verify_digest(&runtime_config_path, profile.decider_config_sha256)?;

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
        let runtime_config: serde_json::Value = read_json(&runtime_config_path)?;
        for (field, expected) in pinned_decider_config() {
            let actual = resolve_config_field(&runtime_config, field).ok_or_else(|| {
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
        verify_digest(&checkpoint, profile.checkpoint_sha256)?;
        Ok(Self {
            checkpoint,
            tokenizer,
        })
    }
}

/// The pinned decider model: verified backbone plus tied-head readout.
pub struct DeciderModel {
    backbone: TextBackbone,
}

impl DeciderModel {
    /// Load the verified checkpoint into the shared FP32 CPU backbone.
    ///
    /// The embedding layout is read from the safetensors header after the
    /// whole-file digest passed, so the reader can seek rows in place. See
    /// [`Self::load_with_device`] for accelerated loads.
    pub fn load(artifacts: &VerifiedArtifacts) -> Result<Self, FamilyError> {
        Self::load_with_device(artifacts, Device::Cpu)
    }

    /// Load the verified checkpoint onto `device`.
    ///
    /// The decoder forward executes on `device`; the tied-embedding row
    /// lookups stay host-side, exactly as on the CPU reference path.
    pub fn load_with_device(
        artifacts: &VerifiedArtifacts,
        device: Device,
    ) -> Result<Self, FamilyError> {
        let layout = EmbeddingLayout::read(&artifacts.checkpoint, VOCAB_SIZE, HIDDEN_SIZE)
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        let embedding =
            crate::qwen35::Qwen35Embedding::from_layout(artifacts.checkpoint.clone(), layout);
        let backbone = TextBackbone::new(
            embedding,
            vec![artifacts.checkpoint.clone()],
            Qwen35Geometry::PINNED,
            device,
        );
        Ok(Self { backbone })
    }

    /// Forward the prompt once and return the next-token logits restricted to
    /// `label_ids`.
    ///
    /// The answer slot is the final prompt position (the `Answer: (` tail);
    /// the tied output-embedding rows of the label tokens are the readout
    /// projection. Nothing is sampled and no continuation token exists.
    pub fn slot_logits(
        &self,
        prompt_ids: &[u32],
        label_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        let hidden = self
            .backbone
            .forward_hidden_with_check(prompt_ids, || control.check())?;
        control.check()?;
        let rows = self.backbone.embedding_rows(label_ids)?;
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
                "slot-logit readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }
}
