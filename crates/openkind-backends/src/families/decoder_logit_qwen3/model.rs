//! Pinned raw-control checkpoint loading and forward execution.
//!
//! Loading verifies the small artifacts and every shard by digest, then
//! memory-maps the shards into the shared parameterized dense-Qwen3 forward
//! ([`qwen3guard::arch::Qwen3Model`]) widened to FP32 at load. The readout
//! dots the final-norm hidden state at the answer slot with the tied
//! embedding rows of the answer letters — for these checkpoints
//! `tie_word_embeddings` is true, so the embedding table is the output head.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::families::decoder_logit_qwen35::model::resolve_config_field;
use crate::families::qwen3guard::arch::{Qwen3Config, Qwen3Model};
use crate::families::support::{read_json, verify_digest, FamilyControl, FamilyError};

use super::{pinned_config, Qwen3LogitProfile};

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Paths to the verified checkpoint shards, in load order.
    pub checkpoints: Vec<PathBuf>,
    /// Path to the verified tokenizer.
    pub tokenizer: PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts of `profile` in place.
    ///
    /// Cheap contract checks run first so a drifted config or tokenizer
    /// fails before the multi-gigabyte shards are streamed for their
    /// digests.
    pub fn verify(model_root: &Path, profile: &Qwen3LogitProfile) -> Result<Self, FamilyError> {
        Self::verify_artifacts(model_root, profile, true)
    }

    // ONNX needs the pinned renderer/config plus its own digest manifest.
    // Native weights are optional there, but any present bytes still verify.
    pub(crate) fn verify_for_execution(
        model_root: &Path,
        profile: &Qwen3LogitProfile,
        execution: crate::device::FamilyExecution,
    ) -> Result<Self, FamilyError> {
        if execution.is_onnx() {
            Self::verify_artifacts(model_root, profile, false)
        } else {
            Self::verify(model_root, profile)
        }
    }

    fn verify_artifacts(
        model_root: &Path,
        profile: &Qwen3LogitProfile,
        native: bool,
    ) -> Result<Self, FamilyError> {
        let tokenizer = model_root.join("tokenizer.json");
        let config_path = model_root.join("config.json");
        for path in [&tokenizer, &config_path] {
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

        let config_json: serde_json::Value = read_json(&config_path)?;
        for (field, expected) in pinned_config(profile) {
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

        let mut checkpoints = Vec::with_capacity(profile.checkpoint_shards.len());
        for (name, digest) in profile
            .checkpoint_shards
            .iter()
            .zip(profile.checkpoint_sha256s)
        {
            let shard = model_root.join(name);
            if (!native) && !shard.exists() {
                continue;
            }
            if !shard.is_file() {
                return Err(FamilyError::Io {
                    path: shard,
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "required pinned artifact is missing",
                    ),
                });
            }
            verify_digest(&shard, digest)?;
            checkpoints.push(shard);
        }
        Ok(Self {
            checkpoints,
            tokenizer,
        })
    }
}

/// The pinned raw-control model: shared dense forward plus tied-head readout.
pub struct Qwen3ControlModel {
    body: Qwen3Model,
    device: Device,
}

impl Qwen3ControlModel {
    /// Load the verified shards into FP32 weights on `device` through the
    /// shared dense-Qwen3 forward.
    ///
    /// The reference path passes the CPU device; accelerated loads thread a
    /// resolved CUDA device through the same candle model code.
    pub fn load(
        artifacts: &VerifiedArtifacts,
        profile: &Qwen3LogitProfile,
        device: Device,
    ) -> Result<Self, FamilyError> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&artifacts.checkpoints, DType::F32, &device)
        }?;
        let body = Qwen3Model::load(
            &Qwen3Config {
                vocab_size: profile.vocab_size,
                hidden_size: profile.hidden_size,
                head_dim: profile.head_dim,
                num_attention_heads: profile.num_attention_heads,
                num_key_value_heads: profile.num_key_value_heads,
                num_hidden_layers: profile.num_hidden_layers,
                intermediate_size: profile.intermediate_size,
                rope_theta: profile.rope_theta,
                rms_norm_eps: 1e-6,
            },
            vb.pp("model"),
        )?;
        Ok(Self { body, device })
    }

    /// Forward the prompt once and return the next-token logits restricted to
    /// `letter_ids`.
    ///
    /// The answer slot is the final prompt position; the tied embedding rows
    /// of the letter tokens are the readout projection. Nothing is sampled
    /// and no continuation token exists.
    pub fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        let hidden = self
            .body
            .last_hidden(prompt_ids, &self.device)
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        control.check()?;
        let rows = self
            .body
            .embed_rows(letter_ids)
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        let (row_count, hidden_size) = rows
            .dims2()
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        if row_count != letter_ids.len() || hidden_size != hidden.dim(0).unwrap_or(0) {
            return Err(FamilyError::Numerical(format!(
                "letter readout geometry mismatch: {row_count} rows x {hidden_size} wide"
            )));
        }
        let hidden: Vec<f32> = hidden
            .to_vec1()
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        let rows: Vec<f32> = rows
            .flatten_all()
            .and_then(|rows| rows.to_vec1())
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        let logits: Vec<f64> = rows
            .chunks_exact(hidden_size)
            .map(|row| {
                row.iter()
                    .zip(&hidden)
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
