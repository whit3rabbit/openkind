//! Pinned laya checkpoint loading and the encoder+head forward path.

use std::path::Path;

use candle_core::{DType, Device};

use crate::families::modernbert::ModernBertModel;
use crate::families::support::{verify_digest, FamilyError};

use super::arch::{LayaDecisionHead, LayaHeadConfig};
use super::{LayaProfile, LayaSpecials};

/// The pinned laya model: ModernBERT body plus the typed decision head.
pub struct LayaModel {
    body: ModernBertModel,
    head: LayaDecisionHead,
    device: Device,
}

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified checkpoint shard.
    pub checkpoint: std::path::PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: std::path::PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place and contract-check both configs.
    pub fn verify(model_root: &Path, profile: &LayaProfile) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join("model.safetensors");
        let tokenizer = model_root.join("tokenizer").join("tokenizer.json");
        let tokenizer_config = model_root.join("tokenizer").join("tokenizer_config.json");
        let encoder_config = model_root.join("encoder").join("config.json");
        let agent_config = model_root.join("rl_agent_config.json");
        for path in [
            &checkpoint,
            &tokenizer,
            &tokenizer_config,
            &encoder_config,
            &agent_config,
        ] {
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
        // Cheap contract checks first; the checkpoint digest streams last.
        verify_digest(&agent_config, profile.agent_config_sha256)?;
        let agent_json: serde_json::Value = crate::families::support::read_json(&agent_config)?;
        check_contract(&agent_json, &profile.pinned_agent_config())?;
        verify_digest(&encoder_config, profile.encoder_config_sha256)?;
        let encoder_json: serde_json::Value = crate::families::support::read_json(&encoder_config)?;
        check_contract(&encoder_json, &profile.pinned_encoder_config())?;
        verify_digest(&tokenizer_config, profile.tokenizer_config_sha256)?;
        verify_digest(&tokenizer, profile.tokenizer_json_sha256)?;
        verify_digest(&checkpoint, profile.checkpoint_sha256)?;
        Ok(Self {
            checkpoint,
            tokenizer,
        })
    }
}

/// Contract comparison with exact integer/string/bool equality and a
/// parser-epsilon tolerance for floats: serde_json's default decimal parse
/// can land one ulp away from the correctly rounded double, which must not
/// fail a pinned checkpoint.
fn values_match(expected: &serde_json::Value, actual: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (expected, actual) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => (a - b).abs() <= 1e-9,
            _ => a == b,
        },
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_match(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| b.get(key).is_some_and(|other| values_match(value, other)))
        }
        (a, b) => a == b,
    }
}

fn check_contract(
    json: &serde_json::Value,
    pinned: &[(&'static str, serde_json::Value)],
) -> Result<(), FamilyError> {
    for (field, expected) in pinned {
        let actual =
            resolve_config_field(json, field).ok_or_else(|| FamilyError::ContractMismatch {
                field,
                expected: expected.to_string(),
                actual: "missing".to_owned(),
            })?;
        if !values_match(expected, &actual) {
            return Err(FamilyError::contract(
                field,
                expected.to_string(),
                actual.to_string(),
            ));
        }
    }
    Ok(())
}

fn resolve_config_field(value: &serde_json::Value, field: &str) -> Option<serde_json::Value> {
    let mut current = value;
    let mut segment = field;
    while let Some((head, rest)) = segment.split_once('.') {
        current = current.get(head)?;
        segment = rest;
    }
    if let Some(name) = segment.strip_suffix("[0]") {
        return current.get(name).and_then(|items| items.get(0)).cloned();
    }
    current.get(segment).cloned()
}

impl LayaModel {
    /// Load the verified checkpoint into FP32 weights on `device`. The shard
    /// stores fp16; the mmap is upcast per tensor at load.
    ///
    /// The reference path passes the CPU device; accelerated loads thread a
    /// resolved CUDA device through the same candle model code.
    pub fn load(
        profile: &LayaProfile,
        artifacts: &VerifiedArtifacts,
        device: Device,
    ) -> Result<Self, FamilyError> {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&artifacts.checkpoint),
                DType::F32,
                &device,
            )
        }?;
        let body = ModernBertModel::load(&profile.modernbert_config(), vb.pp("encoder"))?;
        let head = LayaDecisionHead::load(
            &LayaHeadConfig {
                hidden_size: profile.encoder.hidden_size,
                head_layers: 2,
            },
            vb.clone(),
        )?;
        // The shard's `temperature` buffer ships as [1, 1, 1]: training
        // never updated it and the reference decodes with the fitted
        // `rl_agent_config.json` tables, which the contract check pins.
        // Read it only to confirm the buffer exists with the expected shape.
        vb.get((3,), "temperature")?;
        Ok(Self { body, head, device })
    }

    /// One forward pass over the rendered sequence: encoder, type
    /// embedding, head layers, and one logit per marker in marker order.
    pub fn option_logits(
        &self,
        token_ids: &[u32],
        markers: &[usize],
        qtype: usize,
    ) -> Result<Vec<f64>, FamilyError> {
        if token_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "encoded sequence is empty".to_owned(),
            ));
        }
        let hidden = self.body.forward(token_ids, &self.device)?;
        let typed = self.head.typed(&hidden, qtype, &self.device)?;
        let logits = self.head.marker_logits(&typed, markers)?;
        if logits.len() != markers.len() {
            return Err(FamilyError::ContractMismatch {
                field: "marker.count",
                expected: format!("{} marker logits", markers.len()),
                actual: format!("{} logits", logits.len()),
            });
        }
        Ok(logits)
    }
}

/// The mask-token surface string of a pinned profile, exported for the
/// state sanitization step in the engine.
pub(crate) fn mask_token_of(specials: &LayaSpecials) -> &'static str {
    specials.mask_token
}
