//! Pinned von checkpoint loading and the encoder+scorer forward path.
//!
//! The reference checkpoint is the author's `option_marker.pt` — a PyTorch
//! pickle whose state dict carries the full `encoder.*` ModernBERT weights
//! plus the `scorer.*` head. candle's native pickle reader decompresses the
//! archive and returns named tensors, so openkind serves the author's
//! artifact byte-for-byte with no conversion or re-hosting.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::families::modernbert::ModernBertModel;
use crate::families::support::{read_json, verify_digest, FamilyError};

use super::arch::{VonScorer, VonScorerConfig};
use super::VonProfile;

/// The pinned von model: ModernBERT body plus the option-marker scorer.
pub struct VonModel {
    body: ModernBertModel,
    scorer: VonScorer,
    device: Device,
    pub(crate) max_sequence_tokens: usize,
}

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified `option_marker.pt` checkpoint.
    pub checkpoint: PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: PathBuf,
}

/// Locate `path` under the model root's `checkpoint/` layout, failing with a
/// family error when missing.
fn require(model_root: &Path, relative: &str) -> Result<PathBuf, FamilyError> {
    let path = model_root.join(relative);
    if !path.is_file() {
        return Err(FamilyError::Io {
            path: path.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "required pinned artifact is missing",
            ),
        });
    }
    Ok(path)
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place and contract-check both configs.
    pub fn verify(model_root: &Path, profile: &VonProfile) -> Result<Self, FamilyError> {
        Self::verify_artifacts(model_root, profile, true)
    }

    // ONNX needs the pinned renderer/config plus its own digest manifest.
    // Native weights are optional there, but any present bytes still verify.
    pub(crate) fn verify_for_execution(
        model_root: &Path,
        profile: &VonProfile,
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
        profile: &VonProfile,
        native: bool,
    ) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join("checkpoint/option_marker.pt");
        let config = require(model_root, "checkpoint/config.json")?;
        let tokenizer = require(model_root, "checkpoint/tokenizer.json")?;
        let tokenizer_config = require(model_root, "checkpoint/tokenizer_config.json")?;
        let calibration = require(model_root, "checkpoint/marker_calibration.json")?;
        // Cheap contract checks first; the checkpoint digest streams last.
        verify_digest(&config, profile.config_sha256)?;
        let config_json: serde_json::Value = read_json(&config)?;
        check_contract(&config_json, &profile.pinned_config())?;
        verify_digest(&calibration, profile.calibration_sha256)?;
        let calibration_json: serde_json::Value = read_json(&calibration)?;
        check_contract(&calibration_json, &profile.pinned_calibration())?;
        verify_digest(&tokenizer_config, profile.tokenizer_config_sha256)?;
        verify_digest(&tokenizer, profile.tokenizer_json_sha256)?;
        if native || checkpoint.exists() {
            verify_digest(&checkpoint, profile.checkpoint_sha256)?;
        }
        Ok(Self {
            checkpoint,
            tokenizer,
        })
    }
}

/// Contract comparison with exact integer/string/bool equality and a
/// parser-epsilon tolerance for floats (serde_json's default decimal parse
/// can land one ulp away from the correctly rounded double).
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

impl VonModel {
    /// Load the verified checkpoint into FP32 weights on `device`. The
    /// pickle stores the training fp32 tensors; every tensor upcasts to F32
    /// at build.
    ///
    /// The reference path passes the CPU device; accelerated loads thread a
    /// resolved CUDA device through the same candle model code.
    pub fn load(
        profile: &VonProfile,
        artifacts: &VerifiedArtifacts,
        device: Device,
    ) -> Result<Self, FamilyError> {
        let tensors = candle_core::pickle::read_all(&artifacts.checkpoint)?;
        let map: HashMap<String, Tensor> = tensors.into_iter().collect();
        if map.is_empty() {
            return Err(FamilyError::InvalidInput(
                "checkpoint pickle contained no tensors".to_owned(),
            ));
        }
        let vb = VarBuilder::from_tensors(map, DType::F32, &device);
        let body = ModernBertModel::load(&profile.modernbert_config(), vb.pp("encoder"))?;
        let scorer = VonScorer::load(
            &VonScorerConfig {
                hidden_size: profile.encoder.hidden_size,
            },
            vb.pp("scorer"),
        )?;
        Ok(Self {
            body,
            scorer,
            device,
            max_sequence_tokens: profile.max_sequence_tokens,
        })
    }

    /// One forward pass over the packed sequence: encoder, gather the marker
    /// hidden states, and score them into one logit per option in order.
    pub fn option_logits(
        &self,
        token_ids: &[u32],
        markers: &[usize],
    ) -> Result<Vec<f64>, FamilyError> {
        if token_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "encoded sequence is empty".to_owned(),
            ));
        }
        if markers.is_empty() {
            return Err(FamilyError::InvalidInput(
                "packed sequence carries no option markers".to_owned(),
            ));
        }
        if token_ids.len() > self.max_sequence_tokens {
            return Err(FamilyError::InvalidInput(format!(
                "token sequence length {} exceeds maximum sequence window {}",
                token_ids.len(),
                self.max_sequence_tokens
            )));
        }
        let hidden = self.body.forward(token_ids, &self.device)?;
        let seq_len = hidden.dim(0)?;
        let positions = Tensor::from_slice(
            &markers
                .iter()
                .map(|&position| position as u32)
                .collect::<Vec<u32>>(),
            markers.len(),
            &self.device,
        )?;
        for marker in markers {
            if *marker >= seq_len {
                return Err(FamilyError::InvalidInput(format!(
                    "marker position {marker} exceeds the packed sequence length {seq_len}"
                )));
            }
        }
        let gathered = hidden.index_select(&positions, 0)?; // (k, hidden)
        let logits = self.scorer.forward(&gathered)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::modernbert::ModernBertConfig;
    use candle_nn::VarBuilder;

    #[test]
    fn option_logits_rejects_sequence_exceeding_max_sequence_tokens() {
        let cfg = ModernBertConfig {
            vocab_size: 100,
            hidden_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 32,
            local_attention: 16,
            global_attn_every_n_layers: 1,
            global_rope_theta: 10_000.0,
            local_rope_theta: 10_000.0,
            norm_eps: 1e-5,
            max_sequence_tokens: 10,
        };
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let body = ModernBertModel::load(&cfg, vb.pp("encoder")).expect("load dummy body");
        let scorer = VonScorer::load(&VonScorerConfig { hidden_size: 16 }, vb.pp("scorer"))
            .expect("load dummy scorer");
        let model = VonModel {
            body,
            scorer,
            device: Device::Cpu,
            max_sequence_tokens: 10,
        };
        let token_ids = vec![1u32; 11];
        let markers = vec![0usize];
        let error = model
            .option_logits(&token_ids, &markers)
            .expect_err("must reject sequence > 10");
        assert!(
            matches!(&error, FamilyError::InvalidInput(msg) if msg.contains("exceeds maximum sequence window 10")),
            "unexpected error: {error}"
        );
    }
}
