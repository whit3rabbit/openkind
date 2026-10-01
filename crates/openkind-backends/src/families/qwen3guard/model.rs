//! Pinned Qwen3Guard-Stream checkpoint loading and head execution.

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::{linear_no_bias, rms_norm, Module, RmsNorm, VarBuilder};

use crate::families::support::{verify_digest, FamilyError};

use super::arch::{config_from_pinned, Qwen3Model};
use super::{pinned_config, CHECKPOINT_SHA256, CONFIG_JSON_SHA256, TOKENIZER_JSON_SHA256};

/// The pinned guard model: Qwen3 body plus the query-side Stream heads.
pub struct Qwen3GuardModel {
    body: Qwen3Model,
    pre: candle_nn::Linear,
    pre_norm: RmsNorm,
    risk_head: candle_nn::Linear,
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
    /// Verify the pinned artifacts in place.
    pub fn verify(model_root: &Path) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join("model.safetensors");
        let tokenizer = model_root.join("tokenizer.json");
        let config_path = model_root.join("config.json");
        for path in [&checkpoint, &tokenizer, &config_path] {
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
        verify_digest(&checkpoint, CHECKPOINT_SHA256)?;
        // Contract-check the pinned config values.
        let config_json: serde_json::Value = crate::families::support::read_json(&config_path)?;
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
        Ok(Self {
            checkpoint,
            tokenizer,
        })
    }
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

impl Qwen3GuardModel {
    /// Load the verified checkpoint into FP32 weights on `device`.
    ///
    /// The reference path passes the CPU device; accelerated loads thread a
    /// resolved CUDA device through the same candle model code.
    pub fn load(artifacts: &VerifiedArtifacts, device: Device) -> Result<Self, FamilyError> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&artifacts.checkpoint),
                DType::F32,
                &device,
            )
        }?;
        let body = Qwen3Model::load(&config_from_pinned(1024, 128), vb.pp("model"))?;
        // Query-side Stream head: hidden → guard_inner → RMSNorm → risk head.
        let pre = linear_no_bias(1024, 512, vb.pp("query_risk_level_category_pre"))?;
        let pre_norm = rms_norm(512, 1e-6, vb.pp("query_risk_level_category_layernorm"))?;
        let risk_head = linear_no_bias(512, 3, vb.pp("query_risk_level_head"))?;
        Ok(Self {
            body,
            pre,
            pre_norm,
            risk_head,
            device,
        })
    }

    /// Forward the prompt once and return the query-side risk-level logits
    /// `[Safe, Unsafe, Controversial]` at the final position.
    pub fn risk_logits(&self, prompt_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        if prompt_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "prompt token ids are empty".to_owned(),
            ));
        }
        let hidden = self.body.last_hidden(prompt_ids, &self.device)?;
        let x = self.pre.forward(&hidden.unsqueeze(0)?)?;
        let x = self.pre_norm.forward(&x)?;
        let logits = self
            .risk_head
            .forward(&x)?
            .squeeze(0)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        if logits.len() != 3 {
            return Err(FamilyError::ContractMismatch {
                field: "risk_head.classes",
                expected: "3 risk-level logits".to_owned(),
                actual: format!("{} logits", logits.len()),
            });
        }
        Ok([
            f64::from(logits[super::CLASS_SAFE]),
            f64::from(logits[super::CLASS_UNSAFE]),
            f64::from(logits[super::CLASS_CONTROVERSIAL]),
        ])
    }
}
