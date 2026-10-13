//! Pinned laya checkpoint loading and the encoder+head forward path.

use std::path::Path;

use candle_core::{DType, Device};

use crate::families::modernbert::ModernBertModel;
use crate::families::support::{
    check_pinned_fields_tolerant, require_file, verify_digest, FamilyError,
};

use super::arch::{LayaDecisionHead, LayaHeadConfig};
use super::{LayaProfile, LayaSpecials};

#[cfg(test)]
mod load_order_tests;

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
        Self::verify_artifacts(model_root, profile, true)
    }

    // ONNX needs the pinned renderer/config plus its own digest manifest.
    // Native weights are optional there, but any present bytes still verify.
    pub(crate) fn verify_for_execution(
        model_root: &Path,
        profile: &LayaProfile,
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
        profile: &LayaProfile,
        native: bool,
    ) -> Result<Self, FamilyError> {
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
            if path == &checkpoint && (!native) && !path.exists() {
                continue;
            }
            require_file(path)?;
        }
        // Cheap contract checks first; the checkpoint digest streams last.
        verify_digest(&agent_config, profile.agent_config_sha256)?;
        let agent_json: serde_json::Value = crate::families::support::read_json(&agent_config)?;
        check_pinned_fields_tolerant(&agent_json, &profile.pinned_agent_config())?;
        verify_digest(&encoder_config, profile.encoder_config_sha256)?;
        let encoder_json: serde_json::Value = crate::families::support::read_json(&encoder_config)?;
        check_pinned_fields_tolerant(&encoder_json, &profile.pinned_encoder_config())?;
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
