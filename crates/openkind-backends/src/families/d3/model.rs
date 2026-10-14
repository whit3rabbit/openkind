//! Pinned d3 checkpoint loading and forward execution.
//!
//! Loading verifies every pinned artifact by digest — the small configs and
//! tokenizer first so drift fails fast, then each checkpoint shard and the
//! separate `readout.safetensors` streamed in place — and never copies
//! weights. The forward reuses the parity-verified native Qwen3.5 layer
//! implementations through the shared [`TextBackbone`] with the
//! profile's **noncausal** geometry and the text-only
//! `language_model.` tensor namespace, then applies the 255-row float32
//! readout matrix at the final prompt position — the `d3-code-readout-v1`
//! last-position pooling.

use std::path::{Path, PathBuf};

use candle_core::Device;

use crate::families::support::{
    check_pinned_fields, check_pinned_fields_with, read_json, require_file_labeled, verify_digest,
    verify_sized_digest, FamilyControl, FamilyError,
};
use crate::qwen35::{EmbeddingLayout, Qwen35Geometry, TextBackbone};

use super::{pinned_config, pinned_decision_config, D3Profile, CODE_TABLE_WIDTH, VOCAB_SIZE};

/// Tensor namespace of the text-only `Qwen3_5Model` checkpoints (no
/// `model.` prefix and no vision-tower indirection; the shipped vision
/// weights stay unused).
pub(super) const TENSOR_PREFIX: &[&str] = &["language_model"];

/// Name of the `[CODE_TABLE_WIDTH, hidden]` float32 readout matrix inside
/// `readout.safetensors`.
pub(super) const READOUT_TENSOR: &str = "weight";

/// Embedding tensor name inside the text-only checkpoints.
pub(super) const EMBEDDING_TENSOR: &str = "language_model.embed_tokens.weight";

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Paths of the verified checkpoint shards, in pinned order.
    pub shards: Vec<PathBuf>,
    /// Path to the verified `readout.safetensors`.
    pub readout: PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts of `profile` in place.
    ///
    /// Cheap contract checks run first so a drifted config, tokenizer, or
    /// `decision_config.json` digest fails before the multi-gigabyte
    /// checkpoint shards are streamed for their digests.
    pub fn verify(model_root: &Path, profile: &D3Profile) -> Result<Self, FamilyError> {
        let config_path = model_root.join("config.json");
        let decision_config_path = model_root.join("decision_config.json");
        let tokenizer = model_root.join("tokenizer.json");
        let tokenizer_config = model_root.join("tokenizer_config.json");
        let chat_template = model_root.join("chat_template.jinja");
        let readout = model_root.join("readout.safetensors");
        let small: [(&Path, &str, &str); 5] = [
            (&config_path, "config.json", profile.config_json_sha256),
            (
                &decision_config_path,
                "decision_config.json",
                profile.decision_config_sha256,
            ),
            (&tokenizer, "tokenizer.json", profile.tokenizer_json_sha256),
            (
                &tokenizer_config,
                "tokenizer_config.json",
                profile.tokenizer_config_json_sha256,
            ),
            (
                &chat_template,
                "chat_template.jinja",
                profile.chat_template_sha256,
            ),
        ];
        for (path, name, _) in &small {
            require_file_labeled(path, &format!("artifact {name}"))?;
        }
        for (path, _, expected) in &small {
            verify_digest(path, expected)?;
        }

        let config_json: serde_json::Value = read_json(&config_path)?;
        check_pinned_fields_with(
            &config_json,
            &pinned_config(profile),
            |field, expected, actual| {
                // RMS epsilon is pinned as f32; a config's decimal `1e-6` is
                // the f64 nearest that value, so compare through the f32
                // rounding.
                if field == "text_config.rms_norm_eps" {
                    actual
                        .as_f64()
                        .is_some_and(|value| value as f32 == profile.geometry.rms_epsilon)
                } else {
                    actual == expected
                }
            },
        )?;
        let decision_config: serde_json::Value = read_json(&decision_config_path)?;
        check_pinned_fields(&decision_config, &pinned_decision_config(profile))?;

        let mut shards = Vec::with_capacity(profile.shards.len());
        for (name, bytes, sha256) in profile.shards {
            let path = model_root.join(name);
            require_file_labeled(&path, &format!("shard {name}"))?;
            verify_sized_digest(&path, name, *bytes, sha256, "checkpoint shard size")?;
            shards.push(path);
        }
        require_file_labeled(&readout, "artifact readout.safetensors")?;
        verify_digest(&readout, profile.readout_sha256)?;
        Ok(Self {
            shards,
            readout,
            tokenizer,
        })
    }
}

/// The pinned d3 model: verified noncausal backbone plus code readout.
pub struct D3Model {
    backbone: TextBackbone,
    /// Row-major `[CODE_TABLE_WIDTH, hidden]` float32 readout rows.
    readout: Vec<f32>,
    hidden_size: usize,
}

impl D3Model {
    /// Load the verified checkpoint into the shared FP32 CPU backbone.
    pub fn load(artifacts: &VerifiedArtifacts, profile: &D3Profile) -> Result<Self, FamilyError> {
        Self::load_with_device(artifacts, profile, Device::Cpu)
    }

    /// Load the verified checkpoint onto `device`.
    ///
    /// The decoder forward executes on `device`; the readout matvec stays
    /// host-side, exactly as on the CPU reference path.
    pub fn load_with_device(
        artifacts: &VerifiedArtifacts,
        profile: &D3Profile,
        device: Device,
    ) -> Result<Self, FamilyError> {
        let geometry: Qwen35Geometry = profile.geometry;
        let shard = artifacts
            .shards
            .first()
            .ok_or_else(|| FamilyError::InvalidInput("profile pins no checkpoint shard".into()))?;
        let layout =
            EmbeddingLayout::read_named(shard, EMBEDDING_TENSOR, VOCAB_SIZE, geometry.hidden_size)
                .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        let embedding = crate::qwen35::Qwen35Embedding::from_layout(shard.clone(), layout);
        let backbone = TextBackbone::new_with_tensor_prefix(
            embedding,
            artifacts.shards.clone(),
            geometry,
            TENSOR_PREFIX,
            device,
        );
        let readout = read_rows(&artifacts.readout, CODE_TABLE_WIDTH, geometry.hidden_size)?;
        Ok(Self {
            backbone,
            readout,
            hidden_size: geometry.hidden_size,
        })
    }

    /// Forward the prompt once and return the code logits restricted to the
    /// first `count` readout rows (the offered options' codes; the
    /// reference masks the remaining rows to negative infinity before the
    /// softmax, which excluding them reproduces exactly).
    ///
    /// The readout position is the final prompt position (the
    /// `<think>\n\n</think>\n\n` generation-prompt tail); pooling is
    /// `last` per the pinned `decision_config.json`. Nothing is sampled
    /// and no continuation token exists.
    pub fn code_logits(
        &self,
        prompt_ids: &[u32],
        count: usize,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        if count == 0 || count > CODE_TABLE_WIDTH {
            return Err(FamilyError::InvalidInput(format!(
                "code readout serves 1..={CODE_TABLE_WIDTH} codes, found {count}"
            )));
        }
        // A question may carry up to `max_row_tokens` prompt tokens and each
        // full-attention layer is quadratic in them, so cancellation and
        // deadlines are also checked inside the attention kernel, not only
        // between layers.
        // The readout reads only the final token, so the last decoder layer
        // runs for that row alone.
        let final_token = self
            .backbone
            .forward_last_hidden_with_attention_check(prompt_ids, || control.check())?;
        control.check()?;
        let mut logits = Vec::with_capacity(count);
        for row in self.readout.chunks_exact(self.hidden_size).take(count) {
            let logit: f64 = row
                .iter()
                .zip(&final_token)
                .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                .sum();
            if !logit.is_finite() {
                return Err(FamilyError::Numerical(
                    "code readout produced a non-finite logit".to_owned(),
                ));
            }
            logits.push(logit);
        }
        Ok(logits)
    }
}

/// Read the pinned `[rows, hidden]` float32 matrix out of a verified
/// safetensors file.
fn read_rows(path: &Path, rows: usize, hidden: usize) -> Result<Vec<f32>, FamilyError> {
    let shard = path.to_path_buf();
    // SAFETY: the caller verified the immutable size and SHA-256 of the
    // read-only file before this mapper opens it. The VarBuilder owns the
    // mapped tensor storage.
    let variables = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&shard),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .map_err(|error| FamilyError::InvalidInput(error.to_string()))?
    };
    let tensor = variables
        .get((rows, hidden), READOUT_TENSOR)
        .map_err(|error| FamilyError::ContractMismatch {
            field: "readout.safetensors:weight",
            expected: format!("[{rows}, {hidden}] float32"),
            actual: format!("{error}"),
        })?;
    let flattened = tensor
        .flatten_all()
        .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
    flattened
        .to_vec1::<f32>()
        .map_err(|error| FamilyError::InvalidInput(format!("readout tensor unreadable: {error}")))
}
