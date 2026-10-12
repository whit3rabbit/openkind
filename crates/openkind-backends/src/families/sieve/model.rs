//! Pinned sieve checkpoint loading and forward execution.
//!
//! Loading verifies every pinned artifact by digest — the base `config.json`
//! and serving configs first so drift fails fast, then the base shards, the
//! LoRA adapter, and the trained head streamed in place — and never copies
//! weights. The forward reuses the parity-verified native Qwen3.5 layer
//! implementations through the shared [`TextBackbone`] over the shared
//! LoRA-merged backend (the adapter delta is computed once per load), then
//! applies the trained 255-row float32 head at the final branch position.

use std::path::{Path, PathBuf};

use candle_core::Device;

use crate::families::decoder_logit_qwen35::model::resolve_config_field;
use crate::families::strands_decider::MergedLoRABackend;
use crate::families::support::{read_json, verify_digest, FamilyControl, FamilyError};
use crate::qwen35::{EmbeddingLayout, Qwen35Geometry, TextBackbone};

use super::{SieveProfile, CODE_TABLE_WIDTH};

/// Vocab size the base embedding table must have.
pub(super) const VOCAB_SIZE: usize = 248_320;

/// Name of the `[CODE_TABLE_WIDTH, hidden]` float32 head inside
/// `head.safetensors` (`sieve4b/model.py:trainable_head_state`).
pub(super) const HEAD_TENSOR: &str = "readout.weight";

/// Embedding tensor name inside the base checkpoint.
pub(super) const EMBEDDING_TENSOR: &str = "model.language_model.embed_tokens.weight";

/// Digest-verified, contract-checked artifacts required by the loader.
///
/// The model root is the installation root: `base/` holds the pinned base
/// checkpoint shards and config, `checkpoint/` holds the serving checkpoint
/// (adapter, head, sieve config, tokenizer).
pub struct VerifiedArtifacts {
    /// Paths of the verified base checkpoint shards, in pinned order.
    pub base_shards: Vec<PathBuf>,
    /// Path to the verified LoRA adapter.
    pub adapter: PathBuf,
    /// Path to the verified trained head.
    pub head: PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts of `profile` in place.
    ///
    /// Cheap contract checks run first so a drifted config or tokenizer
    /// fails before the multi-gigabyte shards are streamed for their
    /// digests.
    pub fn verify(model_root: &Path, profile: &SieveProfile) -> Result<Self, FamilyError> {
        let base_dir = model_root.join("base");
        let checkpoint_dir = model_root.join("checkpoint");
        let base_config = base_dir.join("config.json");
        let sieve_config = checkpoint_dir.join("sieve_config.json");
        let tokenizer = checkpoint_dir.join("tokenizer.json");
        let adapter = checkpoint_dir.join("adapter_model.safetensors");
        let head = checkpoint_dir.join("head.safetensors");
        for path in [
            &base_config,
            &sieve_config,
            &tokenizer,
            checkpoint_dir.join("adapter_config.json").as_path(),
            checkpoint_dir.join("tokenizer_config.json").as_path(),
            checkpoint_dir.join("provenance.json").as_path(),
            checkpoint_dir.join("LICENSE").as_path(),
            &head,
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
        verify_digest(&base_config, profile.base_config_sha256)?;
        for (name, bytes, sha256) in profile.checkpoint_files {
            let path = checkpoint_dir.join(name);
            if !path.is_file() {
                return Err(FamilyError::Io {
                    path: path.clone(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("required pinned checkpoint file {name} is missing"),
                    ),
                });
            }
            let metadata = std::fs::metadata(&path).map_err(|source| FamilyError::Io {
                path: path.clone(),
                source,
            })?;
            if metadata.len() != *bytes {
                return Err(FamilyError::ContractMismatch {
                    field: "checkpoint file size",
                    expected: format!("{name} is {bytes} bytes"),
                    actual: format!("{} bytes", metadata.len()),
                });
            }
            verify_digest(&path, sha256)?;
        }

        let config_json: serde_json::Value = read_json(&base_config)?;
        for (field, expected) in pinned_base_config(profile) {
            let actual = resolve_config_field(&config_json, field).ok_or_else(|| {
                FamilyError::ContractMismatch {
                    field,
                    expected: expected.to_string(),
                    actual: "missing".to_owned(),
                }
            })?;
            // RMS epsilon is pinned as f32; a config's decimal `1e-6` is the
            // f64 nearest that value, so compare through the f32 rounding.
            let matches = if field == "text_config.rms_norm_eps" {
                actual
                    .as_f64()
                    .is_some_and(|value| value as f32 == profile.geometry.rms_epsilon)
            } else {
                actual == expected
            };
            if !matches {
                return Err(FamilyError::contract(field, expected, actual));
            }
        }
        let sieve_json: serde_json::Value = read_json(&sieve_config)?;
        for (field, expected) in pinned_sieve_config(profile) {
            let actual = resolve_config_field(&sieve_json, field).ok_or_else(|| {
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

        let mut base_shards = Vec::with_capacity(profile.base_shards.len());
        for (name, bytes, sha256) in profile.base_shards {
            let path = base_dir.join(name);
            if !path.is_file() {
                return Err(FamilyError::Io {
                    path: path.clone(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("required pinned base shard {name} is missing"),
                    ),
                });
            }
            let metadata = std::fs::metadata(&path).map_err(|source| FamilyError::Io {
                path: path.clone(),
                source,
            })?;
            if metadata.len() != *bytes {
                return Err(FamilyError::ContractMismatch {
                    field: "base shard size",
                    expected: format!("{name} is {bytes} bytes"),
                    actual: format!("{} bytes", metadata.len()),
                });
            }
            verify_digest(&path, sha256)?;
            base_shards.push(path);
        }
        Ok(Self {
            base_shards,
            adapter,
            head,
            tokenizer,
        })
    }
}

/// Pinned base `config.json` values derived from the profile geometry.
pub(crate) fn pinned_base_config(profile: &SieveProfile) -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    let geometry = profile.geometry;
    vec![
        ("model_type", json!("qwen3_5")),
        ("text_config.hidden_size", json!(geometry.hidden_size)),
        (
            "text_config.intermediate_size",
            json!(geometry.intermediate_size),
        ),
        ("text_config.num_hidden_layers", json!(geometry.layer_count)),
        (
            "text_config.full_attention_interval",
            json!(geometry.full_attention_interval),
        ),
        (
            "text_config.linear_num_key_heads",
            json!(geometry.key_heads),
        ),
        (
            "text_config.linear_num_value_heads",
            json!(geometry.value_heads),
        ),
        ("text_config.linear_key_head_dim", json!(geometry.head_dim)),
        (
            "text_config.linear_value_head_dim",
            json!(geometry.head_dim),
        ),
        (
            "text_config.linear_conv_kernel_dim",
            json!(geometry.conv_kernel),
        ),
        (
            "text_config.num_attention_heads",
            json!(geometry.attention_heads),
        ),
        ("text_config.num_key_value_heads", json!(geometry.kv_heads)),
        ("text_config.head_dim", json!(geometry.attention_head_dim)),
        ("text_config.rms_norm_eps", json!(geometry.rms_epsilon)),
        ("text_config.vocab_size", json!(VOCAB_SIZE)),
    ]
}

/// Pinned `sieve_config.json` values the loader enforces before trusting
/// weights.
pub(crate) fn pinned_sieve_config(
    profile: &SieveProfile,
) -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("base", json!(profile.base_id)),
        ("revision", json!(profile.base_revision)),
        ("pointer", json!(false)),
    ]
}

/// Read the pinned `[rows, hidden]` float32 head out of a verified
/// safetensors file.
fn read_head(path: &Path, rows: usize, hidden: usize) -> Result<Vec<f32>, FamilyError> {
    let head_path = path.to_path_buf();
    // SAFETY: the caller verified the immutable size and SHA-256 of the
    // read-only file before this mapper opens it. The VarBuilder owns the
    // mapped tensor storage.
    let variables = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&head_path),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .map_err(|error| FamilyError::InvalidInput(error.to_string()))?
    };
    let tensor = variables
        .get((rows, hidden), HEAD_TENSOR)
        .map_err(|error| FamilyError::ContractMismatch {
            field: "head.safetensors:readout.weight",
            expected: format!("[{rows}, {hidden}] float32"),
            actual: format!("{error}"),
        })?;
    let flattened = tensor
        .flatten_all()
        .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
    flattened
        .to_vec1::<f32>()
        .map_err(|error| FamilyError::InvalidInput(format!("head tensor unreadable: {error}")))
}

/// The pinned sieve model: LoRA-merged causal backbone plus trained head.
pub struct SieveModel {
    backbone: TextBackbone,
    /// Row-major `[CODE_TABLE_WIDTH, hidden]` float32 head rows.
    readout: Vec<f32>,
    hidden_size: usize,
}

impl SieveModel {
    /// Load the verified base checkpoint, merge the adapter, and build the
    /// FP32 CPU backbone plus head.
    pub fn load(
        artifacts: &VerifiedArtifacts,
        profile: &SieveProfile,
    ) -> Result<Self, FamilyError> {
        Self::load_with_device(artifacts, profile, Device::Cpu)
    }

    /// Load the verified checkpoint onto `device`.
    pub fn load_with_device(
        artifacts: &VerifiedArtifacts,
        profile: &SieveProfile,
        device: Device,
    ) -> Result<Self, FamilyError> {
        let geometry: Qwen35Geometry = profile.geometry;
        let first_shard = artifacts
            .base_shards
            .first()
            .ok_or_else(|| FamilyError::InvalidInput("profile pins no base shard".into()))?;
        let layout = EmbeddingLayout::read_named(
            first_shard,
            EMBEDDING_TENSOR,
            VOCAB_SIZE,
            geometry.hidden_size,
        )
        .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        let embedding = crate::qwen35::Qwen35Embedding::from_layout(first_shard.clone(), layout);
        let merged = MergedLoRABackend::build(
            &artifacts.base_shards,
            &artifacts.adapter,
            &device,
            profile.lora_rank,
            profile.lora_scale,
        )?;
        let backbone = TextBackbone::new_with_adapter_backend(
            embedding,
            std::sync::Arc::new(merged),
            geometry,
            device,
        );
        let readout = read_head(&artifacts.head, CODE_TABLE_WIDTH, geometry.hidden_size)?;
        Ok(Self {
            backbone,
            readout,
            hidden_size: geometry.hidden_size,
        })
    }

    /// Forward the prompt once and return the code logits restricted to the
    /// first `count` head rows (the offered options' codes; the reference
    /// masks the remaining rows before the softmax, which excluding them
    /// reproduces exactly).
    ///
    /// The readout position is the final prompt position (the assistant
    /// tail); the row is prefix and branch concatenated, so the causal
    /// mask lets the tail attend to the whole prompt.
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
        let hidden = self
            .backbone
            .forward_hidden_with_check(prompt_ids, || control.check())?;
        control.check()?;
        let token_start = (prompt_ids.len() - 1) * self.hidden_size;
        let final_token = &hidden[token_start..];
        let mut logits = Vec::with_capacity(count);
        for row in self.readout.chunks_exact(self.hidden_size).take(count) {
            let logit: f64 = row
                .iter()
                .zip(final_token)
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
