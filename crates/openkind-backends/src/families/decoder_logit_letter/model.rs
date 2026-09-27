//! Pinned Qwen2.5-0.5B-Instruct checkpoint loading and forward execution.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};

use crate::families::support::{verify_digest, FamilyError};

use super::{pinned_config, CHECKPOINT_SHA256, CONFIG_JSON_SHA256, TOKENIZER_JSON_SHA256};

/// The pinned decoder-letter model: loader + forward over the candle `qwen2`
/// implementation, FP32 on CPU.
pub struct DecoderLetterModel {
    // `ModelForCausalLM` accumulates KV cache inside `forward(&mut self)`;
    // evaluation is serialized through this mutex and the cache is cleared
    // after every question, so no state survives a request.
    model: Mutex<ModelForCausalLM>,
    device: Device,
}

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified checkpoint shard.
    pub checkpoint: std::path::PathBuf,
    /// Decoded, contract-checked candle `qwen2` config.
    pub config: Config,
    /// Path to the verified tokenizer.
    pub tokenizer: std::path::PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place and decode the model config.
    ///
    /// Checkpoint shards are multi-hundred-megabyte files: verification
    /// streams them where they live and nothing is copied to temporary
    /// storage.
    pub fn verify(model_root: &Path) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join("model.safetensors");
        let tokenizer_path = model_root.join("tokenizer.json");
        let config_path = model_root.join("config.json");
        for path in [&checkpoint, &tokenizer_path, &config_path] {
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
        // Cheap contract checks first: config digest/contract and tokenizer
        // digest fail fast before the multi-hundred-megabyte checkpoint is
        // streamed for its digest.
        verify_digest(&config_path, CONFIG_JSON_SHA256)?;
        verify_digest(&tokenizer_path, TOKENIZER_JSON_SHA256)?;
        verify_digest(&checkpoint, CHECKPOINT_SHA256)?;

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
        let config: Config =
            serde_json::from_value(config_json).map_err(|error| FamilyError::Json {
                path: config_path.to_path_buf(),
                source: error,
            })?;
        Ok(Self {
            checkpoint,
            config,
            tokenizer: tokenizer_path,
        })
    }
}

fn resolve_config_field(value: &serde_json::Value, field: &str) -> Option<serde_json::Value> {
    if let Some(name) = field.strip_suffix("[0]") {
        return value.get(name).and_then(|items| items.get(0)).cloned();
    }
    value.get(field).cloned()
}

impl DecoderLetterModel {
    /// Load the verified checkpoint into FP32 CPU weights.
    pub fn load(artifacts: &VerifiedArtifacts) -> Result<Self, FamilyError> {
        let device = Device::Cpu;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&artifacts.checkpoint),
                DType::F32,
                &device,
            )
        }?;
        let model = ModelForCausalLM::new(&artifacts.config, vb)?;
        Ok(Self {
            model: Mutex::new(model),
            device,
        })
    }

    /// Forward the prompt once and return the next-token logits restricted to
    /// `letter_ids`. The answer slot is the final prompt position; nothing is
    /// sampled and no continuation tokens exist.
    pub fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<Vec<f64>, FamilyError> {
        if prompt_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "prompt token ids are empty".to_owned(),
            ));
        }
        let input = Tensor::new(prompt_ids, &self.device)?.unsqueeze(0)?;
        let logits = {
            let mut model = self.model.lock().map_err(|_| {
                FamilyError::InvalidInput("decoder-letter model lock poisoned".to_owned())
            })?;
            // (1, 1, vocab) logits for the final prompt position.
            let logits = model.forward(&input, 0)?;
            model.clear_kv_cache();
            logits
        };
        let logits = logits.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        let selected: Vec<f64> = letter_ids
            .iter()
            .map(|token| {
                usize::try_from(*token)
                    .ok()
                    .and_then(|index| logits.get(index))
                    .map(|logit| f64::from(*logit))
                    .ok_or_else(|| FamilyError::ContractMismatch {
                        field: "letter_token_contract",
                        expected: "letter token inside the model vocabulary".to_owned(),
                        actual: format!("letter token id {token} out of range"),
                    })
            })
            .collect::<Result<Vec<f64>, FamilyError>>()?;
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_field_resolution_reads_architecture_lists() {
        let value = serde_json::json!({"architectures": ["Qwen2ForCausalLM"]});
        assert_eq!(
            resolve_config_field(&value, "architectures[0]"),
            Some(serde_json::json!("Qwen2ForCausalLM"))
        );
        assert_eq!(resolve_config_field(&value, "architectures[1]"), None);
        assert_eq!(resolve_config_field(&value, "model_type"), None);
    }
}
