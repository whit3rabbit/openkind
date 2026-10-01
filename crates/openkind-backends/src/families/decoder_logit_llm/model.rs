//! Pinned Qwen2.5-0.5B-Instruct GGUF (q8_0) checkpoint loading and forward
//! execution through the candle quantized-runner binding.

use std::path::Path;
use std::sync::Mutex;

use candle_core::quantized::gguf_file as gguf;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;

use crate::families::support::{verify_digest, FamilyError};

use super::{pinned_metadata, CHECKPOINT_SHA256, GGUF_VARIANT, TOKENIZER_JSON_SHA256};

/// The pinned GGUF model: loader + forward over the candle quantized
/// `qwen2` runner on CPU.
pub struct DecoderLlmModel {
    // Quantized runners accumulate KV cache inside `forward(&mut self)`;
    // evaluation is serialized through this mutex and the cache resets
    // because every question runs from `index_pos = 0`.
    model: Mutex<ModelWeights>,
    device: Device,
}

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified GGUF checkpoint.
    pub checkpoint: std::path::PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: std::path::PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place.
    ///
    /// The GGUF checkpoint is a single multi-hundred-megabyte file:
    /// verification streams it where it lives and nothing is copied.
    pub fn verify(model_root: &Path) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join(GGUF_VARIANT);
        let tokenizer = model_root.join("tokenizer.json");
        for path in [&checkpoint, &tokenizer] {
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
        verify_digest(&tokenizer, TOKENIZER_JSON_SHA256)?;
        verify_digest(&checkpoint, CHECKPOINT_SHA256)?;
        Ok(Self {
            checkpoint,
            tokenizer,
        })
    }
}

fn metadata_value<'a>(
    content: &'a gguf::Content,
    key: &str,
) -> Result<&'a gguf::Value, FamilyError> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| FamilyError::ContractMismatch {
            field: "gguf.metadata",
            expected: format!("`{key}` present"),
            actual: "missing".to_owned(),
        })
}

impl DecoderLlmModel {
    /// Load the verified GGUF checkpoint through the candle quantized runner.
    pub fn load(artifacts: &VerifiedArtifacts) -> Result<Self, FamilyError> {
        Self::load_with_device(artifacts, Device::Cpu)
    }

    /// Load the verified GGUF checkpoint onto `device`.
    ///
    /// Quantized GGUF execution follows the candle quantized runner; CUDA
    /// requires the `cuda` feature and fails closed when unavailable.
    pub fn load_with_device(
        artifacts: &VerifiedArtifacts,
        device: Device,
    ) -> Result<Self, FamilyError> {
        let file =
            std::fs::File::open(&artifacts.checkpoint).map_err(|source| FamilyError::Io {
                path: artifacts.checkpoint.clone(),
                source,
            })?;
        let mut reader = std::io::BufReader::new(file);
        let content = gguf::Content::read(&mut reader).map_err(FamilyError::Candle)?;
        for (key, expected) in pinned_metadata() {
            let actual = match metadata_value(&content, key)? {
                gguf::Value::String(value) => value.clone(),
                gguf::Value::U32(value) => value.to_string(),
                gguf::Value::U64(value) => value.to_string(),
                gguf::Value::F32(value) => value.to_string(),
                gguf::Value::F64(value) => value.to_string(),
                gguf::Value::Bool(value) => value.to_string(),
                other => format!("{other:?}"),
            };
            if actual != expected {
                return Err(FamilyError::contract(key, expected, actual));
            }
        }
        let model =
            ModelWeights::from_gguf(content, &mut reader, &device).map_err(FamilyError::Candle)?;
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
                FamilyError::InvalidInput("decoder-llm model lock poisoned".to_owned())
            })?;
            // (1, vocab) logits for the final prompt position; the runner
            // never advances past `index_pos = 0` for a fresh prompt, so no
            // cache survives between questions.
            model.forward(&input, 0).map_err(FamilyError::Candle)?
        };
        let logits = logits.squeeze(0)?.to_vec1::<f32>()?;
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
