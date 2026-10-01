//! Pinned cross-encoder checkpoint loading and forward execution.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module};
use candle_transformers::models::bert::{BertModel, Config};

use crate::families::support::{verify_digest, FamilyError};

use super::{pinned_config, CHECKPOINT_SHA256, CONFIG_JSON_SHA256, TOKENIZER_JSON_SHA256};

/// The pinned schema-scorer model: BERT body + pooler + scalar classifier,
/// FP32 on CPU. The scalar logit is the relevance score of one
/// query–passage pair.
pub struct SchemaScorerModel {
    model: BertModel,
    pooler: Linear,
    classifier: Linear,
    device: Device,
}

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified checkpoint shard.
    pub checkpoint: std::path::PathBuf,
    /// Decoded, contract-checked candle `bert` config.
    pub config: Config,
    /// Path to the verified tokenizer.
    pub tokenizer: std::path::PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place and decode the model config.
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

impl SchemaScorerModel {
    /// Load the verified checkpoint into FP32 weights on `device`.
    ///
    /// The reference path passes the CPU device; accelerated loads thread a
    /// resolved CUDA device through the same candle model code.
    pub fn load(artifacts: &VerifiedArtifacts, device: Device) -> Result<Self, FamilyError> {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&artifacts.checkpoint),
                DType::F32,
                &device,
            )
        }?;
        // `BertModel::load` falls back to the `bert.` prefix on its own.
        let model = BertModel::load(vb.clone(), &artifacts.config)?;
        let hidden = 384_usize;
        // The cross-encoder classification head: tanh pooler over `[CLS]`,
        // then the single-logit classifier.
        let pooler = candle_nn::linear(hidden, hidden, vb.pp("bert.pooler.dense"))?;
        let classifier = candle_nn::linear(hidden, 1, vb.pp("classifier"))?;
        Ok(Self {
            model,
            pooler,
            classifier,
            device,
        })
    }

    /// Forward one query–passage pair and return the scalar relevance logit.
    pub fn relevance_logit(&self, pair_ids: &[u32]) -> Result<f64, FamilyError> {
        let length = pair_ids.len();
        if length == 0 {
            return Err(FamilyError::InvalidInput(
                "encoded pair is empty".to_owned(),
            ));
        }
        let input = Tensor::new(pair_ids, &self.device)?.unsqueeze(0)?;
        // Token-type ids: sequence A for every position, batched shape
        // (1, L) so the embedding add stays batch-aligned.
        let token_type_ids = Tensor::zeros((1, length), candle_core::DType::U32, &self.device)?;
        let hidden = self.model.forward(&input, &token_type_ids, None)?;
        let cls = hidden.i((0, 0))?.unsqueeze(0)?;
        let pooled = cls.apply(&self.pooler)?.tanh()?;
        let logit = self.classifier.forward(&pooled)?.squeeze(0)?.squeeze(0)?;
        Ok(f64::from(logit.to_scalar::<f32>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_field_resolution_walks_nested_paths() {
        let value = serde_json::json!({
            "architectures": ["BertForSequenceClassification"],
            "id2label": { "0": "LABEL_0" }
        });
        assert_eq!(
            resolve_config_field(&value, "architectures[0]"),
            Some(serde_json::json!("BertForSequenceClassification"))
        );
        assert_eq!(
            resolve_config_field(&value, "id2label.0"),
            Some(serde_json::json!("LABEL_0"))
        );
    }
}
