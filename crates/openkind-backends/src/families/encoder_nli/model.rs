//! Pinned DistilBERT MNLI checkpoint loading and forward execution.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Activation, Linear, Module};
use candle_transformers::models::distilbert::{Config, DistilBertModel};

use crate::families::support::{verify_digest, FamilyError};

use super::{pinned_config, CHECKPOINT_SHA256, CONFIG_JSON_SHA256, VOCAB_TXT_SHA256};

/// The pinned encoder-NLI model: DistilBERT body plus the sequence-
/// classification head (`pre_classifier` + `classifier`), FP32 on CPU.
pub struct EncoderNliModel {
    model: DistilBertModel,
    pre_classifier: Linear,
    classifier: Linear,
    device: Device,
}

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified checkpoint shard.
    pub checkpoint: std::path::PathBuf,
    /// Decoded, contract-checked candle `distilbert` config.
    pub config: Config,
    /// Path to the verified WordPiece vocabulary.
    pub vocab: std::path::PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place and decode the model config.
    pub fn verify(model_root: &Path) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join("model.safetensors");
        let vocab = model_root.join("vocab.txt");
        let config_path = model_root.join("config.json");
        for path in [&checkpoint, &vocab, &config_path] {
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
        verify_digest(&config_path, CONFIG_JSON_SHA256)?;
        verify_digest(&vocab, VOCAB_TXT_SHA256)?;
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
            vocab,
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

impl EncoderNliModel {
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
        // `DistilBertModel::load` falls back to the `distilbert.` prefix on
        // its own, so the full checkpoint builder works directly.
        let model = DistilBertModel::load(vb.clone(), &artifacts.config)?;
        let dim = 768_usize;
        let labels = 3_usize;
        let pre_classifier = candle_nn::linear(dim, dim, vb.pp("pre_classifier"))?;
        let classifier = candle_nn::linear(dim, labels, vb.pp("classifier"))?;
        Ok(Self {
            model,
            pre_classifier,
            classifier,
            device,
        })
    }

    /// Forward one premise–hypothesis pair and return the three-way NLI
    /// logits in checkpoint label order
    /// (`[ENTAILMENT, NEUTRAL, CONTRADICTION]`).
    pub fn nli_logits(&self, pair_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        let length = pair_ids.len();
        if length == 0 {
            return Err(FamilyError::InvalidInput(
                "encoded pair is empty".to_owned(),
            ));
        }
        let input = Tensor::new(pair_ids, &self.device)?.unsqueeze(0)?;
        // The candle distilbert implementation fills attention scores with
        // -inf where the mask is non-zero, so an unpadded sequence passes an
        // all-zero (b, 1, 1, L) mask: nothing is masked.
        let mask = Tensor::zeros((1, 1, 1, length), DType::U8, &self.device)?;
        let hidden = self.model.forward(&input, &mask)?;
        let cls = hidden.i((0, 0))?.unsqueeze(0)?;
        let pooled = self
            .pre_classifier
            .forward(&cls)?
            .apply(&Activation::Relu)?;
        let logits = self
            .classifier
            .forward(&pooled)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        if logits.len() != 3 {
            return Err(FamilyError::ContractMismatch {
                field: "classifier.labels",
                expected: "3 NLI logits".to_owned(),
                actual: format!("{} logits", logits.len()),
            });
        }
        Ok([
            f64::from(logits[super::LABEL_ENTAILMENT]),
            f64::from(logits[super::LABEL_NEUTRAL]),
            f64::from(logits[super::LABEL_CONTRADICTION]),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_field_resolution_walks_nested_paths() {
        let value = serde_json::json!({
            "architectures": ["DistilBertForSequenceClassification"],
            "id2label": { "0": "ENTAILMENT" }
        });
        assert_eq!(
            resolve_config_field(&value, "architectures[0]"),
            Some(serde_json::json!("DistilBertForSequenceClassification"))
        );
        assert_eq!(
            resolve_config_field(&value, "id2label.0"),
            Some(serde_json::json!("ENTAILMENT"))
        );
        assert_eq!(resolve_config_field(&value, "missing.path"), None);
    }
}
