//! Pinned GLiClass uni-encoder checkpoint loading and head execution.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Activation, Linear, Module};

use crate::families::support::{verify_digest, FamilyError};

use crate::families::modernbert::{ModernBertConfig, ModernBertModel};

use super::{pinned_config, CHECKPOINT_SHA256, CONFIG_JSON_SHA256, TOKENIZER_JSON_SHA256};

/// Build the ModernBERT-base configuration from the pinned checkpoint values.
fn config_from_pinned(max_sequence_tokens: usize) -> ModernBertConfig {
    ModernBertConfig {
        vocab_size: 50_370,
        hidden_size: 768,
        num_attention_heads: 12,
        num_hidden_layers: 22,
        intermediate_size: 1_152,
        local_attention: 128,
        global_attn_every_n_layers: 3,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
        norm_eps: 1e-5,
        max_sequence_tokens,
    }
}

/// The pinned label-marker model: ModernBERT body plus the GLiClass
/// projector pair and dot-product scorer.
pub struct EncoderInstructLabelModel {
    body: ModernBertModel,
    text_projector: Projector,
    classes_projector: Projector,
    device: Device,
}

/// Two bias `Linear` layers with a GELU activation between them.
struct Projector {
    linear_1: Linear,
    linear_2: Linear,
}

impl Projector {
    fn load(vb: candle_nn::VarBuilder) -> Result<Self, candle_core::Error> {
        Ok(Self {
            linear_1: linear(768, 768, vb.pp("linear_1"))?,
            linear_2: linear(768, 768, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.linear_2
            .forward(&self.linear_1.forward(xs)?.apply(&Activation::Gelu)?)
    }
}

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified checkpoint shard.
    pub checkpoint: std::path::PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: std::path::PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place and contract-check the config.
    pub fn verify(model_root: &Path) -> Result<Self, FamilyError> {
        Self::verify_artifacts(model_root, true)
    }

    // ONNX needs the pinned renderer/config plus its own digest manifest.
    // Native weights are optional there, but any present bytes still verify.
    pub(crate) fn verify_for_execution(
        model_root: &Path,
        execution: crate::device::FamilyExecution,
    ) -> Result<Self, FamilyError> {
        if execution.is_onnx() {
            Self::verify_artifacts(model_root, false)
        } else {
            Self::verify(model_root)
        }
    }

    fn verify_artifacts(model_root: &Path, native: bool) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join("model.safetensors");
        let tokenizer = model_root.join("tokenizer.json");
        let config_path = model_root.join("config.json");
        for path in [&checkpoint, &tokenizer, &config_path] {
            if path == &checkpoint && (!native) && !path.exists() {
                continue;
            }
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
        verify_digest(&tokenizer, TOKENIZER_JSON_SHA256)?;
        if native || checkpoint.exists() {
            verify_digest(&checkpoint, CHECKPOINT_SHA256)?;
        }
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

impl EncoderInstructLabelModel {
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
        let body = ModernBertModel::load(
            &config_from_pinned(super::MAX_SEQUENCE_TOKENS),
            vb.pp("model").pp("encoder_model"),
        )?;
        let text_projector = Projector::load(vb.pp("model").pp("text_projector"))?;
        let classes_projector = Projector::load(vb.pp("model").pp("classes_projector"))?;
        Ok(Self {
            body,
            text_projector,
            classes_projector,
            device,
        })
    }

    /// One forward pass over the rendered marker+state sequence, returning
    /// one raw dot-product logit per `<<LABEL>>` marker in token order.
    ///
    /// The checkpoint's `logit_scale` is unused: the pinned config sets
    /// `normalize_features: false`, so the reference never applies it.
    pub fn marker_logits(&self, token_ids: &[u32]) -> Result<Vec<f64>, FamilyError> {
        if token_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "encoded sequence is empty".to_owned(),
            ));
        }
        let marker_positions: Vec<usize> = token_ids
            .iter()
            .enumerate()
            .filter(|&(_, &token)| token == super::CLASS_TOKEN_ID)
            .map(|(position, _)| position)
            .collect();
        if marker_positions.is_empty() {
            return Err(FamilyError::InvalidInput(
                "encoded sequence contains no `<<LABEL>>` marker".to_owned(),
            ));
        }
        let hidden = self.body.forward(token_ids, &self.device)?;
        // Text representation: hidden state at position 0 (the `[CLS]`
        // the tokenizer template prepends), projected. `hidden` is
        // `(seq, hidden_size)` after the body squeezes the batch axis.
        let text_hidden = hidden.narrow(0, 0, 1)?;
        let text_projected = self.text_projector.forward(&text_hidden)?;
        // Class representations: hidden states at the marker positions.
        let rows = hidden;
        let positions = Tensor::from_slice(
            &(marker_positions
                .iter()
                .map(|&p| p as u32)
                .collect::<Vec<u32>>()),
            marker_positions.len(),
            &self.device,
        )?;
        let class_hidden = rows.index_select(&positions, 0)?;
        let classes_projected = self.classes_projector.forward(&class_hidden)?;
        // Dot-product scorer: one logit per marker.
        let logits = classes_projected
            .broadcast_mul(&text_projected)?
            .sum(candle_core::D::Minus1)?;
        let logits = logits.to_vec1::<f32>()?;
        Ok(logits.into_iter().map(f64::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_field_resolution_walks_nested_paths() {
        let value = serde_json::json!({
            "architectures": ["GLiClassModel"],
            "encoder_config": { "model_type": "modernbert" }
        });
        assert_eq!(
            resolve_config_field(&value, "architectures[0]"),
            Some(serde_json::json!("GLiClassModel"))
        );
        assert_eq!(
            resolve_config_field(&value, "encoder_config.model_type"),
            Some(serde_json::json!("modernbert"))
        );
        assert_eq!(resolve_config_field(&value, "missing.path"), None);
    }
}
