//! ONNX execution adapter for the pinned decoder-letter profile (feature
//! `onnx`).
//!
//! The adapter loads `model.onnx` from the profile root — exported from the
//! same pinned `Qwen/Qwen2.5-0.5B-Instruct` checkpoint the candle loader
//! verifies, with the language-model head folded into the graph — and
//! reproduces the family readout through ONNX Runtime instead of candle:
//!
//! - inputs: `input_ids` (`int64`, `[1, L]`) and `attention_mask` (`int64`,
//!   `[1, L]`);
//! - output: `logits` (`float32`, `[1, L, vocab]`); the adapter reads the
//!   last-position row — the answer slot — and restricts it to the option
//!   letter tokens exactly as the candle path does.
//!
//! Unpadded sequences pass an all-one mask, mirroring the candle path. See
//! `docs/ONNX.md` for the export contract and promotion gates.

use std::path::{Path, PathBuf};

use crate::families::support::FamilyError;
use crate::onnx::{OnnxAcceleration, OnnxError, OnnxModel, OnnxRuntimeSettings, OnnxTensorSpec};

/// Artifact file name inside the profile root.
const MODEL_FILE: &str = "model.onnx";
/// Required token-id input name.
const INPUT_IDS: &str = "input_ids";
/// Required attention-mask input name.
const ATTENTION_MASK: &str = "attention_mask";
/// Required logits output name.
const LOGITS: &str = "logits";

/// ONNX-backed decoder-letter model executing the pinned letter-logit
/// readout through ONNX Runtime.
pub struct DecoderLetterOnnxModel {
    model: OnnxModel,
}

impl DecoderLetterOnnxModel {
    /// Load `model.onnx` from the profile root with the selected execution
    /// provider.
    ///
    /// # Errors
    /// Fails closed on a missing artifact, an unloadable runtime library, an
    /// unavailable CUDA execution provider, or a signature mismatch.
    pub fn load(model_root: &Path, acceleration: OnnxAcceleration) -> Result<Self, FamilyError> {
        let model = OnnxModel::load(
            &model_root.join(MODEL_FILE),
            &OnnxRuntimeSettings::system(),
            acceleration,
            &[
                OnnxTensorSpec::i64(INPUT_IDS, &[1, -1]),
                OnnxTensorSpec::i64(ATTENTION_MASK, &[1, -1]),
            ],
            &[OnnxTensorSpec::f32(LOGITS, &[1, -1, -1])],
        )?;
        Ok(Self { model })
    }

    /// Forward the prompt once and return the next-token logits restricted to
    /// `letter_ids`, matching the candle model. The answer slot is the final
    /// prompt position; nothing is sampled and no continuation token exists.
    ///
    /// # Errors
    /// Returns [`FamilyError`] for session failures, readout-shape
    /// mismatches, and out-of-vocabulary letter tokens.
    pub fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<Vec<f64>, FamilyError> {
        let length = prompt_ids.len();
        if length == 0 {
            return Err(FamilyError::InvalidInput(
                "prompt token ids are empty".to_owned(),
            ));
        }
        let input_ids: Vec<i64> = prompt_ids.iter().map(|&id| i64::from(id)).collect();
        // Unpadded sequence: every position is attended, mirroring the
        // candle path's causal forward with nothing masked.
        let attention_mask = vec![1_i64; length];
        let inputs = vec![
            (
                INPUT_IDS.to_owned(),
                ort::value::Tensor::from_array((vec![1_usize, length], input_ids))
                    .map_err(OnnxError::from)?
                    .into_dyn(),
            ),
            (
                ATTENTION_MASK.to_owned(),
                ort::value::Tensor::from_array((vec![1_usize, length], attention_mask))
                    .map_err(OnnxError::from)?
                    .into_dyn(),
            ),
        ];
        let row: Vec<f32> = self.model.run(inputs, |outputs| {
            let (shape, data) = outputs[LOGITS].try_extract_tensor::<f32>()?;
            if shape.len() != 3
                || shape[0] != 1
                || shape[1] != i64::try_from(length).unwrap_or(i64::MAX)
            {
                return Err(OnnxError::Signature {
                    artifact: PathBuf::new(),
                    message: format!("`{LOGITS}` must be [1, {length}, vocab], got {shape:?}"),
                });
            }
            let vocab = usize::try_from(shape[2]).map_err(|_| OnnxError::Signature {
                artifact: PathBuf::new(),
                message: format!("`{LOGITS}` has an invalid vocab dimension {}", shape[2]),
            })?;
            let Some(expected) = vocab.checked_mul(length) else {
                return Err(OnnxError::Signature {
                    artifact: PathBuf::new(),
                    message: format!("`{LOGITS}` shape {shape:?} overflows the readout"),
                });
            };
            if vocab == 0 || data.len() != expected {
                return Err(OnnxError::Signature {
                    artifact: PathBuf::new(),
                    message: format!(
                        "`{LOGITS}` holds {} values, expected {expected} for {shape:?}",
                        data.len()
                    ),
                });
            }
            // Last-position row: the answer slot is the final prompt
            // position.
            Ok(data[(length - 1) * vocab..length * vocab].to_vec())
        })?;
        let selected: Vec<f64> = letter_ids
            .iter()
            .map(|token| {
                usize::try_from(*token)
                    .ok()
                    .and_then(|index| row.get(index))
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
