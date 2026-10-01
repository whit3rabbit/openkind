//! ONNX execution adapter for the pinned encoder-NLI profile (feature
//! `onnx`).
//!
//! The adapter loads `model.onnx` from the profile root — exported from the
//! same pinned DistilBERT-MNLI checkpoint the candle loader verifies — and
//! reproduces the family readout through ONNX Runtime instead of candle:
//!
//! - inputs: `input_ids` (`int64`, `[1, L]`) and `attention_mask`
//!   (`int64`, `[1, L]`);
//! - output: `logits` (`float32`, `[1, 3]`) in the pinned checkpoint label
//!   order.
//!
//! Unpadded sequences pass an all-one mask, mirroring the candle path. See
//! `docs/ONNX.md` for the export contract and promotion gates.

use std::path::{Path, PathBuf};

use crate::families::support::FamilyError;
use crate::onnx::{OnnxAcceleration, OnnxError, OnnxModel, OnnxRuntimeSettings, OnnxTensorSpec};

use super::{LABEL_CONTRADICTION, LABEL_ENTAILMENT, LABEL_NEUTRAL};

/// Artifact file name inside the profile root.
const MODEL_FILE: &str = "model.onnx";
/// Required token-id input name.
const INPUT_IDS: &str = "input_ids";
/// Required attention-mask input name.
const ATTENTION_MASK: &str = "attention_mask";
/// Required logits output name.
const LOGITS: &str = "logits";
/// Number of NLI labels the artifact must emit.
const LABEL_COUNT: usize = 3;

/// ONNX-backed encoder-NLI model executing the pinned readout through ONNX
/// Runtime.
pub struct EncoderNliOnnxModel {
    model: OnnxModel,
}

impl EncoderNliOnnxModel {
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
            &[OnnxTensorSpec::f32(LOGITS, &[1, 3])],
        )?;
        Ok(Self { model })
    }

    /// Forward one premise–hypothesis pair and return the three-way NLI
    /// logits in checkpoint label order
    /// (`[ENTAILMENT, NEUTRAL, CONTRADICTION]`), matching the candle model.
    ///
    /// # Errors
    /// Returns [`FamilyError`] for session failures and readout-shape
    /// mismatches.
    pub fn nli_logits(&self, pair_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        let length = pair_ids.len();
        if length == 0 {
            return Err(FamilyError::InvalidInput(
                "encoded pair is empty".to_owned(),
            ));
        }
        let input_ids: Vec<i64> = pair_ids.iter().map(|&id| i64::from(id)).collect();
        // Unpadded sequence: every position is attended, mirroring the
        // candle path's all-zero (nothing-masked) tensor.
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
        let logits: [f32; LABEL_COUNT] = self.model.run(inputs, |outputs| {
            let (shape, data) = outputs[LOGITS].try_extract_tensor::<f32>()?;
            if shape.len() != 2
                || shape[0] != 1
                || shape[1] != i64::try_from(LABEL_COUNT).unwrap_or(i64::MAX)
            {
                return Err(OnnxError::Signature {
                    artifact: PathBuf::new(),
                    message: format!("`{LOGITS}` must be [1, {LABEL_COUNT}], got {shape:?}"),
                });
            }
            Ok([data[0], data[1], data[2]])
        })?;
        Ok([
            f64::from(logits[LABEL_ENTAILMENT]),
            f64::from(logits[LABEL_NEUTRAL]),
            f64::from(logits[LABEL_CONTRADICTION]),
        ])
    }
}
