//! ONNX execution adapter for the pinned encoder-instruct-label profile
//! (feature `onnx`).
//!
//! The adapter loads `model.onnx` from the profile root — exported from the
//! same digest-verified GLiClass uni-encoder checkpoint the candle loader
//! verifies — and reproduces the family readout through ONNX Runtime
//! instead of candle. The artifact must contain the full decision head: the
//! ModernBERT encoder body, the gather at the `<<LABEL>>` class-marker
//! positions, both GLiClass projector pairs (`Linear → GELU → Linear`), the
//! position-0 text projection, and the dot-product scorer, so the adapter
//! feeds token ids and reads the final per-marker logits:
//!
//! - inputs: `input_ids` (`int64`, `[1, L]`), `attention_mask` (`int64`,
//!   `[1, L]`), and `marker_positions` (`int64`, `[K]`, the offsets of the
//!   `<<LABEL>>` class markers in token order). The marker positions are a
//!   runtime input because the class count varies per question; the
//!   artifact stays valid for every label count up to the trained
//!   `max_num_classes` of 25. The adapter derives the positions with the
//!   same vocabulary scan the candle model applies before its gather.
//! - output: `logits` (`float32`, `[1, K]`), one raw dot-product logit per
//!   marker in token order, matching what the candle `marker_logits`
//!   returns before the engine's calibration temperature.
//!
//! Unpadded sequences pass an all-one attention mask, mirroring the candle
//! path. See `docs/ONNX.md` for the export contract and promotion gates.

use std::path::{Path, PathBuf};

use crate::families::support::{onnx_marker_positions, FamilyError};
use crate::onnx::{OnnxAcceleration, OnnxError, OnnxModel, OnnxRuntimeSettings, OnnxTensorSpec};

use super::CLASS_TOKEN_ID;

/// Artifact file name inside the profile root.
const MODEL_FILE: &str = "model.onnx";
/// Required token-id input name.
const INPUT_IDS: &str = "input_ids";
/// Required attention-mask input name.
const ATTENTION_MASK: &str = "attention_mask";
/// Required marker-position input name.
const MARKER_POSITIONS: &str = "marker_positions";
/// Required logits output name.
const LOGITS: &str = "logits";

/// ONNX-backed GLiClass model executing the pinned label-marker readout
/// through ONNX Runtime.
pub struct EncoderInstructLabelOnnxModel {
    model: OnnxModel,
}

impl EncoderInstructLabelOnnxModel {
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
                OnnxTensorSpec::i64(MARKER_POSITIONS, &[-1]),
            ],
            &[OnnxTensorSpec::f32(LOGITS, &[1, -1])],
        )?;
        Ok(Self { model })
    }

    /// Forward one rendered marker+state sequence and return one raw
    /// dot-product logit per `<<LABEL>>` marker in token order, matching the
    /// candle model's `marker_logits` (before the engine's calibration
    /// temperature).
    ///
    /// # Errors
    /// Returns [`FamilyError`] for empty or marker-free inputs, session
    /// failures, and readout-shape mismatches.
    pub fn marker_logits(&self, token_ids: &[u32]) -> Result<Vec<f64>, FamilyError> {
        if token_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "encoded sequence is empty".to_owned(),
            ));
        }
        // Same vocabulary scan the candle model applies before its gather:
        // one class-marker position per rendered candidate, in token order.
        let marker_positions: Vec<usize> = token_ids
            .iter()
            .enumerate()
            .filter(|&(_, &token)| token == CLASS_TOKEN_ID)
            .map(|(position, _)| position)
            .collect();
        if marker_positions.is_empty() {
            return Err(FamilyError::InvalidInput(
                "encoded sequence contains no `<<LABEL>>` marker".to_owned(),
            ));
        }
        let length = token_ids.len();
        let input_ids: Vec<i64> = token_ids.iter().map(|&id| i64::from(id)).collect();
        // Unpadded sequence: every position is attended, mirroring the
        // candle path.
        let attention_mask = vec![1_i64; length];
        let positions = onnx_marker_positions(&marker_positions, length)?;
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
            (
                MARKER_POSITIONS.to_owned(),
                ort::value::Tensor::from_array((vec![marker_positions.len()], positions))
                    .map_err(OnnxError::from)?
                    .into_dyn(),
            ),
        ];
        let count = marker_positions.len();
        let count_i64 = i64::try_from(count).unwrap_or(i64::MAX);
        self.model
            .run(inputs, |outputs| {
                let (shape, data) = outputs[LOGITS].try_extract_tensor::<f32>()?;
                if shape.len() != 2 || shape[0] != 1 || shape[1] != count_i64 {
                    return Err(OnnxError::Signature {
                        artifact: PathBuf::new(),
                        message: format!("`{LOGITS}` must be [1, {count}], got {shape:?}"),
                    });
                }
                if data.len() != count {
                    return Err(OnnxError::Signature {
                        artifact: PathBuf::new(),
                        message: format!(
                            "`{LOGITS}` carried {} values, expected {count}",
                            data.len()
                        ),
                    });
                }
                Ok(data
                    .iter()
                    .map(|&logit| f64::from(logit))
                    .collect::<Vec<f64>>())
            })
            .map_err(FamilyError::from)
    }
}
