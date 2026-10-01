//! ONNX execution adapter for the pinned laya profiles (feature `onnx`).
//!
//! The adapter loads `model.onnx` from the profile model root — exported
//! from the same digest-verified checkpoint the candle loader verifies —
//! and reproduces the family readout through ONNX Runtime instead of
//! candle. The artifact must contain the full decision path: the
//! ModernBERT encoder body, the type-embedding add, the two pre-norm head
//! encoder layers, the gather at the option-marker positions, and the
//! marker scorer (`LayerNorm → Linear → GELU → Linear(1)`), so the adapter
//! feeds token ids and reads the final per-option logits:
//!
//! - inputs: `input_ids` (`int64`, `[1, L]`), `attention_mask` (`int64`,
//!   `[1, L]`), `question_type` (`int64`, `[1]`, the type-embedding row for
//!   the question type — choice `0`, score `1`, noul `2`), and
//!   `marker_positions` (`int64`, `[K]`, the marker offsets in token
//!   order). The marker positions and question type are runtime inputs
//!   because the candle readout selects them per question; exporting them
//!   as inputs keeps one artifact per profile for all question types and
//!   option counts.
//! - output: `logits` (`float32`, `[1, K]`), one logit per marker in
//!   marker order, matching what the candle `option_logits` returns before
//!   temperature calibration.
//!
//! Unpadded sequences pass an all-one attention mask, mirroring the candle
//! path's all-attended tensor. The engine applies the pinned temperature
//! tables after this readout, exactly as on the candle path. See
//! `docs/ONNX.md` for the export contract and promotion gates.

use std::path::{Path, PathBuf};

use crate::families::support::{onnx_marker_positions, FamilyError};
use crate::onnx::{OnnxAcceleration, OnnxError, OnnxModel, OnnxRuntimeSettings, OnnxTensorSpec};

/// Artifact file name inside the profile model root.
const MODEL_FILE: &str = "model.onnx";
/// Required token-id input name.
const INPUT_IDS: &str = "input_ids";
/// Required attention-mask input name.
const ATTENTION_MASK: &str = "attention_mask";
/// Required question-type input name (type-embedding row selector).
const QUESTION_TYPE: &str = "question_type";
/// Required marker-position input name.
const MARKER_POSITIONS: &str = "marker_positions";
/// Required logits output name.
const LOGITS: &str = "logits";

/// ONNX-backed laya model executing the pinned typed-decision readout
/// through ONNX Runtime.
pub struct LayaOnnxModel {
    model: OnnxModel,
}

impl LayaOnnxModel {
    /// Load `model.onnx` from the profile model root with the selected
    /// execution provider.
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
                OnnxTensorSpec::i64(QUESTION_TYPE, &[1]),
                OnnxTensorSpec::i64(MARKER_POSITIONS, &[-1]),
            ],
            &[OnnxTensorSpec::f32(LOGITS, &[1, -1])],
        )?;
        Ok(Self { model })
    }

    /// Forward one rendered sequence and return one logit per marker in
    /// marker order, matching the candle model's `option_logits` (before
    /// the engine's temperature calibration).
    ///
    /// # Errors
    /// Returns [`FamilyError`] for empty inputs, session failures, and
    /// readout-shape mismatches.
    pub fn option_logits(
        &self,
        token_ids: &[u32],
        markers: &[usize],
        qtype: usize,
    ) -> Result<Vec<f64>, FamilyError> {
        if token_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "encoded sequence is empty".to_owned(),
            ));
        }
        if markers.is_empty() {
            return Err(FamilyError::InvalidInput(
                "rendered sequence carries no option markers".to_owned(),
            ));
        }
        let length = token_ids.len();
        if qtype > 2 {
            return Err(FamilyError::InvalidInput(format!(
                "question type {qtype} must be choice 0, score 1, or noul 2"
            )));
        }
        let input_ids: Vec<i64> = token_ids.iter().map(|&id| i64::from(id)).collect();
        // Unpadded sequence: every position is attended, mirroring the
        // candle path.
        let attention_mask = vec![1_i64; length];
        let question_type = vec![qtype as i64];
        let marker_positions = onnx_marker_positions(markers, length)?;
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
                QUESTION_TYPE.to_owned(),
                ort::value::Tensor::from_array((vec![1_usize], question_type))
                    .map_err(OnnxError::from)?
                    .into_dyn(),
            ),
            (
                MARKER_POSITIONS.to_owned(),
                ort::value::Tensor::from_array((vec![markers.len()], marker_positions))
                    .map_err(OnnxError::from)?
                    .into_dyn(),
            ),
        ];
        let count = markers.len();
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
