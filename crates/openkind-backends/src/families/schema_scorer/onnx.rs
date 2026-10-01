//! ONNX execution adapter for the pinned schema-scorer profile (feature
//! `onnx`).
//!
//! The adapter loads `model.onnx` from the profile root — exported from the
//! same pinned `cross-encoder/ms-marco-MiniLM-L-6-v2` checkpoint the candle
//! loader verifies, with the pooler and scalar classifier folded into the
//! graph — and reproduces the family readout through ONNX Runtime instead of
//! candle:
//!
//! - inputs: `input_ids` (`int64`, `[1, L]`), `attention_mask` (`int64`,
//!   `[1, L]`), and `token_type_ids` (`int64`, `[1, L]`);
//! - output: `logits` (`float32`, `[1, 1]`) — the scalar relevance logit of
//!   one query–passage pair.
//!
//! Unpadded sequences pass an all-one attention mask, and every position is
//! sequence A (`token_type_ids` all zero), mirroring the candle path. See
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
/// Required token-type-id input name.
const TOKEN_TYPE_IDS: &str = "token_type_ids";
/// Required logits output name.
const LOGITS: &str = "logits";

/// ONNX-backed schema-scorer model executing the pinned relevance readout
/// through ONNX Runtime.
pub struct SchemaScorerOnnxModel {
    model: OnnxModel,
}

impl SchemaScorerOnnxModel {
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
                OnnxTensorSpec::i64(TOKEN_TYPE_IDS, &[1, -1]),
            ],
            &[OnnxTensorSpec::f32(LOGITS, &[1, 1])],
        )?;
        Ok(Self { model })
    }

    /// Forward one query–passage pair and return the scalar relevance logit,
    /// matching the candle model.
    ///
    /// # Errors
    /// Returns [`FamilyError`] for session failures and readout-shape
    /// mismatches.
    pub fn relevance_logit(&self, pair_ids: &[u32]) -> Result<f64, FamilyError> {
        let length = pair_ids.len();
        if length == 0 {
            return Err(FamilyError::InvalidInput(
                "encoded pair is empty".to_owned(),
            ));
        }
        let input_ids: Vec<i64> = pair_ids.iter().map(|&id| i64::from(id)).collect();
        // Unpadded sequence: every position is attended, mirroring the
        // candle path's absent (nothing-masked) mask tensor.
        let attention_mask = vec![1_i64; length];
        // Sequence A for every position, mirroring the candle path's
        // all-zero token-type tensor.
        let token_type_ids = vec![0_i64; length];
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
                TOKEN_TYPE_IDS.to_owned(),
                ort::value::Tensor::from_array((vec![1_usize, length], token_type_ids))
                    .map_err(OnnxError::from)?
                    .into_dyn(),
            ),
        ];
        let logit: f32 = self.model.run(inputs, |outputs| {
            let (shape, data) = outputs[LOGITS].try_extract_tensor::<f32>()?;
            if shape.len() != 2 || shape[0] != 1 || shape[1] != 1 {
                return Err(OnnxError::Signature {
                    artifact: PathBuf::new(),
                    message: format!("`{LOGITS}` must be [1, 1], got {shape:?}"),
                });
            }
            Ok(data[0])
        })?;
        Ok(f64::from(logit))
    }
}
