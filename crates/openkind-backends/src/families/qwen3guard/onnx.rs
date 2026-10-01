//! ONNX execution adapter for the pinned qwen3guard profile (feature
//! `onnx`).
//!
//! The adapter loads `model.onnx` from the profile root — exported from the
//! same pinned `Qwen/Qwen3Guard-Stream-0.6B` checkpoint the candle loader
//! verifies, with the query-side Stream head (pre → RMSNorm → risk head)
//! folded into the graph — and reproduces the family readout through ONNX
//! Runtime instead of candle:
//!
//! - inputs: `input_ids` (`int64`, `[1, L]`) and `attention_mask` (`int64`,
//!   `[1, L]`);
//! - output: `logits` (`float32`, `[1, 3]`) in the checkpoint class order
//!   `[Safe, Unsafe, Controversial]` at the final prompt position.
//!
//! Unpadded sequences pass an all-one mask, mirroring the candle path. See
//! `docs/ONNX.md` for the export contract and promotion gates.

use std::path::{Path, PathBuf};

use crate::families::support::FamilyError;
use crate::onnx::{OnnxAcceleration, OnnxError, OnnxModel, OnnxRuntimeSettings, OnnxTensorSpec};

use super::{CLASS_CONTROVERSIAL, CLASS_SAFE, CLASS_UNSAFE};

/// Artifact file name inside the profile root.
const MODEL_FILE: &str = "model.onnx";
/// Required token-id input name.
const INPUT_IDS: &str = "input_ids";
/// Required attention-mask input name.
const ATTENTION_MASK: &str = "attention_mask";
/// Required logits output name.
const LOGITS: &str = "logits";
/// Number of risk-level classes the artifact must emit.
const CLASS_COUNT: usize = 3;

/// ONNX-backed qwen3guard model executing the pinned query-side risk readout
/// through ONNX Runtime.
pub struct Qwen3GuardOnnxModel {
    model: OnnxModel,
}

impl Qwen3GuardOnnxModel {
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

    /// Forward the prompt once and return the query-side risk-level logits
    /// `[Safe, Unsafe, Controversial]` at the final position, matching the
    /// candle model.
    ///
    /// # Errors
    /// Returns [`FamilyError`] for session failures and readout-shape
    /// mismatches.
    pub fn risk_logits(&self, prompt_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        let length = prompt_ids.len();
        if length == 0 {
            return Err(FamilyError::InvalidInput(
                "prompt token ids are empty".to_owned(),
            ));
        }
        let input_ids: Vec<i64> = prompt_ids.iter().map(|&id| i64::from(id)).collect();
        // Unpadded sequence: every position is attended, mirroring the
        // candle path's causal-forward with nothing masked.
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
        let logits: [f32; CLASS_COUNT] = self.model.run(inputs, |outputs| {
            let (shape, data) = outputs[LOGITS].try_extract_tensor::<f32>()?;
            if shape.len() != 2
                || shape[0] != 1
                || shape[1] != i64::try_from(CLASS_COUNT).unwrap_or(i64::MAX)
            {
                return Err(OnnxError::Signature {
                    artifact: PathBuf::new(),
                    message: format!("`{LOGITS}` must be [1, {CLASS_COUNT}], got {shape:?}"),
                });
            }
            Ok([data[0], data[1], data[2]])
        })?;
        Ok([
            f64::from(logits[CLASS_SAFE]),
            f64::from(logits[CLASS_UNSAFE]),
            f64::from(logits[CLASS_CONTROVERSIAL]),
        ])
    }
}
