//! MLX/Metal execution backend for the pinned decoder-letter profile
//! (feature `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, calibration temperature, and
//! wire mapping feed the shared dense Qwen MLX body in FP32 (`model.rs`).
//! The candle CPU path remains the correctness oracle; parity gates compare
//! against the committed golden fixtures.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::path::PathBuf;
use std::sync::Arc;

use openkind_core::{ModelInfo, SystemRequest, SystemResponse};

use crate::families::letter_renderer::{evaluate_letter_request, LetterRenderer};
use crate::families::support::{
    BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator, FamilyLimits,
};
use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

use self::model::DecoderLetterMlxModel;
use super::model::VerifiedArtifacts;

pub(super) mod model;

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// single-file BF16 checkpoint widened to FP32 arrays computed on the Metal
/// GPU through the shared dense Qwen MLX body.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-decoder-logit-letter";

/// Backend id reported by the MLX engine serving this profile.
pub const MLX_BACKEND_ID: &str = "decoder-logit-letter/mlx-fp32";

/// Loaded pinned decoder-letter engine on the MLX backend.
pub struct DecoderLetterMlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: LetterRenderer,
    model: DecoderLetterMlxModel,
}

/// Filesystem configuration for the pinned decoder-letter profile on the
/// MLX backend.
#[derive(Debug, Clone)]
pub struct DecoderLetterMlxEngineConfig {
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// and `tokenizer.json`. Artifacts are verified in place; nothing is
    /// copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DecoderLetterMlxEngineConfig {
    /// Fill admission defaults around the given model root.
    pub fn new(model_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 2,
                retry_after_ms: 1_000,
                evaluation_timeout: Some(std::time::Duration::from_secs(600)),
            },
        }
    }
}

impl DecoderLetterMlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(
        config: DecoderLetterMlxEngineConfig,
    ) -> Result<BoundedFamilyEngine, super::DecoderLetterError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = LetterRenderer::load(&artifacts.tokenizer, super::MAX_SEQUENCE_TOKENS)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        // No outer `runtime.execute` here: the model load and every forward
        // take the process-wide execution lock themselves, and the lock is
        // not reentrant.
        let model = DecoderLetterMlxModel::load(&artifacts, &runtime)?;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for DecoderLetterMlxEngine {
    fn backend_id(&self) -> &str {
        MLX_BACKEND_ID
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} letter-logit decision engine ({}, MLX FP32).",
                super::BACKBONE_ID,
                super::PROFILE_ID
            ),
            release_date: "2026-09-26".into(),
        }
    }

    fn reports_input_tokens(&self) -> bool {
        true
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        evaluate_letter_request(
            &self.inner.renderer,
            &|prompt_ids: &[u32], letter_ids: &[u32]| {
                // A single MLX forward is uninterruptible, so cancellation
                // and the deadline are observed on either side of it.
                control.check()?;
                let logits = self.inner.model.letter_logits(prompt_ids, letter_ids)?;
                control.check()?;
                Ok(logits)
            },
            super::CALIBRATION_TEMPERATURE,
            request,
            control,
        )
    }
}
