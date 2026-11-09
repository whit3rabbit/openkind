//! MLX/Metal execution backend for the pinned decoder-logit-llm (GGUF)
//! profile (feature `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified GGUF artifact, renderer, calibration temperature,
//! and wire mapping feed the dequantized FP32 MLX dense forward
//! (`model.rs`). The candle CPU q8_0 path remains the correctness oracle;
//! the MLX backend executes the checkpoint's dequantized FP32 weights (a
//! declared arithmetic-identity difference its parity gate bounds at the
//! workspace 0.005 probability budget), and the backend id names that
//! difference. Parity gates compare against the committed golden fixtures.
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

use self::model::DecoderLlmMlxModel;
use super::model::VerifiedArtifacts;

pub(super) mod model;

/// Arithmetic/device identity of the family MLX execution path.
pub use self::model::MLX_EXECUTION_ARITHMETIC_ID;

/// Backend id reported by the MLX engine serving this profile.
pub const MLX_BACKEND_ID: &str = "decoder-logit-llm/mlx-fp32";

/// Loaded pinned decoder-logit-llm engine on the MLX backend.
pub struct DecoderLlmMlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: LetterRenderer,
    model: DecoderLlmMlxModel,
}

/// Filesystem configuration for the pinned decoder-logit-llm profile on the
/// MLX backend.
#[derive(Debug, Clone)]
pub struct DecoderLlmMlxEngineConfig {
    /// Model root containing the pinned `qwen2.5-0.5b-instruct-q8_0.gguf`
    /// and `tokenizer.json`. Artifacts are verified in place; nothing is
    /// copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DecoderLlmMlxEngineConfig {
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

impl DecoderLlmMlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(
        config: DecoderLlmMlxEngineConfig,
    ) -> Result<BoundedFamilyEngine, super::DecoderLlmError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = LetterRenderer::load(&artifacts.tokenizer, super::MAX_SEQUENCE_TOKENS)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        // The GGUF dequantization and every array construction run under
        // the process-wide MLX execution lock; the lock is not reentrant,
        // so this load must not already hold it. Forwards take the lock
        // themselves in `model.rs`.
        let model = runtime.execute(|| DecoderLlmMlxModel::load(&artifacts, &runtime))??;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for DecoderLlmMlxEngine {
    fn backend_id(&self) -> &str {
        MLX_BACKEND_ID
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} ({}, GGUF q8_0 dequantized to FP32) letter-logit decision engine \
                 ({}, MLX).",
                super::BACKBONE_ID,
                super::GGUF_VARIANT,
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
