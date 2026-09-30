//! MLX/Metal execution backend for the pinned `decoder-logit-qwen35`
//! profile (feature `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, calibration and knockout
//! temperatures, and wire mapping feed the parity-verified MLX Qwen3.5
//! backbone in FP32 (`model.rs`). The Candle CPU path remains the
//! correctness oracle; parity gates compare against the committed golden
//! fixtures.
//!
//! Every MLX operation — model load and per-pass forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::path::PathBuf;
use std::sync::Arc;

use openkind_core::{ModelInfo, SystemRequest, SystemResponse};

use crate::families::support::{
    BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator, FamilyLimits,
};
use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

use super::engine::{evaluate_with, Jevk5PassSource};
use super::model::VerifiedArtifacts;
use super::renderer::{Jevk5Renderer, RenderedPass};
use super::{DecoderLogitQwen35Error, BACKBONE_ID, PROFILE_ID};

pub(super) mod model;

/// Arithmetic/device identity of the JevK5 MLX execution path: the pinned
/// single-file BF16 checkpoint widened to FP32 arrays computed on the Metal
/// GPU through the shared Qwen3.5 MLX backbone.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-jevk5";

/// Loaded pinned JevK5 engine on the MLX backend.
pub struct DecoderLogitQwen35MlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: Jevk5Renderer,
    model: model::Jevk5MlxModel,
}

impl Jevk5PassSource for Inner {
    fn renderer(&self) -> &Jevk5Renderer {
        &self.renderer
    }

    fn letter_logits(
        &self,
        pass: &RenderedPass,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        control.check()?;
        // No outer `runtime.execute` here: `MlxQwen35Backbone::prefill`
        // takes the process-wide execution lock itself, and the lock is not
        // reentrant.
        let logits = self
            .model
            .letter_logits(pass.prompt_ids(), pass.letter_ids())?;
        control.check()?;
        Ok(logits)
    }
}

/// Filesystem configuration for the pinned JevK5 profile on the MLX backend.
#[derive(Debug, Clone)]
pub struct DecoderLogitQwen35MlxEngineConfig {
    /// Model root containing the pinned artifacts. Verified in place;
    /// nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DecoderLogitQwen35MlxEngineConfig {
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

impl DecoderLogitQwen35MlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(
        config: DecoderLogitQwen35MlxEngineConfig,
    ) -> Result<BoundedFamilyEngine, DecoderLogitQwen35Error> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = Jevk5Renderer::load(&artifacts.tokenizer)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        // No outer `runtime.execute`: the weight store and the backbone
        // manage the process-wide execution lock internally, and the lock is
        // not reentrant.
        let model = model::Jevk5MlxModel::load(&artifacts, &runtime)?;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for DecoderLogitQwen35MlxEngine {
    fn backend_id(&self) -> &str {
        "decoder-logit-qwen35/mlx-fp32"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} letter-logit decision engine ({PROFILE_ID}, MLX FP32)."
            ),
            release_date: "2026-09-27".into(),
        }
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        evaluate_with(self.inner.as_ref(), request, control)
    }
}
