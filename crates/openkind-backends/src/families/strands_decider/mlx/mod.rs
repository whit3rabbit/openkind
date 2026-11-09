//! MLX/Metal execution backend for the pinned `strands-decider` profiles
//! (feature `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, calibration temperatures, and
//! wire mapping feed the geometry-parameterized causal MLX survey model at
//! the Qwen3.5-2B geometry in FP32 with the pinned adapter merged at load
//! (`model.rs`). The Candle CPU path remains the correctness oracle; parity
//! gates compare against the committed golden fixtures.
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

use super::engine::{evaluate_with, StrandsPassSource};
use super::model::VerifiedArtifacts;
use super::renderer::StrandsRenderer;
use super::{StrandsDeciderError, StrandsDeciderProfile, BASE_MODEL_ID};

pub(super) mod model;

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// base checkpoint widened to FP32 arrays with the pinned adapter merged at
/// load, computed on the Metal GPU through the geometry-parameterized causal
/// survey model.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-strands";

/// Loaded pinned profile engine on the MLX backend.
pub struct StrandsMlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: StrandsRenderer,
    model: model::StrandsMlxModel,
    profile: &'static StrandsDeciderProfile,
}

impl StrandsPassSource for Inner {
    fn profile(&self) -> &'static StrandsDeciderProfile {
        self.profile
    }

    fn renderer(&self) -> &StrandsRenderer {
        &self.renderer
    }

    fn rows_logits(
        &self,
        row: &super::renderer::RenderedRow,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        control.check()?;
        // No outer `runtime.execute` here: the model's forward takes the
        // process-wide execution lock itself, and the lock is not reentrant.
        let logits = self.model.rows_logits(row)?;
        control.check()?;
        Ok(logits)
    }
}

/// Filesystem configuration for one pinned profile on the MLX backend.
#[derive(Debug, Clone)]
pub struct StrandsMlxEngineConfig {
    /// Model root containing the pinned checkpoint artifacts. Verified in
    /// place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Base-model root containing the pinned `Qwen3.5-2B-Base` checkpoint.
    pub base_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
    /// Explicit profile to load. When `None`, auto-detected from artifacts.
    pub profile: Option<&'static StrandsDeciderProfile>,
}

impl StrandsMlxEngineConfig {
    /// Fill admission defaults around the given roots.
    pub fn new(model_root: impl Into<PathBuf>, base_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            base_root: base_root.into(),
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 2,
                retry_after_ms: 1_000,
                evaluation_timeout: Some(std::time::Duration::from_secs(600)),
            },
            profile: None,
        }
    }
}

impl StrandsMlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(
        config: StrandsMlxEngineConfig,
    ) -> Result<BoundedFamilyEngine, StrandsDeciderError> {
        let artifacts = VerifiedArtifacts::verify_with_profile(
            &config.model_root,
            &config.base_root,
            config.profile,
        )?;
        let profile = artifacts.profile;
        let renderer = StrandsRenderer::load(&artifacts.tokenizer)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        let model = model::StrandsMlxModel::load(&artifacts, &runtime)?;
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                profile,
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for StrandsMlxEngine {
    fn backend_id(&self) -> &str {
        self.inner.profile.mlx_backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} pointer decision engine on {} (MLX FP32).",
                self.inner.profile.backbone_id, BASE_MODEL_ID
            ),
            release_date: self.inner.profile.release_date.into(),
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
        evaluate_with(self.inner.as_ref(), request, control)
    }
}
