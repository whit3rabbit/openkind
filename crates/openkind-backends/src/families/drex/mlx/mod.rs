//! MLX/Metal execution backend for the pinned `drex-v15` profile (feature
//! `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, calibration temperature, and
//! wire mapping feed the geometry-parameterized causal MLX survey model at
//! the 9B hybrid geometry in FP32 (`model.rs`). The Candle CPU path remains
//! the correctness oracle; parity gates compare against the committed golden
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

use crate::families::kev::renderer::KevRenderer;

use super::engine::{evaluate_with, DrexPassSource};
use super::model::VerifiedArtifacts;
use super::{DrexError, BACKBONE_ID, MAX_ROW_TOKENS, MAX_STATE_TOKENS};

pub(super) mod model;

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// BF16 shards widened to FP32 arrays computed on the Metal GPU through the
/// geometry-parameterized causal survey model.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-drex";

/// MLX backend id reported by the engine serving this profile.
pub const MLX_BACKEND_ID: &str = "drex-v15/mlx-fp32";

/// Loaded pinned profile engine on the MLX backend.
pub struct DrexMlxEngine {
    inner: Inner,
}

struct Inner {
    renderer: KevRenderer,
    model: model::DrexMlxModel,
}

impl DrexPassSource for DrexMlxEngine {
    fn renderer(&self) -> &KevRenderer {
        &self.inner.renderer
    }

    fn row_logits(
        &self,
        row: &crate::families::kev::model::QuestionRow,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        control.check()?;
        // No outer `runtime.execute` here: the model's forward takes the
        // process-wide execution lock itself, and the lock is not reentrant.
        let logits = self.inner.model.row_logits(row)?;
        control.check()?;
        Ok(logits)
    }
}

/// Filesystem configuration for the pinned profile on the MLX backend.
#[derive(Debug, Clone)]
pub struct DrexMlxEngineConfig {
    /// Model root containing the pinned shards, configs, tokenizer, and
    /// `head.pt`. Verified in place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DrexMlxEngineConfig {
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

impl DrexMlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(config: DrexMlxEngineConfig) -> Result<BoundedFamilyEngine, DrexError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer =
            KevRenderer::load_with_limits(&artifacts.tokenizer, MAX_STATE_TOKENS, MAX_ROW_TOKENS)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        let model = model::DrexMlxModel::load(&artifacts, &runtime)?;
        let engine = Self {
            inner: Inner { renderer, model },
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for DrexMlxEngine {
    fn backend_id(&self) -> &str {
        MLX_BACKEND_ID
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} pointer decision engine on a Qwen3.5 9B hybrid \
                 backbone ({MLX_BACKEND_ID}). Nace AI Open RAIL-M weights: inference \
                 only, no distillation or teacher use."
            ),
            release_date: "2026-11-07".into(),
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
        evaluate_with(self, request, control)
    }
}
