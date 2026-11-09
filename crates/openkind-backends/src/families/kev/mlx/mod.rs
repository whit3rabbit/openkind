//! MLX/Metal execution backend for the pinned `kev` profile (feature
//! `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, calibration temperature, and
//! wire mapping feed the shared dense-Qwen MLX backbone at the pinned
//! Qwen3-0.6B-Base geometry in FP32 with the verified PEFT adapter merged
//! host-side exactly like the CPU oracle's merge (`model.rs`). The Candle
//! CPU path remains the correctness oracle; parity gates compare against
//! the committed golden fixtures.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::path::PathBuf;
use std::sync::Arc;

use openkind_core::{ModelInfo, SystemRequest, SystemResponse};

use crate::families::support::{
    BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator, FamilyLimits,
};
use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

use super::engine::{evaluate_with, KevPassSource};
use super::model::VerifiedArtifacts;
use super::renderer::KevRenderer;
use super::{KevError, BACKBONE_ID, BASE_MODEL_ID, MLX_BACKEND_ID};

pub(super) mod model;

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// base checkpoint widened to FP32 arrays with the pinned adapter merged at
/// load, computed on the Metal GPU through the shared dense Qwen MLX
/// backbone.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-kev";

/// Loaded pinned kev engine on the MLX backend.
pub struct KevMlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: KevRenderer,
    model: model::KevMlxModel,
}

impl KevPassSource for Inner {
    fn renderer(&self) -> &KevRenderer {
        &self.renderer
    }

    fn rows_logits(
        &self,
        rows: &[u32],
        row_lens: &[usize],
        readouts: &[(usize, Vec<usize>)],
        control: &FamilyControl,
    ) -> Result<Vec<Vec<f64>>, FamilyError> {
        control.check()?;
        // No outer `runtime.execute` here: the model's forward takes the
        // process-wide execution lock itself, and the lock is not reentrant.
        let logits = self.model.rows_logits(rows, row_lens, readouts)?;
        control.check()?;
        Ok(logits)
    }
}

/// Filesystem configuration for the pinned profile on the MLX backend.
#[derive(Debug, Clone)]
pub struct KevMlxEngineConfig {
    /// Model root containing the pinned checkpoint artifacts
    /// (`adapter_model.safetensors`, `tokenizer.json`, and
    /// `head.safetensors`). Verified in place; nothing is copied or
    /// downloaded.
    pub model_root: PathBuf,
    /// Base-model root containing the pinned `Qwen3-0.6B-Base`
    /// `model.safetensors` and `config.json`.
    pub base_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl KevMlxEngineConfig {
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
        }
    }
}

impl KevMlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(config: KevMlxEngineConfig) -> Result<BoundedFamilyEngine, KevError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root, &config.base_root)?;
        let renderer = KevRenderer::load(&artifacts.tokenizer)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        let model = model::KevMlxModel::load(&artifacts, &runtime)?;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for KevMlxEngine {
    fn backend_id(&self) -> &str {
        MLX_BACKEND_ID
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} pointer decision engine on {BASE_MODEL_ID} \
                 (MLX FP32)."
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
        evaluate_with(self.inner.as_ref(), request, control)
    }
}
