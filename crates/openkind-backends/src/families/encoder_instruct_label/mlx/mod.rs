//! MLX/Metal execution backend for the pinned `encoder-instruct-label`
//! profile (feature `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, calibration temperature, and
//! wire mapping feed an MLX array forward (`model.rs`) that mirrors the
//! candle CPU arithmetic in FP32. The Candle CPU path remains the
//! correctness oracle; parity gates compare against the committed golden
//! fixtures.
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

use super::engine::evaluate_with_logits;
use super::model::VerifiedArtifacts;
use super::renderer::EncoderInstructLabelRenderer;
use super::{EncoderInstructLabelError, BACKBONE_ID, PROFILE_ID};

pub(super) mod model;

/// Arithmetic/device identity of the GLiClass MLX execution path: FP32
/// arrays loaded from the FP32-stored pinned shard and computed on the Metal
/// GPU.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-gliclass-modern-base";

/// Loaded pinned label-marker engine backed by MLX.
pub struct EncoderInstructLabelMlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: EncoderInstructLabelRenderer,
    model: model::GliclassMlxModel,
    runtime: Arc<MlxRuntime>,
}

/// Filesystem configuration for the pinned label-marker profile on the MLX
/// backend.
#[derive(Debug, Clone)]
pub struct EncoderInstructLabelMlxEngineConfig {
    /// Model root containing the pinned artifacts. Verified in place;
    /// nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl EncoderInstructLabelMlxEngineConfig {
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

impl EncoderInstructLabelMlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(
        config: EncoderInstructLabelMlxEngineConfig,
    ) -> Result<BoundedFamilyEngine, EncoderInstructLabelError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = EncoderInstructLabelRenderer::load(&artifacts.tokenizer)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        let model = runtime.execute(|| model::GliclassMlxModel::load(&artifacts.checkpoint))??;
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                runtime,
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for EncoderInstructLabelMlxEngine {
    fn backend_id(&self) -> &str {
        "encoder-instruct-label/mlx-fp32"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} label-marker decision engine ({PROFILE_ID}, MLX FP32)."
            ),
            release_date: "2026-09-26".into(),
        }
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        evaluate_with_logits(
            &|markers: &[String], state: &str| {
                let ids = self.inner.renderer.render(markers, state)?;
                let count = ids.len();
                let positions = model::marker_positions(&ids);
                if positions.len() != markers.len() {
                    return Err(FamilyError::ContractMismatch {
                        field: "marker.count",
                        expected: format!("{} marker positions", markers.len()),
                        actual: format!("{} positions", positions.len()),
                    });
                }
                let logits = self
                    .inner
                    .runtime
                    .execute(|| self.inner.model.marker_logits(&ids, &positions))??;
                Ok((logits, count))
            },
            request,
            control,
        )
    }
}
