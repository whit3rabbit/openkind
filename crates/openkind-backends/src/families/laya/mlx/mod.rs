//! MLX/Metal execution backend for the pinned laya profiles (feature `mlx`,
//! macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, temperature tables, and wire
//! mapping feed an MLX array forward (`model.rs`) that mirrors the candle
//! CPU arithmetic in FP32. The Candle CPU path remains the correctness
//! oracle; parity gates compare against the committed golden fixtures.
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
use super::renderer::LayaRenderer;
use super::{LayaError, LayaProfile};

pub(super) mod model;

/// Arithmetic/device identity of the laya MLX execution path: FP16-stored
/// shards upcast to FP32 arrays computed on the Metal GPU.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-laya";

/// Loaded pinned laya engine backed by MLX.
pub struct LayaMlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    profile: &'static LayaProfile,
    renderer: LayaRenderer,
    model: model::LayaMlxModel,
    runtime: Arc<MlxRuntime>,
}

/// Filesystem configuration for one pinned laya profile on the MLX backend.
#[derive(Debug, Clone)]
pub struct LayaMlxEngineConfig {
    /// The pinned profile to load.
    pub profile: &'static LayaProfile,
    /// Model root containing the pinned artifacts. Verified in place;
    /// nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl LayaMlxEngineConfig {
    /// Fill admission defaults around the given model root.
    pub fn new(profile: &'static LayaProfile, model_root: impl Into<PathBuf>) -> Self {
        Self {
            profile,
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

impl LayaMlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    pub fn load(config: LayaMlxEngineConfig) -> Result<BoundedFamilyEngine, LayaError> {
        let profile = config.profile;
        let artifacts = VerifiedArtifacts::verify(&config.model_root, profile)?;
        let renderer = LayaRenderer::load(&artifacts.tokenizer, &profile.specials)?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        let model =
            runtime.execute(|| model::LayaMlxModel::load(profile, &artifacts.checkpoint))??;
        let engine = Self {
            inner: Arc::new(Inner {
                profile,
                renderer,
                model,
                runtime,
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for LayaMlxEngine {
    fn backend_id(&self) -> &str {
        match self.inner.profile.loader_id {
            "laya-english" => "laya-english/mlx-fp32",
            "laya-multilingual" => "laya-multilingual/mlx-fp32",
            _ => "laya-typed-decisions/mlx-fp32",
        }
    }

    fn model_metadata(&self) -> ModelInfo {
        let profile = self.inner.profile;
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} decision encoder ({}, MLX FP32).",
                profile.backbone_id, profile.profile_id
            ),
            release_date: "2026-09-24".into(),
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
        evaluate_with_logits(
            self.inner.profile,
            &self.inner.renderer,
            &|token_ids: &[u32], markers: &[usize], qtype: usize| {
                // Every MLX forward runs on the serialized GPU stream.
                Ok(self
                    .inner
                    .runtime
                    .execute(|| self.inner.model.option_logits(token_ids, markers, qtype))??)
            },
            request,
            control,
        )
    }
}
