//! MLX/Metal execution backend for the pinned d3 profiles (feature `mlx`,
//! macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, code readout, and wire mapping
//! feed an MLX array forward ([`model`]) that mirrors the candle CPU
//! arithmetic at the profile's frozen noncausal geometry. The Candle CPU
//! path remains the correctness oracle; parity gates compare against the
//! committed golden fixtures.
//!
//! Precision is per profile: `d3-edge`, `d3-lite`, `d3-nano`, and
//! `d3-mini` execute FP32 (BF16 weights widen exactly on load);
//! `d3-flash` — whose FP32 weights exceed the reference host's wired
//! limit — quantizes every layer projection to affine 4-bit (group 64) at
//! load.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::sync::Arc;

use openkind_core::{ModelInfo, SystemRequest, SystemResponse};

use crate::families::d3::engine::{load_renderer, plan_owned_request, score_planned_question};
use crate::families::d3::renderer::D3Renderer;
use crate::families::support::{BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator};
use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

mod model;

use model::D3MlxModel;

/// Arithmetic/device identity of the FP32 d3 MLX execution path.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-qwen35-text-noncausal";

/// Arithmetic/device identity of the `d3-flash` affine-4 MLX execution path.
pub const MLX_AFFINE4_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-affine4-qwen35-text-noncausal";

/// Loaded pinned engine for one profile of the family on MLX.
pub struct D3MlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    profile: &'static super::D3Profile,
    renderer: D3Renderer,
    model: D3MlxModel,
    backend_id: String,
}

impl D3MlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    ///
    /// `d3-flash` quantizes at load; every other profile executes FP32.
    pub fn load(config: super::D3EngineConfig) -> Result<BoundedFamilyEngine, super::D3Error> {
        let profile = config.profile;
        let artifacts =
            crate::families::d3::model::VerifiedArtifacts::verify(&config.model_root, profile)?;
        let renderer = load_renderer(&config.model_root, &artifacts, profile)?;
        let runtime =
            Arc::new(MlxRuntime::new(MlxRuntimeConfig::default()).map_err(FamilyError::from)?);
        let quantize = profile.loader_id == "d3-flash";
        let model = runtime
            .execute(|| D3MlxModel::load(&artifacts, profile, &runtime, quantize))
            .map_err(FamilyError::from)??;
        let engine = Self {
            inner: Arc::new(Inner {
                profile,
                renderer,
                model,
                backend_id: mlx_backend_id(profile),
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

/// Backend id of one profile on this backend: `d3-<short>/mlx-fp32`, or
/// `d3-flash/mlx-affine4` for the at-load quantized profile.
fn mlx_backend_id(profile: &'static super::D3Profile) -> String {
    let short = profile
        .loader_id
        .strip_prefix("d3-")
        .unwrap_or(profile.loader_id);
    if short == "flash" {
        "d3-flash/mlx-affine4".to_owned()
    } else {
        format!("d3-{short}/mlx-fp32")
    }
}

impl FamilyEvaluator for D3MlxEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} noncausal code-readout decision engine ({}, {}).",
                self.inner.profile.backbone_id,
                self.inner.profile.profile_id,
                self.inner.backend_id
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
        control.check()?;
        // Every question is validated, rendered, and budget-checked before
        // the first forward, exactly as the CPU engine plans them.
        let planned = plan_owned_request(
            self.inner.profile,
            &self.inner.renderer,
            crate::families::wire::state_value(request.state),
            crate::families::wire::into_sorted_questions(request.questions)?,
        )?;
        let mut answers: std::collections::HashMap<
            String,
            openkind_core::Answer,
            openkind_core::WireHashState,
        > = std::collections::HashMap::with_capacity_and_hasher(
            planned.questions.len(),
            Default::default(),
        );
        for (id, question) in planned.questions {
            control.check()?;
            let model = &self.inner.model;
            let answer =
                score_planned_question(self.inner.profile, question, control, |ids, count| {
                    model.code_logits(ids, count, control)
                })?;
            answers.insert(id, answer);
        }
        control.check()?;
        Ok(crate::families::wire::system_response(
            request.model,
            answers,
            planned.input_tokens,
        ))
    }
}
