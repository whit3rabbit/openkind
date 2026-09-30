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
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
    FamilyLimits,
};
use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

use super::engine::{laya_state_text, render_question_inputs, resolve_temperature};
use super::model::{mask_token_of, VerifiedArtifacts};
use super::renderer::{build_sequence, LayaRenderer, SequencePlan};
use super::{LayaError, LayaProfile, MAX_CANDIDATES};

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

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        control.check()?;
        let profile = self.inner.profile;
        let state_serialized = laya_state_text(&request.state)?;
        if state_serialized.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "state must contain non-whitespace text".to_owned(),
            ));
        }
        let truncate_left = matches!(request.state, openkind_core::State::Array(_));
        let state_text = state_serialized.replace(mask_token_of(&profile.specials), " ");
        let state_ids = self.inner.renderer.encode_plain(&state_text)?;

        let mut question_ids: Vec<_> = request.questions.keys().cloned().collect();
        question_ids.sort();
        if question_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "request contains no questions".to_owned(),
            ));
        }

        let mut answers = std::collections::HashMap::<
            String,
            openkind_core::Answer,
            openkind_core::WireHashState,
        >::with_capacity_and_hasher(
            question_ids.len(), Default::default()
        );
        let mut input_tokens: u64 = 0;
        for id in &question_ids {
            control.check()?;
            let question = request
                .questions
                .get(id)
                .expect("question id came from the request map");
            let inputs = render_question_inputs(question)?;
            if inputs.options.len() > MAX_CANDIDATES {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` offers {} options but the profile accepts at most {MAX_CANDIDATES}",
                    inputs.options.len()
                )));
            }
            let plan = SequencePlan {
                renderer: &self.inner.renderer,
                specials: &profile.specials,
                max_len: profile.max_sequence_tokens,
                head_max_len: profile.head_max_len,
                truncate_left,
            };
            let rendered = build_sequence(
                &plan,
                inputs.type_name,
                &crate::families::wire::instruction_text(question)?,
                &inputs.options,
                &state_ids,
            )?;
            input_tokens = input_tokens.saturating_add(rendered.ids.len() as u64);
            let logits = self.inner.runtime.execute(|| {
                self.inner
                    .model
                    .option_logits(&rendered.ids, &rendered.markers, inputs.qtype)
            })??;
            let temperature = resolve_temperature(
                profile,
                inputs.type_name,
                inputs.qtype,
                inputs.options.len(),
            );
            let probabilities = temperature_softmax(&logits, temperature)?;
            let unpacked = crate::families::wire::unpack_question(id, question)?;
            let answer =
                crate::families::wire::answer_from_probabilities(&unpacked, &probabilities)?;
            answers.insert(id.clone(), answer);
        }
        control.check()?;
        Ok(SystemResponse {
            model: request.model,
            answers,
            usage: openkind_core::Usage {
                input_tokens: u32::try_from(input_tokens).unwrap_or(u32::MAX),
                output_tokens: 0,
            },
        })
    }
}
