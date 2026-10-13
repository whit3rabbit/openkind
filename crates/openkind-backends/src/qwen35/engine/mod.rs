//! Direct Jev wire adapter for the pinned native Qwen3.5 profile.

mod backbone;
mod canonical;
mod eval;
mod mapping;

pub(super) use canonical::state_text;
pub(super) use mapping::{instruction_text, question_candidates};

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};
use openkind_engine::{DecisionEngine, EngineResult, ProbabilitySpace};

pub use self::backbone::Qwen35Backend;
use self::backbone::{load_backbone, EngineBackbone};
use self::eval::{evaluate_request, map_evaluation_error};
use super::{
    ExecutionControl, Qwen35Error, Qwen35Tokenizer, ReferenceBundle, SchedulerConfig,
    ScoreSummaryHead,
};
use crate::families::support::{
    outcome_label, saturating_millis, AdmissionGate, NATIVE_GATE_METRICS,
};

/// Reserved Choice criteria key that explicitly exposes the model's semantic-none mass.
///
/// Native Choice requests must include this key with a non-empty description. The adapter
/// never appends an unrequested option and never discards or renormalizes none probability.
pub const SEMANTIC_NONE_OPTION: &str = "__none__";

/// Filesystem and admission settings for the direct native engine.
#[derive(Debug, Clone)]
pub struct Qwen35EngineConfig {
    /// Selected-profile bundle directory containing the fitted head.
    pub bundle_root: PathBuf,
    /// Pinned Qwen checkpoint directory containing both safetensors shards.
    pub checkpoint_root: PathBuf,
    /// Digest-locked exported tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Execution backend executing tokens for this engine.
    pub backend: Qwen35Backend,
    /// Adaptive execution policy and memory admission settings.
    pub scheduler: SchedulerConfig,
    /// Maximum model evaluations executing concurrently.
    pub max_concurrent_requests: usize,
    /// Maximum additional requests allowed to wait for execution.
    pub max_queued_requests: usize,
    /// Suggested backoff returned when admission is full.
    pub retry_after_ms: u64,
    /// Queue-inclusive request deadline. `None` leaves model evaluation unbounded.
    pub evaluation_timeout: Option<Duration>,
}

pub(super) struct EngineInner {
    pub(super) tokenizer: Qwen35Tokenizer,
    pub(super) backbone: EngineBackbone,
    pub(super) head: ScoreSummaryHead,
    pub(super) scheduler: SchedulerConfig,
    pub(super) max_concurrent_requests: usize,
    pub(super) backend_id: &'static str,
    pub(super) evaluation_timeout: Option<Duration>,
}

/// Native CPU implementation registered directly behind [`DecisionEngine`].
pub struct Qwen35DecisionEngine {
    inner: Arc<EngineInner>,
    gate: AdmissionGate,
}

impl Qwen35DecisionEngine {
    /// Load every pinned artifact offline and construct the bounded native engine.
    pub fn load(config: Qwen35EngineConfig) -> Result<Self, Qwen35Error> {
        if config
            .evaluation_timeout
            .is_some_and(|timeout| Instant::now().checked_add(timeout).is_none())
        {
            return Err(Qwen35Error::InvalidInput(
                "native evaluation timeout is outside the monotonic clock range".into(),
            ));
        }
        let tokenizer = Qwen35Tokenizer::from_file(&config.tokenizer_path)?;
        let mut scheduler = config.scheduler;
        let backbone = load_backbone(config.backend, &config.checkpoint_root, &mut scheduler)?;
        let bundle = ReferenceBundle::load(&config.bundle_root)?;
        // The adapter branches on the declared probability space: this native
        // adapter implements exactly one, and a profile declaring any other
        // space fails explicitly at load instead of silently discarding
        // semantic-none mass or renormalizing the remainder.
        match bundle.profile().execution().probability_space() {
            ProbabilitySpace::OfferedOptionsPlusSemanticNone => {}
            declared => {
                return Err(Qwen35Error::ContractMismatch {
                    field: "execution.probability_space",
                    expected: ProbabilitySpace::OfferedOptionsPlusSemanticNone
                        .as_str()
                        .to_owned(),
                    actual: declared.as_str().to_owned(),
                });
            }
        }
        let concurrent = config.max_concurrent_requests.max(1);
        Ok(Self {
            inner: Arc::new(EngineInner {
                tokenizer,
                backbone,
                head: bundle.head().clone(),
                scheduler,
                max_concurrent_requests: concurrent,
                backend_id: config.backend.as_str(),
                evaluation_timeout: config.evaluation_timeout,
            }),
            gate: AdmissionGate::new(
                concurrent,
                config.max_queued_requests,
                config.retry_after_ms,
                NATIVE_GATE_METRICS,
            ),
        })
    }
}

#[async_trait]
impl DecisionEngine for Qwen35DecisionEngine {
    fn backend_id(&self) -> &str {
        self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        let flavor = if self.inner.backend_id == Qwen35Backend::NativeCpu.as_str() {
            "native CPU reference"
        } else {
            "MLX reference-ops"
        };
        ModelInfo {
            name: String::new(),
            description: format!("Pinned Qwen3.5-4B {flavor} engine with explicit semantic none."),
            release_date: "2026-09-20".into(),
        }
    }

    // `evaluate_request` bills the encoded state root plus every question
    // branch, and the state root is never empty, so the count is nonzero.
    fn reports_input_tokens(&self) -> bool {
        true
    }

    async fn evaluate(&self, request: SystemRequest) -> EngineResult<SystemResponse> {
        let started = Instant::now();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut cancellation = self.gate.cancellation_guard(cancelled.clone());
        let result = self.evaluate_controlled(request, started, cancelled).await;
        cancellation.disarm();
        result
    }
}

impl Qwen35DecisionEngine {
    async fn evaluate_controlled(
        &self,
        request: SystemRequest,
        started: Instant,
        cancelled: Arc<AtomicBool>,
    ) -> EngineResult<SystemResponse> {
        let timeout_ms = self.inner.evaluation_timeout.map(saturating_millis);
        let deadline = self
            .inner
            .evaluation_timeout
            .map(|timeout| started.checked_add(timeout).unwrap_or(started));
        let inner = self.inner.clone();
        let backend = self.backend_id().to_owned();
        let control = ExecutionControl::new(cancelled, deadline);
        let evaluated = self
            .gate
            .run(
                &backend,
                started,
                deadline,
                timeout_ms.unwrap_or_default(),
                move || evaluate_request(&inner, request, &control),
            )
            .await?;
        let result = evaluated.map_err(|error| map_evaluation_error(&backend, error, timeout_ms));
        self.gate.record(started, outcome_label(&result));
        result
    }
}
