//! openpick-engine: runtime-agnostic decision engine trait.
//!
//! Phase 1 ships one implementation: `MockEngine`, which returns
//! deterministic-but-jittered fake answers. Phase 2 will replace it with
//! `candle`, GGUF, and native backends — the trait surface is what the
//! HTTP/gRPC layers talk to, so the server doesn't change when we swap
//! implementations.

pub mod mock;

pub use mock::MockEngine;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use openpick_core::ModelInfo;
use openpick_core::{
    validate_request, Answer, SystemRequest, SystemResponse, ValidationError,
};
use thiserror::Error;
use tracing::instrument;

/// Things that can go wrong inside the engine. Validation failures are
/// surfaced as their own variant so the HTTP layer can map to 422.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid request: {0}")]
    Invalid(#[from] ValidationError),

    #[error("no backend registered for model `{0}`")]
    UnknownModel(String),

    #[error("backend `{backend}` failed: {message}")]
    Backend { backend: String, message: String },
}

pub type EngineResult<T> = Result<T, EngineError>;

/// The trait every backend implements. The engine is **runtime-agnostic**:
/// it doesn't know it's being served over HTTP or gRPC — that's deliberate.
/// Callers pass `SystemRequest` in and get `SystemResponse` back.
#[async_trait]
pub trait DecisionEngine: Send + Sync {
    /// Identifier of this backend, e.g. `"mock"`, `"qwen-3b-candle"`.
    fn backend_id(&self) -> &str;

    /// Public metadata shown by `GET /v1/models`. The default is a
    /// minimal mock entry; real backends should override with a real
    /// description and release date.
    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: self.backend_id().to_string(),
            description: format!("{} backend", self.backend_id()),
            release_date: "1970-01-01".to_string(),
        }
    }

    /// Evaluate a single request. Returns one answer per question id.
    async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse>;

    /// Estimate input tokens (cheap, before evaluation). The mock uses
    /// a rough `chars / 4` heuristic; real backends will use a tokenizer.
    fn estimate_input_tokens(&self, req: &SystemRequest) -> u32 {
        let state_chars = match &req.state {
            openpick_core::State::Text(s) => s.len(),
            openpick_core::State::Object(m) => serde_json::to_string(m)
                .map(|s| s.len())
                .unwrap_or(0),
            openpick_core::State::Array(a) => serde_json::to_string(a)
                .map(|s| s.len())
                .unwrap_or(0),
        };
        let instr_chars: usize = req
            .questions
            .values()
            .map(|q| match q {
                openpick_core::Question::Noul(n) => {
                    serde_json::to_string(&n.instructions).map(|s| s.len()).unwrap_or(0)
                }
                openpick_core::Question::Choice(c) => {
                    serde_json::to_string(&c.instructions).map(|s| s.len()).unwrap_or(0)
                }
                openpick_core::Question::Score(s) => {
                    serde_json::to_string(&s.instructions).map(|s| s.len()).unwrap_or(0)
                }
            })
            .sum();
        ((state_chars + instr_chars) / 4) as u32
    }
}

/// A registry mapping model alias → engine. Lets the server dispatch by
/// the `model` field in the request without the engine itself knowing.
#[derive(Default, Clone)]
pub struct EngineRegistry {
    engines: HashMap<String, Arc<dyn DecisionEngine>>,
}

impl std::fmt::Debug for EngineRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineRegistry")
            .field("models", &self.engines.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, alias: impl Into<String>, engine: Arc<dyn DecisionEngine>) {
        self.engines.insert(alias.into(), engine);
    }

    pub fn get(&self, model: &str) -> Option<Arc<dyn DecisionEngine>> {
        self.engines.get(model).cloned()
    }

    /// Sorted list of registered alias names.
    pub fn models(&self) -> Vec<String> {
        let mut v: Vec<String> = self.engines.keys().cloned().collect();
        v.sort();
        v
    }

    /// `GET /v1/models` payload — one entry per registered alias,
    /// pulling metadata from the engine itself.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let aliases = self.models();
        aliases
            .into_iter()
            .filter_map(|name| {
                self.engines
                    .get(&name)
                    .map(|engine| engine.model_metadata())
                    .map(|mut m| {
                        // The alias registered with the engine is the
                        // canonical name the SDK uses, not the engine's
                        // internal backend id. Override here.
                        m.name = name;
                        m
                    })
            })
            .collect()
    }
}

/// Validate + dispatch. The HTTP and gRPC layers both call this — it
/// contains the cross-cutting logic (validation, telemetry, routing).
#[instrument(skip(req, registry), fields(model = %req.model, n_questions = req.questions.len()))]
pub async fn dispatch(
    req: SystemRequest,
    registry: &EngineRegistry,
) -> EngineResult<SystemResponse> {
    metrics::counter!("openpick_requests_total").increment(1);

    let start = std::time::Instant::now();
    let engine = registry
        .get(&req.model)
        .ok_or_else(|| EngineError::UnknownModel(req.model.clone()))?;

    validate_request(&req)?;
    let input_tokens = engine.estimate_input_tokens(&req);

    let mut resp = engine.evaluate(req).await?;

    // If the backend didn't fill in usage, do it from the estimator.
    // Real backends will fill it precisely.
    if resp.usage.input_tokens == 0 {
        resp.usage.input_tokens = input_tokens;
    }
    resp.usage.output_tokens = estimate_output_tokens(&resp);

    let elapsed_ms = start.elapsed().as_millis() as f64;
    metrics::histogram!("openpick_request_duration_ms").record(elapsed_ms);
    metrics::counter!("openpick_responses_total").increment(1);

    Ok(resp)
}

fn estimate_output_tokens(resp: &SystemResponse) -> u32 {
    // Noul = 1 token. Choice = 1 (just the picked label).
    // Score = ~ level descriptions worth of tokens.
    resp.answers
        .values()
        .map(|a| match a {
            Answer::Noul(_) => 1,
            Answer::Choice(_) => 1,
            Answer::Score(_) => 4,
        })
        .sum()
}