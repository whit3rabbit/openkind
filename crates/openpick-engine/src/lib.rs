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
use openpick_core::{validate_request, Answer, SystemRequest, SystemResponse, ValidationError};
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
            openpick_core::State::Object(m) => {
                serde_json::to_string(m).map(|s| s.len()).unwrap_or(0)
            }
            openpick_core::State::Array(a) => {
                serde_json::to_string(a).map(|s| s.len()).unwrap_or(0)
            }
        };
        let instr_chars: usize = req
            .questions
            .values()
            .map(|q| match q {
                openpick_core::Question::Noul(n) => serde_json::to_string(&n.instructions)
                    .map(|s| s.len())
                    .unwrap_or(0),
                openpick_core::Question::Choice(c) => serde_json::to_string(&c.instructions)
                    .map(|s| s.len())
                    .unwrap_or(0),
                openpick_core::Question::Score(s) => serde_json::to_string(&s.instructions)
                    .map(|s| s.len())
                    .unwrap_or(0),
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

#[cfg(test)]
mod tests {
    use super::*;
    use openpick_core::{NoulQuestion, Question, ScoreQuestion, State};

    fn make_test_request(model: &str) -> SystemRequest {
        let mut questions = HashMap::new();
        questions.insert(
            "urgent".into(),
            Question::Noul(NoulQuestion {
                instructions: serde_json::json!("Is this urgent?"),
                criteria: None,
            }),
        );
        SystemRequest {
            state: State::Text("Please fix now!".into()),
            model: model.into(),
            questions,
        }
    }

    #[test]
    fn engine_registry_register_and_get() {
        let mut registry = EngineRegistry::new();
        let engine = Arc::new(MockEngine::new());
        registry.register("mock-alias", engine.clone());

        assert!(registry.get("mock-alias").is_some());
        assert!(registry.get("non-existent").is_none());
    }

    #[test]
    fn engine_registry_models_are_sorted() {
        let mut registry = EngineRegistry::new();
        let engine = Arc::new(MockEngine::new());
        registry.register("zeta", engine.clone());
        registry.register("alpha", engine.clone());
        registry.register("mid", engine);

        assert_eq!(registry.models(), vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn engine_registry_list_models_overrides_canonical_alias() {
        let mut registry = EngineRegistry::new();
        let engine = Arc::new(MockEngine::with_backend("internal-id"));
        registry.register("public-model-name", engine);

        let models = registry.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "public-model-name");
    }

    #[test]
    fn engine_registry_debug_format() {
        let mut registry = EngineRegistry::new();
        registry.register("m1", Arc::new(MockEngine::new()));
        let debug_str = format!("{registry:?}");
        assert!(debug_str.contains("EngineRegistry"));
        assert!(debug_str.contains("m1"));
    }

    #[tokio::test]
    async fn dispatch_successful_and_populates_token_usage() {
        let mut registry = EngineRegistry::new();
        registry.register("mock", Arc::new(MockEngine::new()));

        let req = make_test_request("mock");
        let resp = dispatch(req, &registry).await.unwrap();

        assert_eq!(resp.model, "mock");
        assert_eq!(resp.answers.len(), 1);
        assert!(resp.usage.input_tokens > 0);
        assert_eq!(resp.usage.output_tokens, 1); // 1 Noul question
    }

    #[tokio::test]
    async fn dispatch_returns_unknown_model_error() {
        let registry = EngineRegistry::new();
        let req = make_test_request("unregistered");
        let err = dispatch(req, &registry).await.unwrap_err();
        assert!(matches!(err, EngineError::UnknownModel(m) if m == "unregistered"));
    }

    #[tokio::test]
    async fn dispatch_returns_validation_error_on_invalid_request() {
        let mut registry = EngineRegistry::new();
        registry.register("mock", Arc::new(MockEngine::new()));

        let req = SystemRequest {
            state: State::Text("hello".into()),
            model: "mock".into(),
            questions: HashMap::new(), // Invalid: empty questions
        };
        let err = dispatch(req, &registry).await.unwrap_err();
        assert!(matches!(
            err,
            EngineError::Invalid(ValidationError::NoQuestions)
        ));
    }

    #[test]
    fn token_estimation_handles_text_object_and_array_states() {
        let engine = MockEngine::new();

        let mut questions = HashMap::new();
        questions.insert(
            "s".into(),
            Question::Score(ScoreQuestion {
                instructions: serde_json::json!("Evaluate"),
                criteria: vec!["L1".into(), "L2".into()],
            }),
        );

        let req_text = SystemRequest {
            state: State::Text("12345678".into()), // 8 chars
            model: "mock".into(),
            questions: questions.clone(),
        };
        let tokens_text = engine.estimate_input_tokens(&req_text);
        assert!(tokens_text > 0);

        let mut obj_map = serde_json::Map::new();
        obj_map.insert("key".into(), serde_json::json!("value with some length"));
        let req_obj = SystemRequest {
            state: State::Object(obj_map),
            model: "mock".into(),
            questions: questions.clone(),
        };
        let tokens_obj = engine.estimate_input_tokens(&req_obj);
        assert!(tokens_obj > 0);

        let req_arr = SystemRequest {
            state: State::Array(vec![
                serde_json::json!("item 1"),
                serde_json::json!("item 2"),
            ]),
            model: "mock".into(),
            questions,
        };
        let tokens_arr = engine.estimate_input_tokens(&req_arr);
        assert!(tokens_arr > 0);
    }
}
