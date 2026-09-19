//! `openpick-engine`: Runtime-agnostic decision engine abstractions and dispatch.
//!
//! # Architecture & Responsibilities
//! `openpick-engine` defines the central execution abstractions for Jev-compatible decision inference.
//! Per `docs/ARCHITECTURE.md`, the engine layer sits between `openpick-core` and the transport
//! layers (`openpick-api`):
//!
//! `core` ⇐ `engine` ⇐ `api` ⇐ `server/cli`.
//!
//! The engine abstraction is completely transport-agnostic: it accepts a [`SystemRequest`] and returns
//! a [`SystemResponse`], remaining oblivious to whether evaluation was triggered over HTTP/REST or gRPC.
//!
//! # Core Components
//! - [`DecisionEngine`]: Trait implemented by inference backends (e.g. [`MockEngine`], Candle, GGUF/llama.cpp, ONNX).
//! - [`EngineRegistry`]: Thread-safe mapping of public model aliases (e.g. `"jev-latest"`, `"mock"`) to engine instances.
//! - [`dispatch`]: Unified entrypoint that orchestrates validation, metrics recording, token estimation, and backend evaluation.

#![warn(missing_docs)]

pub mod mock;

pub use mock::MockEngine;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use openpick_core::ModelInfo;
use openpick_core::{
    validate_request, validate_response, Answer, Question, SystemRequest, SystemResponse,
    ValidationError,
};
use thiserror::Error;
use tracing::instrument;

/// Errors that can occur during engine execution or model dispatch.
///
/// These errors are translated into corresponding HTTP/gRPC status codes by the API layer.
#[derive(Debug, Error)]
pub enum EngineError {
    /// Request body failed schema validation. Mapped to HTTP 422 Unprocessable Entity.
    #[error("invalid request: {0}")]
    Invalid(#[from] ValidationError),

    /// Requested model alias is not registered. Mapped to HTTP 404 Not Found.
    #[error("no backend registered for model `{0}`")]
    UnknownModel(String),

    /// Underlying backend driver encountered an internal execution failure. Mapped to HTTP 500.
    #[error("backend `{backend}` failed: {message}")]
    Backend {
        /// Identifier of the failing backend.
        backend: String,
        /// Descriptive failure message.
        message: String,
    },
}

/// Specialized Result alias for engine operations returning an [`EngineError`].
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
            openpick_core::State::Object(m) => count_json_bytes(m),
            openpick_core::State::Array(a) => count_json_bytes(a),
        };
        let instr_chars: usize = req
            .questions
            .values()
            .map(|q| match q {
                openpick_core::Question::Noul(n) => count_json_bytes(&n.instructions),
                openpick_core::Question::Choice(c) => count_json_bytes(&c.instructions),
                openpick_core::Question::Score(s) => count_json_bytes(&s.instructions),
            })
            .sum();
        let total_chars = state_chars.saturating_add(instr_chars);
        if total_chars == 0 {
            0
        } else {
            u32::try_from(total_chars.div_ceil(4)).unwrap_or(u32::MAX)
        }
    }
}

/// Zero-allocation byte counter implementing std::io::Write.
struct ByteCounter(usize);

impl std::io::Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn count_json_bytes<T: serde::Serialize + ?Sized>(val: &T) -> usize {
    let mut counter = ByteCounter(0);
    serde_json::to_writer(&mut counter, val)
        .map(|()| counter.0)
        .unwrap_or(0)
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
    /// Construct an empty `EngineRegistry`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a decision engine under the specified model alias (e.g. `"jev-latest"`).
    pub fn register(&mut self, alias: impl Into<String>, engine: Arc<dyn DecisionEngine>) {
        self.engines.insert(alias.into(), engine);
    }

    /// Look up a decision engine by its registered model alias.
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

/// Per-question reference data for response validation: question id → the
/// choice-criteria keys a choice answer must respect (empty for noul/score).
/// Passing the full id set lets `validate_response` also enforce that the
/// engine answered exactly the requested questions, no more and no less.
fn response_criteria(req: &SystemRequest) -> HashMap<String, Vec<String>> {
    req.questions
        .iter()
        .map(|(id, q)| {
            let keys = match q {
                Question::Noul(_) => Vec::new(),
                Question::Choice(c) => {
                    let mut keys: Vec<String> = c.criteria.keys().cloned().collect();
                    keys.sort();
                    keys
                }
                Question::Score(s) => (0..s.criteria.len()).map(|i| i.to_string()).collect(),
            };
            (id.clone(), keys)
        })
        .collect()
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
    let criteria = response_criteria(&req);
    let input_tokens = engine.estimate_input_tokens(&req);

    let mut resp = engine.evaluate(req).await?;

    // Never forward a contract-violating engine response to the client:
    // a bad answer shape is a backend fault, so it maps to Backend (500),
    // not to a client-facing 422.
    if let Err(validation) = validate_response(&resp, &criteria) {
        return Err(EngineError::Backend {
            backend: engine.backend_id().to_string(),
            message: format!("backend returned an invalid response: {validation}"),
        });
    }

    // If the backend didn't fill in usage, do it from the estimator.
    // Real backends will fill it precisely.
    if resp.usage.input_tokens == 0 {
        resp.usage.input_tokens = input_tokens;
    }
    if resp.usage.output_tokens == 0 {
        resp.usage.output_tokens = estimate_output_tokens(&resp);
    }

    let elapsed_ms = start.elapsed().as_millis() as f64;
    metrics::histogram!("openpick_request_duration_ms").record(elapsed_ms);
    metrics::counter!("openpick_responses_total").increment(1);

    Ok(resp)
}

fn estimate_output_tokens(resp: &SystemResponse) -> u32 {
    // Noul = 1 token. Choice = 1 (just the picked label).
    // Score = ~ level descriptions worth of tokens.
    let sum: usize = resp
        .answers
        .values()
        .map(|a| match a {
            Answer::Noul(_) => 1,
            Answer::Choice(_) => 1,
            Answer::Score(_) => 4,
        })
        .sum();
    u32::try_from(sum).unwrap_or(u32::MAX)
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

    struct CustomUsageEngine;
    #[async_trait]
    impl DecisionEngine for CustomUsageEngine {
        fn backend_id(&self) -> &str {
            "custom"
        }
        async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
            let mut answers = HashMap::new();
            for id in req.questions.keys() {
                answers.insert(
                    id.clone(),
                    Answer::Noul(openpick_core::NoulAnswer { noul: 0.5 }),
                );
            }
            Ok(SystemResponse {
                model: req.model,
                answers,
                usage: openpick_core::Usage {
                    input_tokens: 42,
                    output_tokens: 99,
                },
            })
        }
    }

    #[tokio::test]
    async fn dispatch_preserves_engine_reported_tokens() {
        let mut registry = EngineRegistry::new();
        registry.register("custom", Arc::new(CustomUsageEngine));

        let req = make_test_request("custom");
        let resp = dispatch(req, &registry).await.unwrap();
        assert_eq!(resp.usage.input_tokens, 42);
        assert_eq!(resp.usage.output_tokens, 99);
    }

    /// Backend that drops one of the requested answers.
    struct MissingAnswerEngine;
    #[async_trait]
    impl DecisionEngine for MissingAnswerEngine {
        fn backend_id(&self) -> &str {
            "dropper"
        }
        async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
            let mut answers = HashMap::new();
            if let Some(first) = req.questions.keys().next() {
                answers.insert(
                    first.clone(),
                    Answer::Noul(openpick_core::NoulAnswer { noul: 0.5 }),
                );
            }
            Ok(SystemResponse {
                model: req.model,
                answers,
                usage: openpick_core::Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            })
        }
    }

    #[tokio::test]
    async fn dispatch_rejects_backend_that_skips_answers() {
        let mut registry = EngineRegistry::new();
        registry.register("dropper", Arc::new(MissingAnswerEngine));

        let mut questions = HashMap::new();
        for id in ["q1", "q2"] {
            questions.insert(
                id.into(),
                Question::Noul(openpick_core::NoulQuestion {
                    instructions: serde_json::json!("?"),
                    criteria: None,
                }),
            );
        }
        let req = SystemRequest {
            state: openpick_core::State::Text("x".into()),
            model: "dropper".into(),
            questions,
        };
        let err = dispatch(req, &registry).await.unwrap_err();
        assert!(
            matches!(err, EngineError::Backend { ref backend, .. } if backend == "dropper"),
            "expected Backend error, got {err:?}"
        );
    }

    /// Backend that returns NaN probabilities (a classic inference bug).
    struct NanProbabilityEngine;
    #[async_trait]
    impl DecisionEngine for NanProbabilityEngine {
        fn backend_id(&self) -> &str {
            "nan-backend"
        }
        async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
            let mut answers = HashMap::new();
            for id in req.questions.keys() {
                let mut probs = HashMap::new();
                probs.insert("a".into(), f64::NAN);
                probs.insert("b".into(), 0.5);
                answers.insert(
                    id.clone(),
                    Answer::Choice(openpick_core::ChoiceAnswer {
                        choice: "a".into(),
                        probabilities: probs,
                        confidence: 0.5,
                    }),
                );
            }
            Ok(SystemResponse {
                model: req.model,
                answers,
                usage: openpick_core::Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            })
        }
    }

    #[tokio::test]
    async fn dispatch_rejects_backend_with_nan_probabilities() {
        let mut registry = EngineRegistry::new();
        registry.register("nan-backend", Arc::new(NanProbabilityEngine));

        let mut questions = HashMap::new();
        let mut criteria = HashMap::new();
        criteria.insert("a".into(), Some("first".into()));
        criteria.insert("b".into(), None);
        questions.insert(
            "pick".into(),
            Question::Choice(openpick_core::ChoiceQuestion {
                instructions: serde_json::json!("pick"),
                criteria,
            }),
        );
        let req = SystemRequest {
            state: openpick_core::State::Text("x".into()),
            model: "nan-backend".into(),
            questions,
        };
        let err = dispatch(req, &registry).await.unwrap_err();
        assert!(matches!(err, EngineError::Backend { .. }), "got {err:?}");
    }
}
