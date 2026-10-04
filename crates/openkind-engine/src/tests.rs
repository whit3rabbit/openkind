//! Unit tests for engine dispatch, registration, and token estimation.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use openkind_core::{
    Answer, ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion, Question, ScoreAnswer,
    ScoreQuestion, State, SystemRequest, SystemResponse, Usage, ValidationError,
};

use crate::dispatch::dispatch;
use crate::engine::DecisionEngine;
use crate::error::{EngineError, EngineResult};
use crate::mock::MockEngine;
use crate::registry::EngineRegistry;

fn make_test_request(model: &str) -> SystemRequest {
    let mut questions = HashMap::default();
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
    let mut reg = EngineRegistry::new();
    assert!(reg.get("mock").is_none());
    reg.register("mock", Arc::new(MockEngine::new()));
    assert!(reg.get("mock").is_some());
}

#[test]
fn engine_registry_models_are_sorted() {
    let mut reg = EngineRegistry::new();
    reg.register("zeta", Arc::new(MockEngine::new()));
    reg.register("alpha", Arc::new(MockEngine::new()));
    reg.register("beta", Arc::new(MockEngine::new()));
    assert_eq!(reg.models(), vec!["alpha", "beta", "zeta"]);
}

#[test]
fn engine_registry_list_models_overrides_canonical_alias() {
    let mut reg = EngineRegistry::new();
    reg.register("alias-to-mock", Arc::new(MockEngine::new()));
    let models = reg.list_models();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "alias-to-mock");
}

#[test]
fn engine_registry_debug_format() {
    let mut reg = EngineRegistry::new();
    reg.register("m1", Arc::new(MockEngine::new()));
    let rendered = format!("{reg:?}");
    assert!(rendered.contains("m1"));
}

#[tokio::test]
async fn dispatch_returns_unknown_model_error() {
    let reg = EngineRegistry::new();
    let req = make_test_request("nonexistent");
    let err = dispatch(req, &reg).await.unwrap_err();
    assert!(matches!(err, EngineError::UnknownModel(m) if m == "nonexistent"));
}

#[tokio::test]
async fn dispatch_returns_validation_error_on_invalid_request() {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    let invalid_req = SystemRequest {
        state: State::Text("hello".into()),
        model: "mock".into(),
        questions: HashMap::default(), // empty questions is invalid
    };
    let err = dispatch(invalid_req, &reg).await.unwrap_err();
    assert!(matches!(
        err,
        EngineError::Invalid(ValidationError::NoQuestions)
    ));
}

#[tokio::test]
async fn dispatch_rejects_backend_answer_with_wrong_primitive() {
    struct WrongPrimitiveBackend;
    #[async_trait]
    impl DecisionEngine for WrongPrimitiveBackend {
        fn backend_id(&self) -> &str {
            "wrong-primitive"
        }

        async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
            let answers = req
                .questions
                .into_keys()
                .map(|id| (id, Answer::Noul(NoulAnswer { noul: 0.5 })))
                .collect();
            Ok(SystemResponse {
                model: "wrong-primitive".into(),
                answers,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            })
        }
    }

    let mut registry = EngineRegistry::new();
    registry.register("wrong-primitive", Arc::new(WrongPrimitiveBackend));
    let request = SystemRequest {
        state: State::Text("test".into()),
        model: "wrong-primitive".into(),
        questions: HashMap::from_iter([(
            "route".into(),
            Question::Choice(ChoiceQuestion {
                instructions: serde_json::json!("Pick"),
                criteria: HashMap::from_iter([("inspect".into(), None)]),
            }),
        )]),
    };
    let err = dispatch(request, &registry).await.unwrap_err();
    assert!(
        matches!(
            &err,
            EngineError::BackendValidation {
                source: ValidationError::AnswerTypeMismatch { .. },
                ..
            }
        ),
        "{err:?}"
    );
    assert!(err.to_string().contains("expected choice answer, got noul"));
}

#[tokio::test]
async fn dispatch_successful_and_populates_token_usage() {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    let req = make_test_request("mock");
    let resp = dispatch(req, &reg).await.unwrap();
    assert_eq!(resp.model, "mock");
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
    assert_eq!(resp.answers.len(), 1);
}

#[tokio::test]
async fn dispatch_preserves_registered_alias_with_different_backend_id() {
    let mut registry = EngineRegistry::new();
    registry.register(
        "public-model",
        Arc::new(MockEngine::with_backend("internal")),
    );

    let response = dispatch(make_test_request("public-model"), &registry)
        .await
        .unwrap();

    assert_eq!(response.model, "public-model");
}

#[tokio::test]
async fn dispatch_preserves_engine_reported_tokens() {
    struct ExplicitTokensBackend;
    #[async_trait]
    impl DecisionEngine for ExplicitTokensBackend {
        fn backend_id(&self) -> &str {
            "explicit"
        }
        async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
            let mut answers = HashMap::default();
            for id in req.questions.keys() {
                answers.insert(id.clone(), Answer::Noul(NoulAnswer { noul: 0.5 }));
            }
            Ok(SystemResponse {
                model: "explicit".into(),
                answers,
                usage: Usage {
                    input_tokens: 999,
                    output_tokens: 888,
                },
            })
        }
    }

    let mut reg = EngineRegistry::new();
    reg.register("explicit", Arc::new(ExplicitTokensBackend));
    let req = make_test_request("explicit");
    let resp = dispatch(req, &reg).await.unwrap();
    assert_eq!(resp.usage.input_tokens, 999);
    assert_eq!(resp.usage.output_tokens, 888);
}

#[tokio::test]
async fn dispatch_estimates_output_tokens_per_answer_kind() {
    struct MultiAnswerBackend;
    #[async_trait]
    impl DecisionEngine for MultiAnswerBackend {
        fn backend_id(&self) -> &str {
            "multi"
        }
        async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
            let mut answers = HashMap::default();
            for (id, q) in &req.questions {
                match q {
                    Question::Noul(_) => {
                        answers.insert(id.clone(), Answer::Noul(NoulAnswer { noul: 0.5 }));
                    }
                    Question::Choice(c) => {
                        let pick = c.criteria.keys().next().cloned().unwrap();
                        let mut probs = HashMap::default();
                        for k in c.criteria.keys() {
                            probs.insert(k.clone(), 1.0 / c.criteria.len() as f64);
                        }
                        answers.insert(
                            id.clone(),
                            Answer::Choice(ChoiceAnswer {
                                choice: pick,
                                probabilities: probs,
                                confidence: 0.8,
                            }),
                        );
                    }
                    Question::Score(s) => {
                        let mut probs = HashMap::default();
                        let mut legend = HashMap::default();
                        for (i, name) in s.criteria.iter().enumerate() {
                            probs.insert(i.to_string(), 1.0 / s.criteria.len() as f64);
                            legend.insert(i.to_string(), name.clone());
                        }
                        answers.insert(
                            id.clone(),
                            Answer::Score(ScoreAnswer {
                                score: (s.criteria.len() - 1) as f64 / 2.0,
                                legend,
                                probabilities: probs,
                                confidence: 0.9,
                            }),
                        );
                    }
                }
            }
            Ok(SystemResponse {
                model: "multi".into(),
                answers,
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
            })
        }
    }

    let mut reg = EngineRegistry::new();
    reg.register("multi", Arc::new(MultiAnswerBackend));
    let mut questions = HashMap::default();
    questions.insert(
        "q1".into(),
        Question::Noul(NoulQuestion {
            instructions: serde_json::json!("Check"),
            criteria: None,
        }),
    );
    let mut choice_criteria = HashMap::default();
    choice_criteria.insert("a".into(), None);
    choice_criteria.insert("b".into(), None);
    questions.insert(
        "q2".into(),
        Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick"),
            criteria: choice_criteria,
        }),
    );
    questions.insert(
        "q3".into(),
        Question::Score(ScoreQuestion {
            instructions: serde_json::json!("Rate"),
            criteria: vec!["Low".into(), "High".into()],
        }),
    );

    let req = SystemRequest {
        state: State::Text("Context".into()),
        model: "multi".into(),
        questions,
    };
    let resp = dispatch(req, &reg).await.unwrap();
    // 1 (Noul) + 1 (Choice) + 4 (Score) = 6 tokens estimated
    assert_eq!(resp.usage.output_tokens, 6);
}

#[tokio::test]
async fn dispatch_rejects_backend_that_skips_answers() {
    struct DroppingBackend;
    #[async_trait]
    impl DecisionEngine for DroppingBackend {
        fn backend_id(&self) -> &str {
            "drop"
        }
        async fn evaluate(&self, _req: SystemRequest) -> EngineResult<SystemResponse> {
            Ok(SystemResponse {
                model: "drop".into(),
                answers: HashMap::default(), // returns nothing!
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            })
        }
    }

    let mut reg = EngineRegistry::new();
    reg.register("drop", Arc::new(DroppingBackend));
    let req = make_test_request("drop");
    let err = dispatch(req, &reg).await.unwrap_err();
    assert!(
        matches!(err, EngineError::BackendValidation { ref backend, .. } if backend == "drop"),
        "expected Backend error, got {err:?}"
    );
}

#[tokio::test]
async fn dispatch_rejects_backend_with_nan_probabilities() {
    struct NanBackend;
    #[async_trait]
    impl DecisionEngine for NanBackend {
        fn backend_id(&self) -> &str {
            "nan"
        }
        async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
            let mut answers = HashMap::default();
            for id in req.questions.keys() {
                answers.insert(id.clone(), Answer::Noul(NoulAnswer { noul: f64::NAN }));
            }
            Ok(SystemResponse {
                model: "nan".into(),
                answers,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            })
        }
    }

    let mut reg = EngineRegistry::new();
    reg.register("nan", Arc::new(NanBackend));
    let req = make_test_request("nan");
    let err = dispatch(req, &reg).await.unwrap_err();
    assert!(
        matches!(err, EngineError::BackendValidation { ref backend, .. } if backend == "nan"),
        "expected Backend error due to NaN in answer, got {err:?}"
    );
}

#[test]
fn default_model_metadata_names_backend() {
    struct BareEngine;
    #[async_trait]
    impl DecisionEngine for BareEngine {
        fn backend_id(&self) -> &str {
            "bare"
        }
        async fn evaluate(&self, _req: SystemRequest) -> EngineResult<SystemResponse> {
            unimplemented!("metadata-only test")
        }
    }
    let engine = BareEngine;
    let meta = engine.model_metadata();
    assert_eq!(meta.name, "bare");
    assert!(meta.description.contains("bare"));
}

#[test]
fn token_estimation_handles_text_object_and_array_states() {
    let engine = MockEngine::new();

    // 1. Text state
    let req_text = SystemRequest {
        state: State::Text("12345678".into()), // 8 chars -> 2 tokens
        model: "mock".into(),
        questions: {
            let mut q = HashMap::default();
            q.insert(
                "q1".into(),
                Question::Noul(NoulQuestion {
                    instructions: serde_json::json!("1234"), // 4 chars
                    criteria: None,
                }),
            );
            q
        },
    };
    // (8 + 6) chars = 14 chars -> div_ceil(4) = 4 tokens
    assert_eq!(engine.estimate_input_tokens(&req_text), 4);

    // 2. Object state
    let mut map = serde_json::Map::new();
    map.insert("key".into(), serde_json::json!("value")); // serialized JSON length
    let req_obj = SystemRequest {
        state: State::Object(map),
        model: "mock".into(),
        questions: HashMap::default(),
    };
    assert!(engine.estimate_input_tokens(&req_obj) > 0);

    // 3. Array state
    let req_arr = SystemRequest {
        state: State::Array(vec![serde_json::json!(1), serde_json::json!(2)]),
        model: "mock".into(),
        questions: HashMap::default(),
    };
    assert!(engine.estimate_input_tokens(&req_arr) > 0);
}

#[test]
fn token_estimation_counts_criteria_for_every_question_kind() {
    let engine = MockEngine::new();
    let description = "1234".repeat(1_000);
    let requests = [
        (
            Question::Noul(NoulQuestion {
                instructions: serde_json::json!("Check"),
                criteria: None,
            }),
            Question::Noul(NoulQuestion {
                instructions: serde_json::json!("Check"),
                criteria: Some(openkind_core::NoulCriteria {
                    r#true: description.clone(),
                    r#false: "No".into(),
                }),
            }),
        ),
        (
            Question::Choice(ChoiceQuestion {
                instructions: serde_json::json!("Check"),
                criteria: HashMap::from_iter([("a".into(), None)]),
            }),
            Question::Choice(ChoiceQuestion {
                instructions: serde_json::json!("Check"),
                criteria: HashMap::from_iter([("a".into(), Some(description.clone()))]),
            }),
        ),
        (
            Question::Score(ScoreQuestion {
                instructions: serde_json::json!("Check"),
                criteria: vec!["a".into()],
            }),
            Question::Score(ScoreQuestion {
                instructions: serde_json::json!("Check"),
                criteria: vec![description],
            }),
        ),
    ];

    for (short, long) in requests {
        let request = |question| SystemRequest {
            state: State::Text("state".into()),
            model: "mock".into(),
            questions: HashMap::from_iter([("q".into(), question)]),
        };
        let short_tokens = engine.estimate_input_tokens(&request(short));
        let long_tokens = engine.estimate_input_tokens(&request(long));
        assert!(
            long_tokens >= short_tokens + 999,
            "a 4,000-byte criterion must contribute to estimated usage"
        );
    }
}

#[tokio::test]
async fn registry_unload_preserves_accepted_handles_and_updates_clones() {
    let mut registry = EngineRegistry::new();
    let engine: Arc<dyn DecisionEngine> = Arc::new(MockEngine::new());
    let weak = Arc::downgrade(&engine);
    registry.register("model", engine);
    let grpc_registry = registry.clone();
    let accepted = grpc_registry.get("model").unwrap();
    assert!(!registry.register_if_absent("model".into(), Arc::new(MockEngine::new())));
    drop(registry.unregister("model"));
    assert!(grpc_registry.models().is_empty());
    assert!(matches!(
        dispatch(make_test_request("model"), &grpc_registry).await,
        Err(EngineError::UnknownModel(_))
    ));
    assert!(accepted.evaluate(make_test_request("model")).await.is_ok());
    assert!(weak.upgrade().is_some());
    drop(accepted);
    assert!(weak.upgrade().is_none());
    assert!(registry.register_if_absent("model".into(), Arc::new(MockEngine::new())));
    assert!(grpc_registry.get("model").is_some());
}

// ------------------------------------------------------------------
// Backend-fault matrix: every contract violation a backend can commit
// maps to EngineError::Backend (HTTP 500), never to a client 422.
// ------------------------------------------------------------------

struct FaultBackend {
    backend_id: &'static str,
    answer: Answer,
    extra_id: Option<&'static str>,
    error: Option<EngineError>,
}

impl FaultBackend {
    fn answer(backend_id: &'static str, answer: Answer) -> Self {
        Self {
            backend_id,
            answer,
            extra_id: None,
            error: None,
        }
    }
}

#[async_trait]
impl DecisionEngine for FaultBackend {
    fn backend_id(&self) -> &str {
        self.backend_id
    }

    async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
        if let Some(error) = &self.error {
            return Err(match error {
                EngineError::Overloaded {
                    backend,
                    retry_after_ms,
                } => EngineError::Overloaded {
                    backend: backend.clone(),
                    retry_after_ms: *retry_after_ms,
                },
                _ => unreachable!("the fault test only synthesizes Overloaded"),
            });
        }
        let mut answers = HashMap::default();
        for id in req.questions.keys() {
            answers.insert(id.clone(), self.answer.clone());
        }
        if let Some(extra) = self.extra_id {
            answers.insert(extra.to_string(), self.answer.clone());
        }
        Ok(SystemResponse {
            model: req.model,
            answers,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
            },
        })
    }
}

fn choice_question(keys: &[&str]) -> Question {
    Question::Choice(ChoiceQuestion {
        instructions: serde_json::json!("Pick"),
        criteria: keys.iter().map(|key| ((*key).to_string(), None)).collect(),
    })
}

fn score_question() -> Question {
    Question::Score(ScoreQuestion {
        instructions: serde_json::json!("Rate"),
        criteria: vec!["low".into(), "high".into()],
    })
}

fn fault_request(question: Question) -> SystemRequest {
    SystemRequest {
        state: State::Text("state".into()),
        model: "faulty".into(),
        questions: HashMap::from_iter([("q".into(), question)]),
    }
}

fn fault_registry(answer: Answer) -> EngineRegistry {
    let mut registry = EngineRegistry::new();
    registry.register(
        "faulty",
        Arc::new(FaultBackend::answer("faulty-backend", answer)),
    );
    registry
}

fn assert_backend_fault(error: EngineError, needle: &str) {
    assert!(
        matches!(error, EngineError::BackendValidation { .. }),
        "expected a backend fault, got: {error:?}"
    );
    assert!(error.to_string().contains(needle), "unexpected: {error}");
}

#[tokio::test]
async fn extra_answers_are_a_backend_fault() {
    let mut registry = EngineRegistry::new();
    let mut backend = FaultBackend::answer(
        "faulty-backend",
        Answer::Choice(ChoiceAnswer {
            choice: "a".into(),
            probabilities: HashMap::from_iter([("a".to_string(), 1.0)]),
            confidence: 0.5,
        }),
    );
    backend.extra_id = Some("ghost");
    registry.register("faulty", Arc::new(backend));

    let error = dispatch(fault_request(choice_question(&["a"])), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "unknown question `ghost`");
}

#[tokio::test]
async fn out_of_list_choices_are_a_backend_fault() {
    let registry = fault_registry(Answer::Choice(ChoiceAnswer {
        choice: "z".into(),
        probabilities: HashMap::from_iter([("a".to_string(), 1.0)]),
        confidence: 0.5,
    }));
    let error = dispatch(fault_request(choice_question(&["a", "b"])), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "not in criteria keys");
}

#[tokio::test]
async fn probability_key_mismatches_are_a_backend_fault() {
    let registry = fault_registry(Answer::Choice(ChoiceAnswer {
        choice: "a".into(),
        probabilities: HashMap::from_iter([("ghost".to_string(), 1.0)]),
        confidence: 0.5,
    }));
    let error = dispatch(fault_request(choice_question(&["a", "b"])), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "keys must match criteria");
}

#[tokio::test]
async fn out_of_range_probabilities_are_a_backend_fault() {
    let registry = fault_registry(Answer::Choice(ChoiceAnswer {
        choice: "a".into(),
        probabilities: HashMap::from_iter([("a".to_string(), 1.5), ("b".to_string(), -0.5)]),
        confidence: 0.5,
    }));
    let error = dispatch(fault_request(choice_question(&["a", "b"])), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "probability must be in 0..=1");
}

#[tokio::test]
async fn non_summing_probabilities_are_a_backend_fault() {
    let registry = fault_registry(Answer::Choice(ChoiceAnswer {
        choice: "a".into(),
        probabilities: HashMap::from_iter([("a".to_string(), 0.5), ("b".to_string(), 0.4)]),
        confidence: 0.5,
    }));
    let error = dispatch(fault_request(choice_question(&["a", "b"])), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "do not sum to 1");
}

#[tokio::test]
async fn out_of_range_confidence_is_a_backend_fault() {
    let registry = fault_registry(Answer::Choice(ChoiceAnswer {
        choice: "a".into(),
        probabilities: HashMap::from_iter([("a".to_string(), 1.0)]),
        confidence: 1.5,
    }));
    let error = dispatch(fault_request(choice_question(&["a"])), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "confidence");
}

#[tokio::test]
async fn score_legend_mismatch_is_a_backend_fault() {
    let registry = fault_registry(Answer::Score(ScoreAnswer {
        score: 0.0,
        probabilities: HashMap::from_iter([("0".to_string(), 1.0)]),
        legend: HashMap::from_iter([("1".to_string(), "high".to_string())]),
        confidence: 0.5,
    }));
    let error = dispatch(fault_request(score_question()), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "legend");
}

#[tokio::test]
async fn word_keyed_score_legends_violate_the_requested_rubric() {
    // Through dispatch the rubric always binds first: a self-consistent
    // word-keyed legend (`low`/`high`) still fails because the requested
    // two-level rubric pins legend slots `0` and `1`. The non-numeric-index
    // branch itself is only reachable for rubric-less responses (covered at
    // the core validator level).
    let registry = fault_registry(Answer::Score(ScoreAnswer {
        score: 0.0,
        probabilities: HashMap::from_iter([("low".to_string(), 1.0), ("high".to_string(), 0.0)]),
        legend: HashMap::from_iter([
            ("low".to_string(), "low".to_string()),
            ("high".to_string(), "high".to_string()),
        ]),
        confidence: 0.5,
    }));
    let error = dispatch(fault_request(score_question()), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "legend must match");
}

#[tokio::test]
async fn out_of_range_scores_are_a_backend_fault() {
    let registry = fault_registry(Answer::Score(ScoreAnswer {
        score: 5.0,
        probabilities: HashMap::from_iter([("0".to_string(), 1.0)]),
        legend: HashMap::from_iter([("0".to_string(), "low".to_string())]),
        confidence: 0.5,
    }));
    let error = dispatch(fault_request(score_question()), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "score");
}

#[tokio::test]
async fn score_expectation_mismatch_is_a_backend_fault() {
    // A full legend whose mass sits entirely at level 0 cannot produce the
    // reported score of 1.0.
    let registry = fault_registry(Answer::Score(ScoreAnswer {
        score: 1.0,
        probabilities: HashMap::from_iter([("0".to_string(), 1.0), ("1".to_string(), 0.0)]),
        legend: HashMap::from_iter([
            ("0".to_string(), "low".to_string()),
            ("1".to_string(), "high".to_string()),
        ]),
        confidence: 0.5,
    }));
    let error = dispatch(fault_request(score_question()), &registry)
        .await
        .unwrap_err();
    assert_backend_fault(error, "expected probability-weighted score 0");
}

#[tokio::test]
async fn backend_errors_propagate_unchanged_through_dispatch() {
    let mut registry = EngineRegistry::new();
    registry.register(
        "faulty",
        Arc::new(FaultBackend {
            backend_id: "faulty-backend",
            answer: Answer::Noul(NoulAnswer { noul: 0.5 }),
            extra_id: None,
            error: Some(EngineError::Overloaded {
                backend: "faulty-backend".into(),
                retry_after_ms: 25,
            }),
        }),
    );
    let error = dispatch(fault_request(choice_question(&["a"])), &registry)
        .await
        .unwrap_err();
    assert!(
        matches!(
            error,
            EngineError::Overloaded {
                retry_after_ms: 25,
                ..
            }
        ),
        "backend errors must not be rewrapped: {error:?}"
    );
}

#[tokio::test]
async fn unknown_model_wins_over_request_validation() {
    // An invalid request (no questions) naming an unregistered model fails
    // at the registry lookup, matching the documented precedence.
    let registry = EngineRegistry::new();
    let request = SystemRequest {
        state: State::Text("state".into()),
        model: "ghost".into(),
        questions: HashMap::default(),
    };
    let error = dispatch(request, &registry).await.unwrap_err();
    assert!(
        matches!(error, EngineError::UnknownModel(ref model) if model == "ghost"),
        "unexpected error: {error:?}"
    );
}
