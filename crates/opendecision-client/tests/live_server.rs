//! End-to-end tests against a real `opendecision-api` HTTP server (in-process,
//! over a real TCP socket): wire conformance with `jev-v1-request.json` /
//! `jev-v1-response.json`, auth, server error envelopes, and the real rate
//! limiter's 429 + `retry-after-ms` contract.
//!
//! These tests are the client-side mirror of the server's `sdk_compat.rs`
//! contract suite — anything they assert about responses is what the
//! TypeSafe Python SDK would also observe.

use std::sync::Arc;
use std::time::Duration;

use opendecision_api::middleware::RateLimiter;
use opendecision_api::{AppState, AuthConfig, RateLimitConfig};
use opendecision_client::error::ApiErrorKind;
use opendecision_client::{question, Client, Error, RetryPolicy};
use opendecision_engine::{EngineRegistry, MockEngine};

/// Spawn a real axum server on an ephemeral port and return its base URL.
async fn spawn_server(auth_key: Option<&str>, rate_limit: Option<RateLimitConfig>) -> String {
    let mut registry = EngineRegistry::new();
    registry.register("mock", Arc::new(MockEngine::new()));
    registry.register("jev-latest", Arc::new(MockEngine::new()));
    let state = AppState::new(registry);
    let auth = AuthConfig::new(auth_key.map(str::to_owned));
    let router = match rate_limit {
        Some(config) => opendecision_api::http::router_with_state_auth_rate_limit(
            state,
            auth,
            16 * 1024 * 1024,
            RateLimiter::new(config),
        ),
        None => opendecision_api::router_with_state(state, auth),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    // ConnectInfo is required for the per-IP rate limiter to see peer addresses.
    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .unwrap();
    });
    format!("http://{addr}")
}

fn test_client(base_url: &str) -> Client {
    Client::builder()
        .api_key("test-key")
        .base_url(base_url)
        .default_model("mock")
        .timeout(Duration::from_secs(5))
        .retry(RetryPolicy::new().max_retries(0))
        .build()
        .unwrap()
}

#[tokio::test]
async fn system_one_round_trip_all_question_types() {
    let base_url = spawn_server(Some("test-key"), None).await;
    let client = test_client(&base_url);

    let response = client
        .system_one(
            question::text("I was charged twice. Please help."),
            [
                ("billing", question::noul("Is this about billing?")),
                (
                    "tone",
                    question::choice("What is the tone?", [("calm", None), ("angry", None)]),
                ),
                (
                    "severity",
                    question::score("How severe?", ["low", "medium", "high"]),
                ),
                (
                    "dispute",
                    question::noul_with("Does the user dispute?", "disputed", "accepted"),
                ),
            ],
        )
        .await
        .unwrap();

    assert_eq!(response.model, "mock");
    assert_eq!(response.answers.len(), 4);
    // Wire types are the shared core structs, so typed access Just Works.
    match &response.answers["billing"] {
        opendecision_client::Answer::Noul(answer) => {
            assert!((0.0..=1.0).contains(&answer.noul));
        }
        other => panic!("expected noul answer, got {other:?}"),
    }
    match &response.answers["tone"] {
        opendecision_client::Answer::Choice(answer) => {
            assert!(["calm", "angry"].contains(&answer.choice.as_str()));
            let sum: f64 = answer.probabilities.values().sum();
            assert!((sum - 1.0).abs() < 1e-9, "probabilities must sum to 1");
        }
        other => panic!("expected choice answer, got {other:?}"),
    }
    match &response.answers["severity"] {
        opendecision_client::Answer::Score(answer) => {
            assert_eq!(answer.legend.len(), 3);
        }
        other => panic!("expected score answer, got {other:?}"),
    }
}

#[tokio::test]
async fn evaluate_with_explicit_system_request() {
    let base_url = spawn_server(Some("test-key"), None).await;
    let client = test_client(&base_url);

    let request = opendecision_client::SystemRequest {
        state: opendecision_client::State::Object(
            serde_json::json!({
                "user": {"id": 7},
                "logs": ["payment failed", "user asked for help"],
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
        model: "mock".into(),
        questions: std::collections::HashMap::from([(
            "billing".to_string(),
            question::noul("Is this about billing?"),
        )]),
    };

    let response = client.evaluate(request).await.unwrap();
    assert_eq!(response.answers.len(), 1);
}

#[tokio::test]
async fn wrong_api_key_is_authentication_error_with_request_id() {
    let base_url = spawn_server(Some("real-key"), None).await;
    let client = Client::builder()
        .api_key("wrong-key")
        .base_url(&base_url)
        .default_model("mock")
        .retry(RetryPolicy::new().max_retries(0))
        .build()
        .unwrap();

    let err = client
        .system_one("hello", [("q", question::noul("Is this billing?"))])
        .await
        .unwrap_err();

    assert_eq!(err.kind(), Some(ApiErrorKind::Authentication));
    assert_eq!(err.status(), Some(401));
    let api = match &err {
        Error::Api(api) => api,
        other => panic!("expected Api error, got {other:?}"),
    };
    assert_eq!(api.code.as_deref(), Some("unauthorized"));
    // request_id_layer stamps every response, 401s included.
    assert!(
        api.request_id.as_deref().unwrap_or_default().len() >= 32,
        "request id must be present: {api:?}"
    );
}

#[tokio::test]
async fn unknown_model_is_not_found() {
    let base_url = spawn_server(Some("test-key"), None).await;
    let client = test_client(&base_url);

    let request = opendecision_client::SystemRequest {
        state: opendecision_client::State::Text("hello".into()),
        model: "no-such-model".into(),
        questions: std::collections::HashMap::from([(
            "q".to_string(),
            question::noul("Is this billing?"),
        )]),
    };
    let err = client.evaluate(request).await.unwrap_err();

    assert_eq!(err.kind(), Some(ApiErrorKind::NotFound));
    assert_eq!(err.status(), Some(404));
    match &err {
        Error::Api(api) => assert_eq!(api.code.as_deref(), Some("unknown_model")),
        other => panic!("expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn schema_violation_is_unprocessable_entity() {
    // Passes deserialization but fails core validation: score rubrics need
    // >= 2 levels. The server 422s with `invalid_body`.
    let base_url = spawn_server(Some("test-key"), None).await;
    let client = test_client(&base_url);

    let err = client
        .system_one(
            "hello",
            [("s", question::score("Rate this", ["only-level"]))],
        )
        .await
        .unwrap_err();

    assert_eq!(err.kind(), Some(ApiErrorKind::UnprocessableEntity));
    assert_eq!(err.status(), Some(422));
    match &err {
        Error::Api(api) => assert_eq!(api.code.as_deref(), Some("invalid_body")),
        other => panic!("expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn list_models_round_trip() {
    let base_url = spawn_server(Some("test-key"), None).await;
    let client = test_client(&base_url);

    let models = client.list_models().await.unwrap();
    let names: Vec<&str> = models.models.iter().map(|m| m.name.as_str()).collect();
    assert!(names.contains(&"mock"));
    assert!(names.contains(&"jev-latest"));
}

#[tokio::test]
async fn health_works_without_valid_key() {
    // /health is outside the auth gate; a wrong key must not matter.
    let base_url = spawn_server(Some("real-key"), None).await;
    let client = Client::builder()
        .api_key("anything")
        .base_url(&base_url)
        .build()
        .unwrap();
    let health = client.health().await.unwrap();
    assert_eq!(health.status, "ok");
}

#[tokio::test]
async fn real_rate_limiter_429_surfaces_rate_limit_error_with_retry_after() {
    let base_url = spawn_server(
        Some("test-key"),
        Some(RateLimitConfig {
            max_requests: 1,
            window: Duration::from_secs(60),
        }),
    )
    .await;
    let client = test_client(&base_url);

    let first = client
        .system_one("hello", [("q", question::noul("Is this billing?"))])
        .await;
    assert!(first.is_ok(), "first request must pass the limiter");

    // Retries disabled: the raw 429 must surface with the server's delay.
    let err = client
        .system_one("hello", [("q", question::noul("Is this billing?"))])
        .await
        .unwrap_err();

    assert_eq!(err.kind(), Some(ApiErrorKind::RateLimit));
    assert_eq!(err.status(), Some(429));
    match &err {
        Error::Api(api) => {
            assert_eq!(api.code.as_deref(), Some("rate_limited"));
            assert!(
                api.retry_after.is_some(),
                "server must send retry-after headers: {api:?}"
            );
            assert!(api.request_id.is_some());
        }
        other => panic!("expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn question_schema_validation_is_left_to_api() {
    // Port of test_clients.py::test_question_schema_validation_is_left_to_api:
    // structurally-constructed but semantically invalid questions go over the
    // wire untouched and the SERVER rejects them with 422 invalid_body.
    let base_url = spawn_server(Some("test-key"), None).await;
    let client = test_client(&base_url);

    // Score rubric with zero levels.
    let err = client
        .system_one(
            "hello",
            [(
                "s",
                opendecision_client::Question::Score(opendecision_core::ScoreQuestion {
                    instructions: serde_json::json!("Rate this"),
                    criteria: Vec::new(),
                }),
            )],
        )
        .await
        .unwrap_err();
    assert_eq!(err.kind(), Some(ApiErrorKind::UnprocessableEntity));
    match &err {
        Error::Api(api) => assert_eq!(api.code.as_deref(), Some("invalid_body")),
        other => panic!("expected Api error, got {other:?}"),
    }

    // Choice with zero options.
    let err = client
        .system_one(
            "hello",
            [(
                "c",
                opendecision_client::Question::Choice(opendecision_core::ChoiceQuestion {
                    instructions: serde_json::json!("Pick one"),
                    criteria: std::collections::HashMap::new(),
                }),
            )],
        )
        .await
        .unwrap_err();
    assert_eq!(err.kind(), Some(ApiErrorKind::UnprocessableEntity));

    // Empty instructions string.
    let err = client
        .system_one(
            "hello",
            [(
                "n",
                opendecision_client::Question::Noul(opendecision_core::NoulQuestion {
                    instructions: serde_json::json!(""),
                    criteria: None,
                }),
            )],
        )
        .await
        .unwrap_err();
    assert_eq!(err.kind(), Some(ApiErrorKind::UnprocessableEntity));
}

#[tokio::test]
async fn missing_api_key_fails_at_build_time() {
    // Build without an explicit key. If the host machine has neither
    // OPENDECISION_API_KEY nor TYPESAFE_API_KEY set, the build must fail
    // with a Config error (rather than a runtime 401).
    let result = Client::builder().base_url("http://127.0.0.1:1").build();
    if std::env::var("OPENDECISION_API_KEY").is_err() && std::env::var("TYPESAFE_API_KEY").is_err()
    {
        match result {
            Err(Error::Config(message)) => {
                assert!(message.contains("API key"), "{message}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn client_evaluates_openapi_documented_examples() {
    let base_url = spawn_server(Some("test-key"), None).await;
    let client = test_client(&base_url);

    // 1. OpenAPI quickstart Noul example
    let resp1 = client
        .system_one(
            "I was charged twice for order #1042.",
            [(
                "billing",
                question::noul("Is this inquiry related to a billing issue?"),
            )],
        )
        .await
        .unwrap();
    assert_eq!(resp1.model, "mock");
    match &resp1.answers["billing"] {
        opendecision_client::Answer::Noul(ans) => {
            assert!((0.0..=1.0).contains(&ans.noul));
        }
        other => panic!("expected Noul, got {other:?}"),
    }

    // 2. OpenAPI multi-question evaluation example
    let resp2 = client
        .system_one(
            "Customer support transcript: Agent resolved issue in 4 minutes.",
            [
                (
                    "is_resolved",
                    question::noul_with(
                        "Was the issue resolved?",
                        "Customer issue was successfully resolved.",
                        "Issue remains unresolved or escalated.",
                    ),
                ),
                (
                    "department",
                    question::choice(
                        "Which team handled this request?",
                        [
                            ("billing", Some("Payments, invoicing, refunds".to_string())),
                            ("technical", Some("Bugs, outages, integrations".to_string())),
                            ("sales", None),
                        ],
                    ),
                ),
                (
                    "satisfaction",
                    question::score(
                        "Rate customer satisfaction",
                        ["Dissatisfied", "Neutral", "Delighted"],
                    ),
                ),
            ],
        )
        .await
        .unwrap();

    assert_eq!(resp2.answers.len(), 3);
    match &resp2.answers["is_resolved"] {
        opendecision_client::Answer::Noul(ans) => assert!((0.0..=1.0).contains(&ans.noul)),
        other => panic!("expected Noul, got {other:?}"),
    }
    match &resp2.answers["department"] {
        opendecision_client::Answer::Choice(ans) => {
            assert!(["billing", "technical", "sales"].contains(&ans.choice.as_str()));
            assert!((0.0..=1.0).contains(&ans.confidence));
        }
        other => panic!("expected Choice, got {other:?}"),
    }
    match &resp2.answers["satisfaction"] {
        opendecision_client::Answer::Score(ans) => {
            assert!((0.0..=2.0).contains(&ans.score));
            assert_eq!(ans.legend.len(), 3);
            assert!((0.0..=1.0).contains(&ans.confidence));
        }
        other => panic!("expected Score, got {other:?}"),
    }
}
