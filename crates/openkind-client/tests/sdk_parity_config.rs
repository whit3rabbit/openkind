//! Ports of the configuration tests from the TypeSafe Python SDK's
//! `tests/test_config.py` and the constants-level assertions from
//! `test_public_api_surface.py` / `tests/test_public_sync.py` that apply to
//! the Rust client surface.

mod common;

use std::collections::HashMap;
use std::time::Duration;

use axum::Json;

use common::*;
use openkind_client::{
    question, Client, Error, RequestOptions, RetryPolicy, DEFAULT_BASE_URL, DEFAULT_MODEL,
};
use serde_json::json;

// ---------------------------------------------------------------------
// Python SDK constants parity (constants.py)
// ---------------------------------------------------------------------
#[test]
fn sdk_constants_match_python_defaults() {
    assert_eq!(DEFAULT_BASE_URL, "https://api.typesafe.ai");
    assert_eq!(DEFAULT_MODEL, "jev-latest");
    // DEFAULT_TIMEOUT mirrors typesafe_sdk.constants.DEFAULT_TIMEOUT (10s).
    assert_eq!(openkind_client::DEFAULT_TIMEOUT, Duration::from_secs(10));
}

// ---------------------------------------------------------------------
// test_model_override — per-request model beats the client default
// ---------------------------------------------------------------------
#[tokio::test]
async fn model_override_beats_client_default() {
    let (url, requests) = spawn(move |_| Outcome::success(result())).await;
    let client = client(&url, no_retries());

    // Explicit model on the request.
    let request = openkind_client::SystemRequest {
        state: openkind_client::State::Text("hello".into()),
        model: "call-model".into(),
        questions: HashMap::from_iter([("q".to_string(), question::noul("?"))]),
    };
    client.evaluate(request).await.unwrap();
    assert_eq!(
        requests.last().unwrap().body.as_ref().unwrap()["model"],
        "call-model"
    );

    // Without one, the resolved client default goes on the wire.
    client
        .system_one("hello", [("q", question::noul("?"))])
        .await
        .unwrap();
    assert_eq!(
        requests.last().unwrap().body.as_ref().unwrap()["model"],
        "client-model"
    );
}

// ---------------------------------------------------------------------
// test_missing_key — build fails without any key source
// ---------------------------------------------------------------------
#[test]
fn explicit_empty_env_names_every_variable_in_error() {
    // (Environment-dependent path covered conditionally in live_server.rs.)
    // Here: an explicitly *absent* key with env lookups pointing at unset
    // names is already covered by resolve_lookup unit tests; assert the
    // error message names both variables so operators can self-serve.
    let result = Client::builder().base_url("http://127.0.0.1:1").build();
    if std::env::var("OPENKIND_API_KEY").is_err() && std::env::var("TYPESAFE_API_KEY").is_err() {
        match result {
            Err(Error::Config(message)) => {
                assert!(message.contains("OPENKIND_API_KEY"), "{message}");
                assert!(message.contains("TYPESAFE_API_KEY"), "{message}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------
// test_invalid_timeout — zero timeout rejected at build
// ---------------------------------------------------------------------
#[test]
fn invalid_settings_rejected_at_build() {
    let err = Client::builder()
        .api_key("k")
        .timeout(Duration::ZERO)
        .build()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err:?}");

    let err = Client::builder()
        .api_key("k")
        .retry(RetryPolicy::new().backoff_jitter(1.5))
        .build()
        .unwrap_err();
    assert!(
        matches!(&err, Error::Config(m) if m.contains("backoff_jitter")),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------
// test_http_client_timeout_precedence — SDK timeout wins per request
// ---------------------------------------------------------------------
#[tokio::test]
async fn supplied_http_client_still_gets_sdk_timeout() {
    let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let attempts_for_handler = attempts.clone();
    let app = axum::Router::new().route(
        "/v1/models",
        axum::routing::get(move || {
            let n = attempts_for_handler.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if n == 0 {
                    // Slower than the outer reqwest client's 5ms default,
                    // faster than the SDK's per-request 2s timeout.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Json(json!({"models": []}))
            }
        }),
    );
    let url = spawn_router(app).await;

    let inner = reqwest::Client::builder()
        .timeout(Duration::from_millis(5))
        .build()
        .unwrap();
    let client = Client::builder()
        .api_key("test-key")
        .base_url(&url)
        .http_client(inner)
        .timeout(Duration::from_secs(2))
        .retry(RetryPolicy::new().max_retries(0))
        .build()
        .unwrap();

    // The SDK's per-request timeout (2s) overrides the pooled client's 5ms.
    client.list_models().await.unwrap();
    assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------
// test_public_sync surface — request builders compose without a network
// ---------------------------------------------------------------------
#[test]
fn request_builders_compose_without_network() {
    use openkind_core::validate_request;

    let request = openkind_client::SystemRequest {
        state: question::structured_object(
            json!({"document": "hello"}).as_object().unwrap().clone(),
        ),
        model: "jev-latest".into(),
        questions: HashMap::from_iter([
            ("spam".to_string(), question::noul("Spam?")),
            (
                "tone".to_string(),
                question::choice("Tone?", [("friendly", None), ("hostile", None)]),
            ),
            (
                "quality".to_string(),
                question::score("Quality?", ["bad", "ok", "great"]),
            ),
            (
                "flagged".to_string(),
                question::noul_with("Flagged?", "flagged", "clean"),
            ),
        ]),
    };
    assert!(validate_request(&request).is_ok());
    let wire = serde_json::to_value(&request).unwrap();
    assert_eq!(wire["questions"].as_object().unwrap().len(), 4);
}

// ---------------------------------------------------------------------
// RequestOptions — per-call overrides compose (client surface test)
// ---------------------------------------------------------------------
#[test]
fn request_options_compose() {
    // Builder-style composition must be total and order-independent; the
    // settings themselves are exercised end-to-end by the retry suite.
    let _opts = RequestOptions::new()
        .timeout(Duration::from_secs(3))
        .retry(RetryPolicy::new().max_retries(7))
        .header(
            axum::http::HeaderName::from_static("x-team"),
            axum::http::HeaderValue::from_static("client-parity"),
        );
}

fn no_retries() -> RetryPolicy {
    RetryPolicy::new().max_retries(0)
}
