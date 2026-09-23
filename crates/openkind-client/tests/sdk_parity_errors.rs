//! Ports of the error-handling tests from the TypeSafe Python SDK's
//! `tests/test_clients.py` (`test_error_mapping`, `test_error_messages`,
//! `test_transport_errors`, `test_headers_timeout_and_logging`) and
//! `tests/test_errors.py` (request context, credential redaction, body edge
//! cases).

mod common;

use axum::http::HeaderMap;
use axum::Json;

use std::time::Duration;

use common::*;
use openkind_client::{ApiErrorKind, Error};
use serde_json::json;

// ---------------------------------------------------------------------
// test_error_mapping — status → kind, request id, retry-after, display
// ---------------------------------------------------------------------
#[tokio::test]
async fn error_mapping_matches_python_exception_taxonomy() {
    let cases: &[(u16, ApiErrorKind)] = &[
        (400, ApiErrorKind::BadRequest),
        (401, ApiErrorKind::Authentication),
        (403, ApiErrorKind::PermissionDenied),
        (404, ApiErrorKind::NotFound),
        (422, ApiErrorKind::UnprocessableEntity),
        (429, ApiErrorKind::RateLimit),
        (500, ApiErrorKind::InternalServer),
        (503, ApiErrorKind::InternalServer),
        (408, ApiErrorKind::Other),
        (409, ApiErrorKind::Other),
        (302, ApiErrorKind::Other),
    ];
    for &(status, kind) in cases {
        let body = json!({"detail": {"message": "Server explanation"}});
        let (url, _requests) = spawn(move |_| {
            Outcome::error(status, body.clone())
                .with_request_id("req_123")
                .with_retry_after_ms(125)
        })
        .await;
        // Retries off so each status lands as a single observed error.
        let client = client(&url, openkind_client::RetryPolicy::new().max_retries(0));
        let err = client.list_models().await.unwrap_err();

        assert_eq!(err.kind(), Some(kind), "status {status}");
        assert_eq!(err.status(), Some(status));
        match &err {
            Error::Api(api) => {
                assert_eq!(
                    api.body.as_ref().unwrap()["detail"]["message"],
                    "Server explanation"
                );
                assert_eq!(api.request_id.as_deref(), Some("req_123"));
                assert_eq!(api.message.as_deref(), Some("Server explanation"));
                assert_eq!(api.retry_after, Some(Duration::from_millis(125)));
                let displayed = api.to_string();
                assert_eq!(
                    displayed,
                    format!("{status}: Server explanation (GET /v1/models) (request_id=req_123)")
                );
            }
            other => panic!("expected Api error for {status}, got {other:?}"),
        }
        if status == 429 {
            assert_eq!(err.retry_after(), Some(Duration::from_millis(125)));
        }
    }
}

// ---------------------------------------------------------------------
// test_error_messages — lenient message extraction priority
// ---------------------------------------------------------------------
#[tokio::test]
async fn error_messages_match_python_extraction_matrix() {
    let cases: &[(serde_json::Value, &str)] = &[
        (
            json!({"error": "error", "message": "message", "detail": "detail"}),
            "error",
        ),
        (
            json!({"error": {"message": "nested error"}, "message": "message"}),
            "nested error",
        ),
        (json!({"message": "message", "detail": "detail"}), "message"),
        (json!({"detail": "detail"}), "detail"),
        (
            json!({"detail": {"message": "nested detail"}}),
            "nested detail",
        ),
        (
            json!({"detail": [
                {"loc": ["body", "questions", "q", "score", "criteria", 0], "msg": "Invalid"},
                {"msg": "Missing"},
                {}
            ]}),
            "questions.q.score.criteria.0: Invalid; Missing",
        ),
    ];
    for (body, message) in cases {
        let expected = body.clone();
        let (url, _requests) = spawn(move |_| Outcome::error(400, expected.clone())).await;
        let client = client(&url, openkind_client::RetryPolicy::new().max_retries(0));
        let err = client.list_models().await.unwrap_err();
        match &err {
            Error::Api(api) => {
                assert_eq!(api.message.as_deref(), Some(*message), "body {body}");
            }
            other => panic!("expected Api error, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn non_json_error_bodies_fall_back_like_python() {
    // Plain-text body: message is the raw text.
    let url = spawn_router(common_plain_text_router("plain text".into())).await;
    let plain_client = client(&url, openkind_client::RetryPolicy::new().max_retries(0));
    match plain_client.list_models().await.unwrap_err() {
        Error::Api(api) => assert_eq!(api.message.as_deref(), Some("plain text")),
        other => panic!("expected Api error, got {other:?}"),
    }

    // A JSON body with no recognizable message: the raw serialized body is
    // surfaced (Python asserts '{"unexpected":true}').
    let url = spawn_router(common_raw_body_router(
        400,
        br#"{"unexpected":true}"#.to_vec(),
    ))
    .await;
    let raw_client = client(&url, openkind_client::RetryPolicy::new().max_retries(0));
    match raw_client.list_models().await.unwrap_err() {
        Error::Api(api) => {
            assert_eq!(api.message.as_deref(), Some(r#"{"unexpected":true}"#));
            assert_eq!(api.code, None);
        }
        other => panic!("expected Api error, got {other:?}"),
    }
}

/// Stub serving a raw non-JSON error body on `/v1/models`.
fn common_plain_text_router(text: String) -> axum::Router {
    axum::Router::new().route(
        "/v1/models",
        axum::routing::get(
            move || async move { (axum::http::StatusCode::BAD_REQUEST, text.clone()) },
        ),
    )
}

/// Stub serving arbitrary bytes with a status on `/v1/models`.
fn common_raw_body_router(status: u16, body: Vec<u8>) -> axum::Router {
    axum::Router::new().route(
        "/v1/models",
        axum::routing::get(move || {
            let body = body.clone();
            async move { (axum::http::StatusCode::from_u16(status).unwrap(), body) }
        }),
    )
}

// ---------------------------------------------------------------------
// test_transport_errors — connection vs timeout classification
// ---------------------------------------------------------------------
#[tokio::test]
async fn transport_errors_classify_like_python() {
    // ReadTimeout analog: stalled response + per-call timeout.
    let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let attempts_for_handler = attempts.clone();
    let app = axum::Router::new().route(
        "/v1/models",
        axum::routing::get(move || {
            let n = attempts_for_handler.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if n == 0 {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Json(json!({"models": []}))
            }
        }),
    );
    let url = spawn_router(app).await;
    let stall_client = client(&url, openkind_client::RetryPolicy::new().max_retries(0));
    let err = stall_client
        .list_models_with(&RequestOptions::new().timeout(Duration::from_millis(50)))
        .await
        .unwrap_err();
    match err {
        Error::Timeout { timeout } => assert_eq!(timeout, Duration::from_millis(50)),
        other => panic!("expected Timeout, got {other:?}"),
    }

    // ConnectError analog: nothing is listening on the port.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let refused_client = client(
        &format!("http://{addr}"),
        openkind_client::RetryPolicy::new().max_retries(0),
    );
    let err = refused_client.list_models().await.unwrap_err();
    assert!(matches!(err, Error::Connection(_)), "got {err:?}");
}

// ---------------------------------------------------------------------
// test_api_error_request_context + endpoint credential redaction
// ---------------------------------------------------------------------
#[tokio::test]
async fn api_error_endpoint_names_resource_without_credentials() {
    let (url, _requests) = spawn(move |_| {
        Outcome::error(
            404,
            json!({"error": {"code": "unknown_model", "message": "nope"}}),
        )
    })
    .await;
    let client = client(&url, openkind_client::RetryPolicy::new().max_retries(0));
    let err = client.list_models().await.unwrap_err();
    match &err {
        Error::Api(api) => {
            assert_eq!(api.endpoint.as_deref(), Some("GET /v1/models"));
            // The endpoint and display never include the bearer key.
            let displayed = api.to_string();
            assert!(!displayed.contains("test-key"), "{displayed}");
        }
        other => panic!("expected Api error, got {other:?}"),
    }

    // Evaluate errors name the evaluate endpoint.
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    match &err {
        Error::Api(api) => assert_eq!(api.endpoint.as_deref(), Some("POST /v1/systemone")),
        other => panic!("expected Api error, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// test_error_body_edge_cases — empty and non-UTF8 bodies
// ---------------------------------------------------------------------
#[tokio::test]
async fn error_body_edge_cases_do_not_panic() {
    // Empty body.
    let url = spawn_router(common_raw_body_router(500, Vec::new())).await;
    let empty_client = client(&url, openkind_client::RetryPolicy::new().max_retries(0));
    match empty_client.list_models().await.unwrap_err() {
        Error::Api(api) => {
            assert_eq!(api.message, None);
            assert_eq!(api.body, None);
        }
        other => panic!("expected Api error, got {other:?}"),
    }

    // Non-UTF8 bytes: lossy conversion, no panic.
    let url = spawn_router(common_raw_body_router(500, vec![0xff, 0xfe, b'x'])).await;
    let lossy_client = client(&url, openkind_client::RetryPolicy::new().max_retries(0));
    match lossy_client.list_models().await.unwrap_err() {
        Error::Api(api) => assert!(api.message.is_some()),
        other => panic!("expected Api error, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// test_exception_reconstruction — errors cross threads (Send + Sync)
// ---------------------------------------------------------------------
#[test]
fn errors_are_send_sync_across_threads() {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<Error>();
    assert_send_sync::<openkind_client::ApiError>();
    assert_send_sync::<openkind_client::RetryPolicy>();
}

// ---------------------------------------------------------------------
// test_headers_timeout_and_logging — protected header precedence
// ---------------------------------------------------------------------
#[tokio::test]
async fn protected_headers_cannot_be_overridden() {
    let received: std::sync::Arc<std::sync::Mutex<Vec<HeaderMap>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let received_for_handler = received.clone();
    let app = axum::Router::new().route(
        "/prefix/v1/systemone",
        axum::routing::post(move |headers: HeaderMap, _body: axum::body::Bytes| {
            let received = received_for_handler.clone();
            async move {
                received.lock().unwrap().push(headers);
                Json(json!({
                    "model": "jev-latest",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "answers": {"q": {"type": "noul", "noul": 0.5}}
                }))
            }
        }),
    );
    // Trailing slashes in the prefix must be trimmed by the client.
    let base = spawn_router(app).await;
    let url = format!("{base}/prefix///");

    // The Python test injects hostile values for every protected header.
    let client = Client::builder()
        .api_key("test-key")
        .base_url(&url)
        .default_model("client-model")
        .timeout(Duration::from_secs(5))
        .header(
            axum::http::HeaderName::from_static("authorization"),
            axum::http::HeaderValue::from_static("injected-secret"),
        )
        .header(
            axum::http::HeaderName::from_static("accept"),
            axum::http::HeaderValue::from_static("text/plain"),
        )
        .header(
            axum::http::HeaderName::from_static("user-agent"),
            axum::http::HeaderValue::from_static("wrong"),
        )
        .header(
            axum::http::HeaderName::from_static("x-typesafe-sdk"),
            axum::http::HeaderValue::from_static("wrong"),
        )
        .header(
            axum::http::HeaderName::from_static("x-typesafe-runtime"),
            axum::http::HeaderValue::from_static("wrong"),
        )
        .header(
            axum::http::HeaderName::from_static("x-team"),
            axum::http::HeaderValue::from_static("default"),
        )
        .header(
            axum::http::HeaderName::from_static("x-default"),
            axum::http::HeaderValue::from_static("kept"),
        )
        .build()
        .unwrap();

    let opts = RequestOptions::new()
        .header(
            axum::http::HeaderName::from_static("x-team"),
            axum::http::HeaderValue::from_static("call"),
        )
        // Smuggling the retry-count header must be ignored.
        .header(
            axum::http::HeaderName::from_static("x-typesafe-retry-count"),
            axum::http::HeaderValue::from_static("99"),
        )
        .header(
            axum::http::HeaderName::from_static("x-call"),
            axum::http::HeaderValue::from_static("present"),
        );

    let response = client
        .system_one_with("hello", [("q", question::noul("?"))], &opts)
        .await;
    assert!(response.is_ok(), "{response:?}");

    let headers = received.lock().unwrap();
    assert_eq!(headers.len(), 1);
    let h = &headers[0];
    assert_eq!(h["authorization"], "Bearer test-key");
    assert_eq!(h["accept"], "application/json");
    assert!(h["user-agent"]
        .to_str()
        .unwrap()
        .starts_with("openkind-client/"));
    assert_eq!(h["x-typesafe-sdk"], h["user-agent"]);
    assert!(h["x-typesafe-runtime"]
        .to_str()
        .unwrap()
        .starts_with("rust"));
    assert!(
        h.get("x-typesafe-retry-count").is_none(),
        "initial attempt carries no retry count, and extras cannot smuggle one"
    );
    assert_eq!(
        h["x-team"], "call",
        "per-call extras win over client defaults"
    );
    assert_eq!(h["x-default"], "kept");
    assert_eq!(h["content-type"], "application/json");
}

use openkind_client::{question, Client, RequestOptions};
