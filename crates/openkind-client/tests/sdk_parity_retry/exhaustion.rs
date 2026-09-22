use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use openkind_client::{Error, RequestOptions, RetryPolicy};
use serde_json::json;

use super::common::*;

// ---------------------------------------------------------------------
// test_exhausted_retry_preserves_final_http_error
// ---------------------------------------------------------------------
#[tokio::test]
async fn exhausted_retry_preserves_final_http_error() {
    let (url, requests) = spawn(move |captured: &Captured| {
        let attempt = captured
            .retry_count
            .as_deref()
            .map_or(1, |c| c.parse().unwrap())
            + 1;
        let (status, id): (u16, String) = match attempt {
            1 => (429, "request-1".into()),
            2 => (500, "request-2".into()),
            _ => (503, "request-3".into()),
        };
        Outcome::error(status, json!({"message": format!("attempt {attempt}")}))
            .with_retry_after_ms(0)
            .with_request_id(id)
    })
    .await;
    let client = client(&url, RetryPolicy::new());
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert_eq!(requests.len(), 3);
    // The LAST response is what the caller sees, not the first.
    assert_eq!(err.status(), Some(503));
    assert_eq!(err.request_id(), Some("request-3"));
    match &err {
        Error::Api(api) => {
            assert_eq!(api.message.as_deref(), Some("attempt 3"));
            assert_eq!(api.body.as_ref().unwrap()["message"], "attempt 3");
            assert_eq!(
                api.to_string(),
                "503: attempt 3 (POST /v1/systemone) (request_id=request-3)"
            );
        }
        other => panic!("expected Api error, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// test_connection_retry_recovers — timeout flavor: retried then recovers
// ---------------------------------------------------------------------
#[tokio::test]
async fn timeout_errors_are_retried_then_recover() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_handler = attempts.clone();
    let app = Router::new().route(
        "/v1/systemone",
        post(move |_req: axum::http::Request<axum::body::Body>| {
            let n = attempts_for_handler.fetch_add(1, Ordering::SeqCst);
            async move {
                if n == 0 {
                    // First attempt stalls past the client timeout.
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Json(json!({
                    "model": "client-model",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "answers": {"q": {"type": "noul", "noul": 0.5}}
                }))
                .into_response()
            }
        }),
    );
    let url = spawn_router(app).await;
    let client = client(
        &url,
        RetryPolicy::new()
            .max_retries(3)
            .backoff_initial(Duration::from_millis(1))
            .backoff_jitter(0.0),
    );
    let started = Instant::now();
    let response = client
        .evaluate_with(
            evaluate_request(),
            &RequestOptions::new().timeout(Duration::from_millis(50)),
        )
        .await
        .expect("timeout error must be retried into success");
    assert_eq!(response.answers.len(), 1);
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert!(
        started.elapsed() >= Duration::from_millis(50),
        "first attempt really timed out"
    );
}

// ---------------------------------------------------------------------
// test_cancel_pending_retry — cancellation during the retry sleep
// ---------------------------------------------------------------------
#[tokio::test]
async fn cancel_pending_retry() {
    let (url, _requests) =
        spawn(move |_| Outcome::error(429, json!({})).with_retry_after_ms(60_000)).await;
    // No total budget: the 60s server-requested delay becomes a real sleep,
    // which is what we cancel (the Python suite blocks inside a mocked sleep).
    let client = client(&url, RetryPolicy::new().max_retries(5).total_timeout(None));

    let task = tokio::spawn(async move { client.evaluate(evaluate_request()).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    task.abort();
    let joined = task.await;
    assert!(
        matches!(joined, Err(join_err) if join_err.is_cancelled()),
        "pending retry must be cancellable"
    );
}
