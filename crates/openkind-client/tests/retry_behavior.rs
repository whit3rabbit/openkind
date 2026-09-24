//! Retry, backoff, and rate-limit mechanics against a deterministic stub
//! server. The stub mirrors the openkind server's error contract exactly
//! (`{"error":{"code","message"}}` envelope, `retry-after-ms` +
//! `Retry-After` headers on 429/529), so timing-sensitive behavior can be
//! tested without real rate-limit windows.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use openkind_client::{Client, Error, RequestOptions, RetryPolicy};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn valid_response() -> Value {
    json!({
        "model": "mock",
        "usage": {"input_tokens": 3, "output_tokens": 4},
        "answers": {"q": {"type": "noul", "noul": 0.75}}
    })
}

fn error_envelope(code: &str, message: &str) -> Value {
    json!({"error": {"code": code, "message": message}})
}

#[derive(Clone)]
struct StubState {
    attempts: Arc<AtomicUsize>,
    /// Captured `x-typesafe-retry-count` header of every request received.
    retry_headers: Arc<Mutex<Vec<Option<String>>>>,
    /// How many leading requests fail (with `fail_status`) before 200.
    failures_before_success: usize,
    fail_status: u16,
    /// `retry-after-ms` value sent with failures (0 = omit).
    retry_after_ms: u64,
}

async fn stub_handler(State(state): State<StubState>) -> Response {
    let attempt = state.attempts.fetch_add(1, Ordering::SeqCst);
    if attempt < state.failures_before_success {
        let mut response = (
            StatusCode::from_u16(state.fail_status).unwrap(),
            Json(error_envelope(
                if state.fail_status == 429 {
                    "rate_limited"
                } else if state.fail_status == 529 {
                    "overloaded"
                } else {
                    "internal_error"
                },
                "stub failure",
            )),
        )
            .into_response();
        if state.retry_after_ms > 0 {
            let ms = state.retry_after_ms.to_string();
            let secs = state.retry_after_ms.div_ceil(1000).to_string();
            response
                .headers_mut()
                .insert("retry-after-ms", ms.parse().unwrap());
            response
                .headers_mut()
                .insert("retry-after", secs.parse().unwrap());
        }
        response
    } else {
        Json(valid_response()).into_response()
    }
}

async fn capture_middleware(
    state: State<StubState>,
    request: HttpRequest<Body>,
    next: axum::middleware::Next,
) -> Response {
    let retry_header = request
        .headers()
        .get("x-typesafe-retry-count")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    state.retry_headers.lock().unwrap().push(retry_header);
    next.run(request).await
}

async fn spawn_stub(state: StubState) -> String {
    let app = Router::new()
        .route("/v1/systemone", post(stub_handler))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            capture_middleware,
        ))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn stub_state(failures: usize, status: u16, retry_after_ms: u64) -> StubState {
    StubState {
        attempts: Arc::new(AtomicUsize::new(0)),
        retry_headers: Arc::new(Mutex::new(Vec::new())),
        failures_before_success: failures,
        fail_status: status,
        retry_after_ms,
    }
}

fn fast_client(base_url: &str, policy: RetryPolicy) -> Client {
    Client::builder()
        .api_key("test-key")
        .base_url(base_url)
        .default_model("mock")
        .timeout(Duration::from_secs(5))
        .retry(policy)
        .build()
        .unwrap()
}

fn evaluate_request() -> openkind_client::SystemRequest {
    openkind_client::SystemRequest {
        state: openkind_client::State::Text("hello".into()),
        model: "mock".into(),
        questions: std::collections::HashMap::from_iter([(
            "q".to_string(),
            openkind_client::question::noul("Is this billing?"),
        )]),
    }
}

async fn spawn_oversized_chunked_response(status: u16) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 4096];
        let _ = stream.read(&mut request).await;
        let reason = if status == 200 { "OK" } else { "Bad Request" };
        stream
            .write_all(
                format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
                )
                .as_bytes(),
            )
            .await
            .unwrap();

        let chunk = vec![b'x'; 64 * 1024];
        for _ in 0..=(openkind_client::MAX_RESPONSE_BODY_SIZE / chunk.len()) {
            if stream
                .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                .await
                .is_err()
                || stream.write_all(&chunk).await.is_err()
                || stream.write_all(b"\r\n").await.is_err()
            {
                break;
            }
        }
        let _ = stream.write_all(b"0\r\n\r\n").await;
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn rejects_oversized_chunked_success_response() {
    let url = spawn_oversized_chunked_response(200).await;
    let client = fast_client(&url, RetryPolicy::new().max_retries(0));

    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert!(matches!(
        err,
        Error::ResponseTooLarge {
            limit: openkind_client::MAX_RESPONSE_BODY_SIZE
        }
    ));
}

#[tokio::test]
async fn rejects_oversized_chunked_error_response_without_retrying() {
    let url = spawn_oversized_chunked_response(400).await;
    let client = fast_client(&url, RetryPolicy::new().max_retries(3));

    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert!(matches!(err, Error::ResponseTooLarge { .. }));
}

#[tokio::test]
async fn chunk_size_line_can_span_socket_reads() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 4096];
        let received = stream.read(&mut request).await.unwrap();
        assert!(received > 0);
        let body = b"{\"status\":\"ok\"}";
        stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        // Force the size line's CRLF to cross reads.
        tokio::time::sleep(Duration::from_millis(5)).await;
        stream.write_all(b"\n").await.unwrap();
        stream.write_all(body).await.unwrap();
        stream.write_all(b"\r\n0\r\n\r\n").await.unwrap();
    });

    let client = fast_client(&base_url, RetryPolicy::new().max_retries(0));
    assert_eq!(client.health().await.unwrap().status, "ok");
    server.await.unwrap();
}

#[tokio::test]
async fn retries_429_until_success_and_sends_retry_count_header() {
    let state = stub_state(2, 429, 1);
    let url = spawn_stub(state.clone()).await;
    let client = fast_client(&url, RetryPolicy::new().max_retries(5).backoff_jitter(0.0));

    let response = client.evaluate(evaluate_request()).await.unwrap();
    assert_eq!(response.answers.len(), 1);
    assert_eq!(
        state.attempts.load(Ordering::SeqCst),
        3,
        "1 initial + 2 retries"
    );

    let headers = state.retry_headers.lock().unwrap();
    assert_eq!(headers[0], None, "initial attempt has no retry header");
    assert_eq!(headers[1].as_deref(), Some("1"));
    assert_eq!(headers[2].as_deref(), Some("2"));
}

#[tokio::test]
async fn exhausted_429_retries_surface_rate_limit_error() {
    let state = stub_state(usize::MAX, 429, 25);
    let url = spawn_stub(state.clone()).await;
    let client = fast_client(&url, RetryPolicy::new().max_retries(2).backoff_jitter(0.0));

    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert_eq!(err.status(), Some(429));
    assert_eq!(err.retry_after(), Some(Duration::from_millis(25)));
    assert_eq!(
        state.attempts.load(Ordering::SeqCst),
        3,
        "1 initial + 2 retries"
    );
}

#[tokio::test]
async fn overloaded_529_is_retried_and_then_succeeds() {
    let state = stub_state(1, 529, 1);
    let url = spawn_stub(state.clone()).await;
    let client = fast_client(&url, RetryPolicy::new().max_retries(3));

    let response = client.evaluate(evaluate_request()).await.unwrap();
    assert_eq!(state.attempts.load(Ordering::SeqCst), 2);
    assert_eq!(response.usage.output_tokens, 4);
}

#[tokio::test]
async fn overloaded_529_without_retries_is_overloaded_kind() {
    let state = stub_state(usize::MAX, 529, 100);
    let url = spawn_stub(state).await;
    let client = fast_client(&url, RetryPolicy::new().max_retries(0));

    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert_eq!(err.kind(), Some(openkind_client::ApiErrorKind::Overloaded));
    assert_eq!(err.retry_after(), Some(Duration::from_millis(100)));
}

#[tokio::test]
async fn server_error_500_is_retried() {
    let state = stub_state(1, 500, 0);
    let url = spawn_stub(state.clone()).await;
    let client = fast_client(&url, RetryPolicy::new().max_retries(2).backoff_jitter(0.0));

    client.evaluate(evaluate_request()).await.unwrap();
    assert_eq!(state.attempts.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn client_errors_are_never_retried() {
    for status in [400, 401, 404, 422] {
        let state = stub_state(usize::MAX, status, 0);
        let url = spawn_stub(state.clone()).await;
        let client = fast_client(&url, RetryPolicy::new().max_retries(5));
        let err = client.evaluate(evaluate_request()).await.unwrap_err();
        assert_eq!(err.status(), Some(status));
        assert_eq!(
            state.attempts.load(Ordering::SeqCst),
            1,
            "status {status} must not be retried"
        );
    }
}

#[tokio::test]
async fn retry_after_header_wins_over_long_backoff() {
    // Backoff would be 10s; the server asks for 5ms. Must finish fast.
    let state = stub_state(1, 429, 5);
    let url = spawn_stub(state).await;
    let client = fast_client(
        &url,
        RetryPolicy::new()
            .max_retries(3)
            .backoff_initial(Duration::from_secs(10)),
    );

    let start = std::time::Instant::now();
    client.evaluate(evaluate_request()).await.unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "retry-after must override 10s backoff, took {:?}",
        start.elapsed()
    );
}

#[tokio::test]
async fn total_timeout_budget_stops_retrying() {
    // Every attempt 429s with a 200ms wait; budget is 100ms so at most a
    // couple of attempts happen and the last error surfaces quickly.
    let state = stub_state(usize::MAX, 429, 200);
    let url = spawn_stub(state.clone()).await;
    let client = fast_client(
        &url,
        RetryPolicy::new()
            .max_retries(100)
            .backoff_initial(Duration::from_millis(1))
            .backoff_jitter(0.0)
            .total_timeout(Some(Duration::from_millis(100))),
    );

    let start = std::time::Instant::now();
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert_eq!(err.status(), Some(429));
    assert!(
        start.elapsed() < Duration::from_secs(3),
        "budget must stop runaway retries, took {:?}",
        start.elapsed()
    );
    let attempts = state.attempts.load(Ordering::SeqCst);
    assert!(attempts <= 4, "budget should cap attempts, got {attempts}");
}

#[tokio::test]
async fn connection_error_is_classified_and_retried() {
    // Bind, learn the port, then drop the listener: connects are refused.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let client = fast_client(
        &format!("http://{addr}"),
        RetryPolicy::new()
            .max_retries(1)
            .backoff_initial(Duration::from_millis(1))
            .backoff_jitter(0.0),
    );
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert!(
        matches!(err, Error::Connection(_)),
        "expected Connection error, got {err:?}"
    );
}

#[tokio::test]
async fn garbage_2xx_body_is_decode_error() {
    let app = Router::new().route("/v1/systemone", post(|| async { "this is not json" }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = fast_client(&format!("http://{addr}"), RetryPolicy::new().max_retries(0));
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    match &err {
        Error::Decode { status, .. } => assert_eq!(*status, 200),
        other => panic!("expected Decode error, got {other:?}"),
    }
    // `Error::status` only reports server-rejection statuses; decode failures
    // are transport-local.
    assert_eq!(err.status(), None);
}

#[tokio::test]
async fn per_call_options_override_client_policy() {
    // Client policy is aggressive; the call disables retries entirely.
    let state = stub_state(1, 500, 0);
    let url = spawn_stub(state.clone()).await;
    let client = fast_client(&url, RetryPolicy::new().max_retries(10));

    let err = client
        .evaluate_with(
            evaluate_request(),
            &RequestOptions::new().retry(RetryPolicy::new().max_retries(0)),
        )
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(500));
    assert_eq!(state.attempts.load(Ordering::SeqCst), 1);
}
