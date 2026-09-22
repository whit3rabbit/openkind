//! Unit tests for request ID middleware and header sanitization.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware,
    routing::get,
    Router,
};
use tower::ServiceExt;

use crate::middleware::{
    is_safe_request_id, request_id_layer, MAX_REQUEST_ID_LEN, REQUEST_ID_HEADER,
};

async fn echo() -> &'static str {
    "ok"
}

fn app() -> Router {
    Router::new()
        .route("/health", get(echo))
        .layer(middleware::from_fn(request_id_layer))
}

#[tokio::test]
async fn response_carries_request_id_header() {
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let id = resp
        .headers()
        .get(&REQUEST_ID_HEADER)
        .expect("response must carry x-typesafe-request-id");
    let s = id.to_str().unwrap();
    assert_eq!(s.len(), 36, "expected uuid length, got {s:?}");
}

#[tokio::test]
async fn inbound_request_id_is_honored() {
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header(&REQUEST_ID_HEADER, "test-id-123")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.headers().get(&REQUEST_ID_HEADER).unwrap(),
        "test-id-123"
    );
}

#[test]
fn is_safe_request_id_accepts_safe_shapes() {
    assert!(is_safe_request_id("a"));
    assert!(is_safe_request_id("550e8400-e29b-41d4-a716-446655440000"));
    assert!(is_safe_request_id("allowed-chars_A.b0"));
    assert!(is_safe_request_id(&"a".repeat(MAX_REQUEST_ID_LEN)));
}

#[test]
fn is_safe_request_id_rejects_unsafe_shapes() {
    assert!(!is_safe_request_id(""));
    assert!(!is_safe_request_id(&"a".repeat(MAX_REQUEST_ID_LEN + 1)));
    assert!(!is_safe_request_id("has space"));
    assert!(!is_safe_request_id("semi;colon"));
    assert!(!is_safe_request_id("new\nline"));
    assert!(!is_safe_request_id("tab\tchar"));
    assert!(!is_safe_request_id("sl/ash"));
    assert!(!is_safe_request_id("quer?y"));
    assert!(!is_safe_request_id("ang<l>e"));
    assert!(!is_safe_request_id("émoji"));
}

#[tokio::test]
async fn inbound_unsafe_request_id_is_sanitized() {
    // Injection attempt with unsafe chars (spaces, brackets, semicolons)
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header(&REQUEST_ID_HEADER, "<script>malicious;id</script>")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let id = resp
        .headers()
        .get(&REQUEST_ID_HEADER)
        .unwrap()
        .to_str()
        .unwrap();
    assert_ne!(id, "<script>malicious;id</script>");
    assert_eq!(id.len(), 36, "should fall back to fresh UUIDv4");

    // Oversized ID attempt (> 128 chars)
    let long_id = "a".repeat(200);
    let resp2 = app()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header(&REQUEST_ID_HEADER, long_id)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let id2 = resp2
        .headers()
        .get(&REQUEST_ID_HEADER)
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(id2.len(), 36, "should fall back to fresh UUIDv4");
}
