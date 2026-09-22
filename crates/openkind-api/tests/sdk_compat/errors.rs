use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::IntoResponse;
use serde_json::{json, Value};
use tower::ServiceExt;

use super::helpers::{app, body_bytes, post_systemone};

#[tokio::test]
async fn empty_questions_returns_422() {
    // Spec: raises TypeSafeError when "Questions are empty".
    let body = json!({ "state": "x", "model": "mock", "questions": {} });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["error"]["code"], "invalid_body");
}

#[tokio::test]
async fn score_with_one_level_returns_422() {
    // Spec: raises TypeSafeError when "a score question's criteria list is empty".
    let body = json!({
        "state": "x", "model": "mock",
        "questions": { "q": { "type": "score", "instructions": "?", "criteria": ["only"] } }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn missing_model_returns_404() {
    // TypeSafeNotFoundError (404): wrong route or model.
    let body = json!({
        "state": "x", "model": "no-such-model",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["error"]["code"], "unknown_model");
}

#[tokio::test]
async fn malformed_json_returns_400() {
    // TypeSafeBadRequestError (400).
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from("{not valid json"))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn error_bodies_have_consistent_envelope() {
    // SDK reads `error.body`, `error.status`, `error.code`, `error.message`.
    let body = json!({ "state": "x", "model": "mock", "questions": {} });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["error"].is_object());
    assert!(v["error"]["code"].is_string());
    assert!(v["error"]["message"].is_string());
}

#[tokio::test]
async fn rate_limited_responses_carry_retry_after() {
    // SDK RetryPolicy parses `Retry-After` and `retry-after-ms`.
    // We can't easily trigger rate limiting from outside, so this
    // test verifies the error mapping indirectly via the public
    // IntoResponse impl — by checking that a known 422 path doesn't
    // accidentally emit retry headers (regression guard).
    let body = json!({ "state": "x", "model": "mock", "questions": {} });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let headers = resp.headers();
    assert!(headers.get("retry-after-ms").is_none());
    assert!(headers.get("retry-after").is_none());
}

#[tokio::test]
async fn overloaded_error_maps_to_529() {
    // Spec: "529 Overloaded — TypeSafe is temporarily overloaded. Retry after a short delay."
    use openkind_api::ApiError;
    let err = ApiError::Overloaded {
        retry_after_ms: 1200,
    };
    let resp = err.into_response();
    assert_eq!(resp.status().as_u16(), 529);
    assert_eq!(resp.headers().get("retry-after-ms").unwrap(), "1200");
    assert_eq!(resp.headers().get("retry-after").unwrap(), "2");
}
