use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::IntoResponse;
use serde_json::{json, Value};
use tower::ServiceExt;

use super::helpers::{app, app_with_auth, body_bytes, post_systemone};

// --- Retries: retry headers on 429 & 529, omitted on success & validation error ---

#[tokio::test]
async fn retries_retry_after_headers_presence_on_rate_limit_and_overload() {
    // Spec: docs.typesafe.ai/sdk/python/api/retries
    // SDK inspects `Retry-After` (seconds) and `retry-after-ms` (ms) on 429 and 529
    use opendecision_api::ApiError;

    // RateLimited (429)
    let rl = ApiError::RateLimited {
        retry_after_ms: 2500,
    };
    let resp = rl.into_response();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(resp.headers().get("retry-after-ms").unwrap(), "2500");
    assert_eq!(resp.headers().get("retry-after").unwrap(), "3");

    // Overloaded (529)
    let ol = ApiError::Overloaded {
        retry_after_ms: 1500,
    };
    let resp = ol.into_response();
    assert_eq!(resp.status().as_u16(), 529);
    assert_eq!(resp.headers().get("retry-after-ms").unwrap(), "1500");
    assert_eq!(resp.headers().get("retry-after").unwrap(), "2");
}

#[tokio::test]
async fn retries_headers_absent_on_success_and_client_error() {
    // Spec: docs.typesafe.ai/sdk/python/api/retries
    // 200 OK must not have retry headers
    let body = json!({
        "state": "Retry check",
        "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert!(resp.headers().get("retry-after").is_none());
    assert!(resp.headers().get("retry-after-ms").is_none());

    // 422 Unprocessable Entity must not have retry headers
    let bad_body = json!({
        "state": "bad",
        "model": "mock",
        "questions": {}
    });
    let (status422, resp422) = post_systemone("/v1/systemone", bad_body, &[]).await;
    assert_eq!(status422, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(resp422.headers().get("retry-after").is_none());
    assert!(resp422.headers().get("retry-after-ms").is_none());
}

// --- Exceptions: SDK exception mapping & consistent error envelope ---

#[tokio::test]
async fn exceptions_all_error_responses_contain_request_id_and_envelope() {
    // Spec: docs.typesafe.ai/sdk/python/api/exceptions
    // SDK exceptions store `.request_id`, `.message`, `.code`, `.http_status`.
    // Test that every error condition sets x-typesafe-request-id and has {"error": {"code", "message"}}

    // 400 Bad JSON -> BadRequestError
    let req400 = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from("{invalid json"))
        .unwrap();
    let resp400 = app().oneshot(req400).await.unwrap();
    assert_eq!(resp400.status(), StatusCode::BAD_REQUEST);
    assert!(resp400.headers().get("x-typesafe-request-id").is_some());
    let v400: Value = serde_json::from_slice(&body_bytes(resp400).await).unwrap();
    assert_eq!(v400["error"]["code"], "bad_json");

    // 401 Unauthorized -> AuthenticationError
    let req401 = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let resp401 = app_with_auth("secret").oneshot(req401).await.unwrap();
    assert_eq!(resp401.status(), StatusCode::UNAUTHORIZED);
    assert!(resp401.headers().get("x-typesafe-request-id").is_some());
    assert!(resp401.headers().get("www-authenticate").is_some());
    let v401: Value = serde_json::from_slice(&body_bytes(resp401).await).unwrap();
    assert_eq!(v401["error"]["code"], "unauthorized");

    // 404 Unknown Model -> NotFoundError
    let body404 = json!({
        "state": "x",
        "model": "nonexistent-model-xyz",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status404, resp404) = post_systemone("/v1/systemone", body404, &[]).await;
    assert_eq!(status404, StatusCode::NOT_FOUND);
    assert!(resp404.headers().get("x-typesafe-request-id").is_some());
    let v404: Value = serde_json::from_slice(&body_bytes(resp404).await).unwrap();
    assert_eq!(v404["error"]["code"], "unknown_model");

    // 422 Unprocessable Entity -> UnprocessableEntityError
    let body422 = json!({
        "state": "x",
        "model": "mock",
        "questions": {}
    });
    let (status422, resp422) = post_systemone("/v1/systemone", body422, &[]).await;
    assert_eq!(status422, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(resp422.headers().get("x-typesafe-request-id").is_some());
    let v422: Value = serde_json::from_slice(&body_bytes(resp422).await).unwrap();
    assert!(v422["error"]["code"].is_string());
}

// --- Constants: DEFAULT_MODEL and AuthConfig API key resolution ---

#[tokio::test]
async fn constants_default_model_jev_latest_is_evaluable() {
    // Spec: docs.typesafe.ai/sdk/python/api/constants
    // SDK default model is `jev-latest`.
    let body = json!({
        "state": "Checking default model",
        "model": "jev-latest",
        "questions": { "q": { "type": "noul", "instructions": "test" } }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["model"], "jev-latest");
}

#[tokio::test]
async fn constants_typesafe_api_key_auth_fallback_integration() {
    // Spec: docs.typesafe.ai/sdk/python/api/constants
    // SDK uses TYPESAFE_API_KEY env var.
    // Ensure that when an auth layer is configured with the key expected by the SDK,
    // clients supplying the Bearer token pass through cleanly.
    let app = app_with_auth("typesafe-test-token-xyz");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .header("authorization", "Bearer typesafe-test-token-xyz")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "state": "auth ok",
                "model": "mock",
                "questions": { "q": { "type": "noul", "instructions": "ok?" } }
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
