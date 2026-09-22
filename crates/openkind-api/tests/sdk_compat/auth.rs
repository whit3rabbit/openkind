use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use openkind_api::{router_with_auth, AuthConfig};
use openkind_engine::{EngineRegistry, MockEngine};
use serde_json::Value;
use tower::ServiceExt;

use super::helpers::{app_with_auth, body_bytes, get};

#[tokio::test]
async fn auth_disabled_by_default_passes_through() {
    let (status, _) = get("/v1/models", &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn auth_enabled_without_token_returns_401() {
    // SDK should raise TypeSafeAuthenticationError(401).
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    let app = router_with_auth(reg, AuthConfig::new(Some("secret-token".into())));

    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert!(resp.headers().get("www-authenticate").is_some());
    assert!(resp.headers().get("x-typesafe-request-id").is_some());
}

#[tokio::test]
async fn auth_enabled_with_wrong_token_returns_401() {
    let app = app_with_auth("correct");
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", "Bearer wrong")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_enabled_with_correct_token_passes() {
    let app = app_with_auth("correct");
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", "Bearer correct")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_does_not_gate_health_or_metrics() {
    let app = app_with_auth("secret");
    for path in ["/health", "/metrics"] {
        let req = Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{path} must remain open");
    }
}

#[tokio::test]
async fn auth_401_body_has_error_envelope() {
    let app = app_with_auth("secret");
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["error"]["code"], "unauthorized");
    assert!(v["error"]["message"].is_string());
}
