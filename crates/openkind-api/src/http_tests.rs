use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use openkind_engine::{EngineRegistry, MockEngine};
use serde_json::json;
use tower::ServiceExt;

use super::*;
use crate::middleware::AuthConfig;
use crate::AppState;

fn app() -> Router {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    router(reg)
}

#[tokio::test]
async fn health_endpoint_returns_ok() {
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
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_text() {
    // Must be 200 with a text body even when no recorder is installed,
    // so scrapers never see errors.
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/plain; version=0.0.4")
    );
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert!(!body.is_empty());
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.starts_with('#') || text.contains("openkind"));
}

#[tokio::test]
async fn models_endpoint_returns_jev_shape() {
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    // Jev shape: top-level `models` array with name/description/release_date.
    assert!(v["models"].is_array(), "expected `models` array, got {v}");
    assert!(v.get("data").is_none());
    assert!(v.get("object").is_none());
    let arr = v["models"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["name"], "mock");
    assert!(arr[0]["description"].is_string());
    assert!(arr[0]["release_date"].is_string());
}

#[tokio::test]
async fn systemone_evaluates_request_and_returns_one_answer_per_question() {
    let body = json!({
        "state": "Help!",
        "model": "mock",
        "questions": {
            "is_urgent": { "type": "noul", "instructions": "?" },
            "dept": {
                "type": "choice",
                "instructions": "?",
                "criteria": { "billing": "pay", "tech": "bugs" }
            },
            "frust": {
                "type": "score",
                "instructions": "?",
                "criteria": ["Calm", "Angry"]
            }
        }
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["model"].as_str(), Some("mock"));
    let answers = v["answers"].as_object().unwrap();
    assert!(answers.contains_key("is_urgent"));
    assert!(answers.contains_key("dept"));
    assert!(answers.contains_key("frust"));
    let usage = &v["usage"];
    assert!(usage["input_tokens"].as_u64().unwrap() > 0);
    assert!(usage["output_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn unknown_model_returns_404() {
    let body = json!({
        "state": "x",
        "model": "no-such-model",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn invalid_body_returns_422() {
    let body = json!({
        "state": "x",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn malformed_json_returns_400() {
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
async fn request_body_exceeding_custom_limit_is_rejected() {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    let custom_app = router_with_state_and_limit(AppState::new(reg), AuthConfig::default(), 1024);
    let big_body = serde_json::to_vec(&json!({
        "state": "x",
        "model": "mock",
        "padding": "x".repeat(2048)
    }))
    .unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(big_body))
        .unwrap();
    let resp = custom_app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn rejected_auth_does_not_consume_rate_limit_budget() {
    let peer: axum::extract::ConnectInfo<std::net::SocketAddr> =
        axum::extract::ConnectInfo("127.0.0.1:40000".parse().unwrap());
    let limited_app = router_with_state_auth_rate_limit(
        AppState::new(EngineRegistry::new()),
        AuthConfig::new(Some("topsecret".into())),
        MAX_PAYLOAD_SIZE_BYTES,
        crate::middleware::RateLimiter::new(crate::middleware::RateLimitConfig {
            max_requests: 1,
            window: std::time::Duration::from_secs(60),
        }),
    );

    for _ in 0..2 {
        let resp = limited_app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .extension(peer)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    let resp = limited_app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "Bearer topsecret")
                .extension(peer)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
