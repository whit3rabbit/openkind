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

fn playground_app(auth: AuthConfig) -> Router {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    router_daemon(
        AppState::new(reg),
        auth,
        MAX_PAYLOAD_SIZE_BYTES,
        crate::middleware::RateLimiter::disabled(),
        true,
    )
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
    // The body is a pre-encoded constant; pin the bytes and content type so
    // probes and SDK health checks keep seeing the JSON shape.
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/json")
    );
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body.as_ref(), b"{\"status\":\"ok\"}");
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
async fn playground_route_is_absent_by_default() {
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/playground")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn playground_route_serves_the_embedded_page_when_enabled() {
    let resp = playground_app(AuthConfig::default())
        .oneshot(
            Request::builder()
                .uri("/playground")
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
        Some("text/html; charset=utf-8")
    );
    // no-store so an upgraded daemon never leaves a stale UI in the cache.
    assert_eq!(
        resp.headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok()),
        Some("no-store")
    );
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("openkind playground"), "marker missing");
}

#[tokio::test]
async fn playground_page_bypasses_auth_while_v1_stays_gated() {
    let gated = playground_app(AuthConfig::new(Some("topsecret".into())));
    let resp = gated
        .clone()
        .oneshot(
            Request::builder()
                .uri("/playground")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = gated
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
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

#[tokio::test]
async fn playground_model_controls_require_opt_in_auth_and_explicit_header() {
    use crate::playground::{PlaygroundModel, PlaygroundModels};
    struct Controls(Arc<EngineRegistry>);
    #[async_trait::async_trait]
    impl PlaygroundModels for Controls {
        async fn list(&self) -> Result<Vec<PlaygroundModel>, ApiError> {
            Ok(vec![PlaygroundModel {
                name: "mock".into(),
                description: "Demo".into(),
                source: "mock".into(),
                loaded: self.0.get("mock").is_some(),
                manageable: true,
            }])
        }
        async fn set_loaded(&self, name: String, loaded: bool) -> Result<(), ApiError> {
            if loaded {
                self.0.register_if_absent(name, Arc::new(MockEngine::new()));
            } else {
                self.0.unregister(&name);
            }
            Ok(())
        }
    }
    let mut registry = EngineRegistry::new();
    registry.register("mock", Arc::new(MockEngine::new()));
    let mut state = AppState::new(registry);
    state.playground_models = Some(Arc::new(Controls(state.registry.clone())));
    let disabled = router_daemon(
        state.clone(),
        AuthConfig::default(),
        MAX_PAYLOAD_SIZE_BYTES,
        crate::RateLimiter::disabled(),
        false,
    );
    let resp = disabled
        .oneshot(
            Request::builder()
                .uri("/playground/api/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let router = router_daemon(
        state.clone(),
        AuthConfig::new(Some("secret".into())),
        MAX_PAYLOAD_SIZE_BYTES,
        crate::RateLimiter::disabled(),
        true,
    );
    let resp = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/playground/api/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert!(resp.headers().contains_key("x-typesafe-request-id"));
    for (header, site, expected) in [
        ("", "same-origin", StatusCode::UNAUTHORIZED),
        ("1", "cross-site", StatusCode::UNAUTHORIZED),
        ("1", "same-origin", StatusCode::OK),
    ] {
        let resp = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/playground/api/models")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .header("x-openkind-playground", header)
                    .header("sec-fetch-site", site)
                    .body(Body::from(r#"{"name":"mock","loaded":false}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), expected);
        assert!(resp.headers().contains_key("x-typesafe-request-id"));
        if expected != StatusCode::OK {
            assert!(state.registry.get("mock").is_some());
        }
    }
    assert!(state.registry.get("mock").is_none());
    let resp = router
        .oneshot(
            Request::builder()
                .uri("/playground/api/models")
                .header("authorization", "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.headers()["cache-control"], "no-store");
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()["models"][0]["loaded"],
        false
    );
}
