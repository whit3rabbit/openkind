use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use opendecision_api::{router_with_auth, AuthConfig};
use opendecision_engine::{EngineRegistry, MockEngine};
use serde_json::Value;
use tower::ServiceExt;

pub fn app() -> axum::Router {
    let mut reg = EngineRegistry::new();
    // Register both `mock` and `jev-latest` so we test the SDK's default.
    reg.register("mock", Arc::new(MockEngine::new()));
    reg.register("jev-latest", Arc::new(MockEngine::new()));
    router_with_auth(reg, AuthConfig::default())
}

pub fn app_with_auth(token: &str) -> axum::Router {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    reg.register("jev-latest", Arc::new(MockEngine::new()));
    router_with_auth(reg, AuthConfig::new(Some(token.into())))
}

pub async fn body_bytes(resp: axum::response::Response) -> Vec<u8> {
    resp.into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec()
}

pub async fn post_systemone(
    uri: &str,
    body: Value,
    headers: &[(&str, &str)],
) -> (StatusCode, axum::response::Response) {
    let mut b = Request::builder().method("POST").uri(uri);
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    let req = b
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    let status = resp.status();
    (status, resp)
}

pub async fn get(uri: &str, headers: &[(&str, &str)]) -> (StatusCode, axum::response::Response) {
    let mut b = Request::builder().method("GET").uri(uri);
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    let resp = app().oneshot(b.body(Body::empty()).unwrap()).await.unwrap();
    let status = resp.status();
    (status, resp)
}
