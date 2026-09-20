//! HTTP layer — axum 0.8.
//!
//! Routes:
//! - `POST /v1/systemone`  → the Jev evaluation endpoint
//! - `GET  /health`        → liveness
//! - `GET  /v1/models`     → list available model aliases (Jev shape)
//! - `GET  /metrics`       → Prometheus scrape
//!
//! Every response is stamped with `x-typesafe-request-id` (the SDK reads
//! this to log per-request correlation), and `/v1/*` is gated by an
//! optional bearer token when `OPENDECISION_API_KEY` is set.

use std::sync::Arc;

use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use opendecision_core::SystemRequest;
use opendecision_engine::{dispatch, EngineRegistry};
use serde_json::json;
use tower_http::trace::TraceLayer;

use crate::error::ApiError;
use crate::middleware::AuthConfig;
use crate::models::ModelsResponse;
use crate::AppState;

/// Maximum allowed request payload size in bytes (16 MB) to protect against DoS memory exhaustion.
pub const MAX_PAYLOAD_SIZE_BYTES: usize = 16 * 1024 * 1024;

/// Build the HTTP router with explicit payload size limit and the default
/// per-IP rate limit (see [`crate::middleware::RateLimitConfig::default`]).
pub fn router_with_state_and_limit(
    state: AppState,
    auth: AuthConfig,
    max_payload_bytes: usize,
) -> Router {
    router_full(
        state,
        auth,
        max_payload_bytes,
        crate::middleware::RateLimiter::new(crate::middleware::RateLimitConfig::default()),
    )
}

/// Build the HTTP router with explicit payload size limit and rate limiting.
pub fn router_with_state_auth_rate_limit(
    state: AppState,
    auth: AuthConfig,
    max_payload_bytes: usize,
    rate_limiter: crate::middleware::RateLimiter,
) -> Router {
    router_full(state, auth, max_payload_bytes, rate_limiter)
}

fn router_full(
    state: AppState,
    auth: AuthConfig,
    max_payload_bytes: usize,
    rate_limiter: crate::middleware::RateLimiter,
) -> Router {
    Router::new()
        // POST /v1/systemone — canonical Jev decision evaluation endpoint.
        .route("/v1/systemone", post(systemone))
        // POST /v1/system_one — SDK alias for decision evaluation endpoint.
        .route("/v1/system_one", post(systemone))
        // GET /v1/models — list registered models and their capabilities.
        .route("/v1/models", get(list_models))
        // GET /health — unauthenticated service liveness probe.
        .route("/health", get(health))
        // GET /metrics — Prometheus text-format scrape target.
        .route("/metrics", get(prometheus_metrics))
        // Order matters: layers added LATER are OUTERMOST. We want
        // request_id outermost so it stamps the response on every code
        // path, including 401s from auth_layer and 429s from the rate
        // limiter (both short-circuit before any handler middleware
        // fires). Rate limiting sits between request-id and auth so
        // unauthenticated traffic is throttled before token comparison.
        .layer(axum::middleware::from_fn_with_state(
            auth,
            crate::middleware::auth_layer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            rate_limiter,
            crate::middleware::rate_limit_layer,
        ))
        .layer(axum::middleware::from_fn(
            crate::middleware::request_id_layer,
        ))
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_bytes))
        .layer(TraceLayer::new_for_http())
        .with_state(Arc::new(state))
}

/// Build the HTTP router with a pre-built state and default 16MB payload limit. Used by the daemon.
pub fn router_with_state(state: AppState, auth: AuthConfig) -> Router {
    router_with_state_and_limit(state, auth, MAX_PAYLOAD_SIZE_BYTES)
}

/// Build the HTTP router with default authentication configuration (no API key required).
pub fn router(registry: EngineRegistry) -> Router {
    router_with_state(AppState::new(registry), AuthConfig::default())
}

/// Build the HTTP router with explicit authentication configuration.
pub fn router_with_auth(registry: EngineRegistry, auth: AuthConfig) -> Router {
    router_with_state(AppState::new(registry), auth)
}

// Re-exported at the crate root for tests.
pub use router_with_state as build_router_with_state;

/// Canonical evaluation handler for POST `/v1/systemone` and `/v1/system_one`.
async fn systemone(
    State(state): State<Arc<AppState>>,
    req: Result<Json<SystemRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<opendecision_core::SystemResponse>, ApiError> {
    let Json(req) = match req {
        Ok(j) => j,
        Err(rejection) => match rejection {
            axum::extract::rejection::JsonRejection::BytesRejection(e) => {
                return Err(ApiError::PayloadTooLarge(e.to_string()));
            }
            axum::extract::rejection::JsonRejection::JsonSyntaxError(e) => {
                return Err(ApiError::BadJson(e.to_string()));
            }
            axum::extract::rejection::JsonRejection::JsonDataError(e) => {
                return Err(ApiError::InvalidBody(e.to_string()));
            }
            other => {
                return Err(ApiError::InvalidBody(other.to_string()));
            }
        },
    };
    let resp = dispatch(req, &state.registry).await?;
    Ok(Json(resp))
}

/// Model listing handler for GET `/v1/models`.
async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let models = state.registry.list_models();
    Json(ModelsResponse::new(models))
}

/// Service liveness probe handler for GET `/health`.
async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

static HANDLE: std::sync::OnceLock<metrics_exporter_prometheus::PrometheusHandle> =
    std::sync::OnceLock::new();

/// Prometheus text-format metrics exporter. Returns a valid empty body
/// even when the recorder hasn't been installed — scrapers should
/// always see 200. Real metrics fire once the server binary calls
/// `install_metrics_recorder()` on startup.
async fn prometheus_metrics() -> impl IntoResponse {
    if let Some(h) = HANDLE.get() {
        (
            axum::http::StatusCode::OK,
            [("content-type", "text/plain; version=0.0.4")],
            h.render(),
        )
    } else {
        (
            axum::http::StatusCode::OK,
            [("content-type", "text/plain; version=0.0.4")],
            "# metrics recorder not installed\n".to_string(),
        )
    }
}

/// Install the Prometheus recorder. Called by the server binary on startup.
/// Exposed here so the binary doesn't have to depend on
/// metrics-exporter-prometheus directly.
pub fn install_metrics_recorder() -> anyhow::Result<()> {
    if HANDLE.get().is_some() {
        return Ok(());
    }
    use metrics_exporter_prometheus::PrometheusBuilder;
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| anyhow::anyhow!("install metrics recorder: {e}"))?;
    let _ = HANDLE.set(handle);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use opendecision_engine::MockEngine;
    use tower::ServiceExt;

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
        assert!(text.starts_with('#') || text.contains("opendecision"));
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
        let custom_app =
            router_with_state_and_limit(AppState::new(reg), AuthConfig::default(), 1024);
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
}
