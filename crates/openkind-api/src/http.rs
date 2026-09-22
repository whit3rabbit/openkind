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
//! optional bearer token when `OPENKIND_API_KEY` is set.

use std::sync::Arc;

use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use openkind_core::SystemRequest;
use openkind_engine::{dispatch, EngineRegistry};
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
        // fires). Authentication sits outside rate limiting so rejected
        // credentials cannot exhaust the budget shared by requests from
        // the same TCP peer (for example, a reverse proxy).
        .layer(axum::middleware::from_fn_with_state(
            rate_limiter,
            crate::middleware::rate_limit_layer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            auth,
            crate::middleware::auth_layer,
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
) -> Result<Json<openkind_core::SystemResponse>, ApiError> {
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
#[path = "http_tests.rs"]
mod tests;
