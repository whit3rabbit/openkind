//! HTTP layer — axum 0.8.
//!
//! Routes:
//! - `POST /v1/systemone`  → the Jev evaluation endpoint
//! - `GET  /health`        → liveness
//! - `GET  /v1/models`     → list available model aliases (Jev shape)
//! - `GET  /metrics`       → Prometheus scrape
//! - `GET  /playground`    → embedded web UI, only with `--playground on`
//! - `POST /v1/arrow`      → unofficial bulk Arrow IPC endpoint, only with
//!   `--arrow on`; outside the TypeSafe wire contract (see [`crate::arrow`])
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
use openkind_core::{ResponseContract, SystemRequest};
use openkind_engine::{dispatch, EngineRegistry};
use tower_http::trace::{DefaultMakeSpan, TraceLayer};
use tracing::Level;

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
        false,
        false,
    )
}

/// Build the HTTP router with explicit payload size limit and rate limiting.
pub fn router_with_state_auth_rate_limit(
    state: AppState,
    auth: AuthConfig,
    max_payload_bytes: usize,
    rate_limiter: crate::middleware::RateLimiter,
) -> Router {
    router_full(state, auth, max_payload_bytes, rate_limiter, false, false)
}

/// Build the daemon HTTP router: explicit payload size limit, rate limiting,
/// and the embedded playground UI when `playground` is set
/// (`openkindd --playground on`).
pub fn router_daemon(
    state: AppState,
    auth: AuthConfig,
    max_payload_bytes: usize,
    rate_limiter: crate::middleware::RateLimiter,
    playground: bool,
) -> Router {
    router_full(
        state,
        auth,
        max_payload_bytes,
        rate_limiter,
        playground,
        false,
    )
}

/// Build the daemon router with the unofficial Arrow endpoint explicitly enabled.
/// `arrow = false` has the same behavior as [`router_daemon`].
pub fn router_daemon_with_arrow(
    state: AppState,
    auth: AuthConfig,
    max_payload_bytes: usize,
    rate_limiter: crate::middleware::RateLimiter,
    playground: bool,
    arrow: bool,
) -> Router {
    router_full(
        state,
        auth,
        max_payload_bytes,
        rate_limiter,
        playground,
        arrow,
    )
}

fn router_full(
    state: AppState,
    auth: AuthConfig,
    max_payload_bytes: usize,
    rate_limiter: crate::middleware::RateLimiter,
    playground: bool,
    arrow: bool,
) -> Router {
    let routes = Router::new()
        // POST /v1/systemone — canonical Jev decision evaluation endpoint.
        .route("/v1/systemone", post(systemone))
        // POST /v1/system_one — SDK alias for decision evaluation endpoint.
        .route("/v1/system_one", post(systemone))
        // GET /v1/models — list registered models and their capabilities.
        .route("/v1/models", get(list_models))
        // GET /health — unauthenticated service liveness probe.
        .route("/health", get(health))
        // GET /metrics — Prometheus text-format scrape target.
        .route("/metrics", get(prometheus_metrics));
    // GET /playground — embedded web UI. Opt-in because it is a developer
    // convenience outside the wire contract. The HTML shell is public;
    // model lifecycle routes stay behind auth.
    let routes = if playground {
        routes
            .route("/playground", get(crate::playground::playground_page))
            .route(
                "/playground/api/models",
                get(crate::playground::list_models).post(crate::playground::change_model),
            )
    } else {
        routes
    };
    // POST /v1/arrow: unofficial bulk Arrow IPC endpoint. Opt-in like the
    // playground because it is outside the TypeSafe wire contract; it still
    // sits behind the /v1 auth gate and rate limiter.
    let routes = if arrow {
        routes.route("/v1/arrow", post(crate::arrow::arrow_batch))
    } else {
        routes
    };
    // A disabled limiter has no observable effect. Leave its middleware
    // off the router so it cannot allocate or dispatch on every request.
    let routes = if rate_limiter.is_enabled() {
        routes.layer(axum::middleware::from_fn_with_state(
            rate_limiter,
            crate::middleware::rate_limit_layer,
        ))
    } else {
        routes
    };
    routes
        // Order matters: layers added LATER are OUTERMOST. We want
        // request_id outermost so it stamps the response on every code
        // path, including 401s from auth_layer and 429s from the rate
        // limiter (both short-circuit before any handler middleware
        // fires). Authentication sits outside rate limiting so rejected
        // credentials cannot exhaust the budget shared by requests from
        // the same TCP peer (for example, a reverse proxy).
        .layer(axum::middleware::from_fn_with_state(
            auth,
            crate::middleware::auth_layer,
        ))
        .layer(axum::middleware::from_fn(
            crate::middleware::request_id_layer,
        ))
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_bytes))
        // Keep request headers out of telemetry. In particular, auth
        // credentials and caller-supplied request IDs must never become span
        // fields; native-engine metrics use only fixed outcome labels.
        .layer(
            TraceLayer::new_for_http().make_span_with(
                DefaultMakeSpan::new()
                    .level(Level::INFO)
                    .include_headers(false),
            ),
        )
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
    headers: axum::http::HeaderMap,
    req: Result<Json<SystemRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<axum::response::Response, ApiError> {
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
    // Proxy-cache mode: the hook decides whether this alias is proxied and
    // answers from the distilling cache or forwards upstream with the
    // caller's own credentials. Everything else dispatches locally.
    if let Some(proxy) = &state.proxy {
        if proxy.wants(&req) {
            let contract = ResponseContract::from_request(&req)
                .map_err(|error| ApiError::InvalidBody(error.to_string()))?;
            let caller_key = if proxy.forwards_caller_credentials() {
                bearer_of(&headers)
            } else {
                None
            };
            let outcome = proxy.evaluate(req, caller_key).await?;
            contract.validate(&outcome.response).map_err(|error| {
                ApiError::BadGateway(format!("upstream returned an invalid response: {error}"))
            })?;
            let mut response = (axum::http::StatusCode::OK, Json(outcome.response)).into_response();
            let headers = response.headers_mut();
            if let Ok(value) = axum::http::HeaderValue::from_str(outcome.source.as_str()) {
                headers.insert(
                    axum::http::HeaderName::from_static("x-openkind-cache"),
                    value,
                );
            }
            if let Some(detail) = &outcome.detail {
                if let Ok(text) = serde_json::to_string(detail) {
                    if let Ok(value) = axum::http::HeaderValue::from_str(&text) {
                        headers.insert(
                            axum::http::HeaderName::from_static("x-openkind-cache-detail"),
                            value,
                        );
                    }
                }
            }
            return Ok(response);
        }
    }
    let resp = dispatch(req, &state.registry).await?;
    Ok((axum::http::StatusCode::OK, Json(resp)).into_response())
}

/// Extract the caller's bearer credential (the upstream Jev key) without
/// logging or storing it beyond the proxy's salted hash.
fn bearer_of(headers: &axum::http::HeaderMap) -> Option<String> {
    let value = headers.get(axum::http::header::AUTHORIZATION)?;
    let value = value.to_str().ok()?;
    let token = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))?;
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_owned())
    }
}

/// Model listing handler for GET `/v1/models`.
async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // In proxy mode the upstream listing is authoritative for proxied
    // aliases; fall back to the local registry when the upstream cannot be
    // reached.
    if let Some(proxy) = &state.proxy {
        if let Some(models) = proxy.models().await {
            return Json(models).into_response();
        }
    }
    let models = state.registry.list_models();
    Json(ModelsResponse::new(models)).into_response()
}

/// Service liveness probe handler for GET `/health`.
///
/// The body is constant, so it is pre-encoded once instead of rebuilding a
/// `serde_json::Value` and re-serializing it on every probe. The bytes and
/// content type match what `Json(json!({"status":"ok"}))` produced.
async fn health() -> impl IntoResponse {
    static HEALTH_BODY: &str = "{\"status\":\"ok\"}";
    (
        [(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        )],
        HEALTH_BODY,
    )
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
