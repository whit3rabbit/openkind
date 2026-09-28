//! Embedded web playground served at `GET /playground` when the daemon runs
//! with `--playground on`.
//!
//! The page is embedded in the binary. Evaluation uses the standard Jev API;
//! opt-in `/playground/api/models` controls delegate to the daemon's local
//! loader and stay outside the public TypeSafe contract.

use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::HeaderValue;
use axum::response::IntoResponse;

/// The playground app, embedded at compile time.
const PLAYGROUND_HTML: &str = include_str!("../assets/playground.html");

/// Marker the router tests pin so an accidentally empty asset fails loudly.
const PLAYGROUND_MARKER: &str = "openkind playground";

/// Handler for `GET /playground`.
///
/// The body is a compile-time constant, so it is served as-is instead of
/// being rebuilt per request. `no-store` keeps browser caches from pinning a
/// stale UI across daemon upgrades.
pub async fn playground_page() -> impl IntoResponse {
    debug_assert!(PLAYGROUND_HTML.contains(PLAYGROUND_MARKER));
    (
        [
            (
                CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            ),
            (CACHE_CONTROL, HeaderValue::from_static("no-store")),
            (axum::http::header::CONTENT_SECURITY_POLICY, HeaderValue::from_static("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")),
            (axum::http::header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff")),
        ],
        PLAYGROUND_HTML,
    )
}

/// A locally available model and its serving state.
#[derive(serde::Serialize)]
pub struct PlaygroundModel {
    /// Alias used in evaluation requests.
    pub name: String,
    /// Human-readable source or profile description.
    pub description: String,
    /// `mock`, `installed`, or `startup`.
    pub source: String,
    /// Whether new requests can use this model.
    pub loaded: bool,
    /// Whether this daemon can explicitly load and unload it.
    pub manageable: bool,
}

/// Daemon-owned artifact loading boundary. The API never opens model files.
#[async_trait::async_trait]
pub trait PlaygroundModels: Send + Sync {
    /// List local installations and configured aliases without downloading.
    async fn list(&self) -> Result<Vec<PlaygroundModel>, crate::ApiError>;
    /// Explicitly load or unload one known local model.
    async fn set_loaded(&self, name: String, loaded: bool) -> Result<(), crate::ApiError>;
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelAction {
    name: String,
    loaded: bool,
}

pub(crate) async fn list_models(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<crate::AppState>>,
) -> Result<impl IntoResponse, crate::ApiError> {
    let manager = state.playground_models.as_ref().ok_or_else(|| {
        crate::ApiError::InvalidBody("Model controls are unavailable on this daemon".into())
    })?;
    Ok((
        [(CACHE_CONTROL, "no-store")],
        axum::Json(serde_json::json!({"models": manager.list().await?})),
    ))
}

pub(crate) async fn change_model(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<crate::AppState>>,
    headers: axum::http::HeaderMap,
    axum::Json(action): axum::Json<ModelAction>,
) -> Result<impl IntoResponse, crate::ApiError> {
    // Requiring a non-simple header prevents another origin from submitting
    // a resource-changing form. This route deliberately has no CORS support.
    if headers
        .get("x-openkind-playground")
        .and_then(|v| v.to_str().ok())
        != Some("1")
        || headers
            .get("sec-fetch-site")
            .is_some_and(|v| v == "cross-site")
    {
        return Err(crate::ApiError::Unauthorized);
    }
    let manager = state.playground_models.as_ref().ok_or_else(|| {
        crate::ApiError::InvalidBody("Model controls are unavailable on this daemon".into())
    })?;
    manager.set_loaded(action.name, action.loaded).await?;
    Ok((
        [(CACHE_CONTROL, "no-store")],
        axum::Json(serde_json::json!({"ok": true})),
    ))
}
