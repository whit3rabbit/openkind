//! API errors. Mapped to HTTP status codes per the Jev spec and the
//! Python SDK exception taxonomy:
//! - 400 `TypeSafeBadRequestError`           — malformed JSON
//! - 401 `TypeSafeAuthenticationError`       — missing/invalid API key
//! - 404 `TypeSafeNotFoundError`             — unknown model alias
//! - 422 `TypeSafeUnprocessableEntityError`  — body validation failed
//! - 429 `TypeSafeRateLimitError`            — too many requests (carries Retry-After)
//! - 529 `TypeSafeAPIOverloadedError`        — server overloaded (carries Retry-After)
//! - 5xx `TypeSafeInternalServerError`       — engine/backend failure
//!
//! Any 429/529 response includes `retry-after-ms` and (if the caller
//! requested it) `Retry-After` HTTP headers, so the SDK's `RetryPolicy`
//! can back off correctly.

use axum::{
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use openpick_engine::EngineError;
use serde_json::json;
use std::time::Duration;

const RETRY_AFTER_MS: HeaderName = HeaderName::from_static("retry-after-ms");
const RETRY_AFTER: HeaderName = HeaderName::from_static("retry-after");

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("invalid request body: {0}")]
    InvalidBody(String),

    #[error("invalid JSON: {0}")]
    BadJson(String),

    #[error("missing or invalid API key")]
    Unauthorized,

    #[error("rate limited; retry after {retry_after_ms} ms")]
    RateLimited { retry_after_ms: u64 },

    #[error("server overloaded; retry after {retry_after_ms} ms")]
    Overloaded { retry_after_ms: u64 },

    #[error("engine error: {0}")]
    Engine(#[from] EngineError),

    #[error("internal error: {0}")]
    Internal(String),
}

impl ApiError {
    fn status_and_code(&self) -> (StatusCode, &'static str) {
        match self {
            ApiError::InvalidBody(_) | ApiError::Engine(EngineError::Invalid(_)) => {
                (StatusCode::UNPROCESSABLE_ENTITY, "invalid_body")
            }
            ApiError::BadJson(_) => (StatusCode::BAD_REQUEST, "bad_json"),
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized"),
            ApiError::RateLimited { .. } => (StatusCode::TOO_MANY_REQUESTS, "rate_limited"),
            ApiError::Overloaded { .. } => (StatusCode::SERVICE_UNAVAILABLE, "overloaded"),
            ApiError::Engine(EngineError::UnknownModel(_)) => {
                (StatusCode::NOT_FOUND, "unknown_model")
            }
            ApiError::Engine(EngineError::Backend { .. }) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "backend_error")
            }
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        }
    }

    /// Retry-After / retry-after-ms in milliseconds, if this error carries one.
    fn retry_after_ms(&self) -> Option<u64> {
        match self {
            ApiError::RateLimited { retry_after_ms }
            | ApiError::Overloaded { retry_after_ms } => Some(*retry_after_ms),
            _ => None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = self.status_and_code();

        let mut headers = HeaderMap::new();
        if let Some(ms) = self.retry_after_ms() {
            // Both: SDK reads either.
            headers.insert(RETRY_AFTER_MS, HeaderValue::from(ms));
            let secs = (ms + 999) / 1000;
            if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
                headers.insert(RETRY_AFTER, v);
            }
        }
        // WWW-Authenticate hint for 401 (the SDK doesn't strictly need it
        // but it's the correct HTTP semantic).
        if matches!(self, ApiError::Unauthorized) {
            if let Ok(v) = HeaderValue::from_str("Bearer") {
                headers.insert(axum::http::header::WWW_AUTHENTICATE, v);
            }
        }

        let body = Json(json!({
            "error": {
                "code": code,
                "message": self.to_string(),
            }
        }));

        (status, headers, body).into_response()
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        ApiError::BadJson(e.to_string())
    }
}

/// Convenience: build a RateLimited with a Duration.
impl From<Duration> for ApiError {
    fn from(d: Duration) -> Self {
        ApiError::RateLimited {
            retry_after_ms: d.as_millis() as u64,
        }
    }
}