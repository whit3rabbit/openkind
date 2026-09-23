//! Error taxonomy and header parsing for the client, mirroring the TypeSafe
//! Python SDK exception hierarchy:
//!
//! | Rust                              | Python SDK                        | HTTP | Server `code`        |
//! |-----------------------------------|-----------------------------------|------|----------------------|
//! | [`ApiErrorKind::BadRequest`]      | `TypeSafeBadRequestError`         | 400  | `bad_json`           |
//! | [`ApiErrorKind::Authentication`]  | `TypeSafeAuthenticationError`     | 401  | `unauthorized`       |
//! | [`ApiErrorKind::NotFound`]        | `TypeSafeNotFoundError`           | 404  | `unknown_model`      |
//! | [`ApiErrorKind::PayloadTooLarge`] | —                                 | 413  | `payload_too_large`  |
//! | [`ApiErrorKind::UnprocessableEntity`] | `TypeSafeUnprocessableEntityError` | 422 | `invalid_body`   |
//! | [`ApiErrorKind::RateLimit`]       | `TypeSafeRateLimitError`          | 429  | `rate_limited`       |
//! | [`ApiErrorKind::Overloaded`]      | `TypeSafeAPIOverloadedError`      | 529  | `overloaded`         |
//! | [`ApiErrorKind::InternalServer`]  | `TypeSafeInternalServerError`     | 5xx  | `backend_error` / `internal_error` |
//!
//! 429/529 responses carry `retry-after-ms` (milliseconds, preferred) and
//! `Retry-After` (seconds, or an HTTP-date) headers; both are parsed into
//! [`ApiError::retry_after`] so [`crate::RetryPolicy`] can back off exactly
//! as long as the server asks.

use std::time::Duration;

mod api_error;
mod envelope;
mod retry_after;

#[cfg(test)]
mod tests;

pub use api_error::{ApiError, ApiErrorKind};
pub use retry_after::parse_retry_after;
pub(crate) use retry_after::parse_retry_after_with;
pub(crate) use retry_after::REQUEST_ID_HEADER;

/// All client failures. [`Error::Api`] wraps the unsuccessful-response case;
/// the other variants cover transport, timeout, decoding, and configuration
/// problems.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The server responded with a non-success status. See [`ApiError`].
    ///
    /// (Boxed to keep `Result<T, Error>` small; deref coercion makes the
    /// inner [`ApiError`] fields directly accessible.)
    #[error("{0}")]
    Api(#[from] Box<ApiError>),

    /// The request could not reach, or the response could not be read from,
    /// the server.
    #[error("connection error: {0}")]
    Connection(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// The request exceeded its per-attempt timeout.
    #[error("request timed out after {timeout:?}")]
    Timeout {
        /// The timeout that was in effect for the failed attempt.
        timeout: Duration,
    },

    /// A response body exceeded the client's fixed memory-safety limit.
    #[error("response body exceeded the {limit}-byte limit")]
    ResponseTooLarge {
        /// Maximum response-body size accepted by the client.
        limit: usize,
    },

    /// A 2xx response body could not be decoded as the expected type.
    #[error("failed to decode response body (status {status}): {source}")]
    Decode {
        /// HTTP status of the undecodable response.
        status: u16,
        /// The first 200 chars of the raw body, to aid debugging.
        body_excerpt: String,
        /// The underlying JSON error.
        #[source]
        source: serde_json::Error,
    },

    /// The client was constructed with invalid settings (missing API key,
    /// bad base URL, non-positive timeout, ...).
    #[error("invalid client configuration: {0}")]
    Config(String),
}

impl Error {
    /// HTTP status, when the error originated from a server response.
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::Api(api) => Some(api.status),
            _ => None,
        }
    }

    /// The server-requested retry delay, when present.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Error::Api(api) => api.retry_after,
            _ => None,
        }
    }

    /// The `x-typesafe-request-id` of the failed request, when present.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Error::Api(api) => api.request_id.as_deref(),
            _ => None,
        }
    }

    /// The error kind classification, when this is an API error.
    pub fn kind(&self) -> Option<ApiErrorKind> {
        match self {
            Error::Api(api) => Some(api.kind()),
            _ => None,
        }
    }
}
