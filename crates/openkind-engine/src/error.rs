//! Errors and Result alias for engine operations and dispatch.

use openkind_core::ValidationError;
use thiserror::Error;

/// Errors that can occur during engine execution or model dispatch.
///
/// These errors are translated into corresponding HTTP/gRPC status codes by the API layer.
#[derive(Debug, Error)]
pub enum EngineError {
    /// Request body failed schema validation. Mapped to HTTP 422 Unprocessable Entity.
    #[error("invalid request: {0}")]
    Invalid(#[from] ValidationError),

    /// Requested model alias is not registered. Mapped to HTTP 404 Not Found.
    #[error("no backend registered for model `{0}`")]
    UnknownModel(String),

    /// Request is valid Jev but unsupported by the selected backend profile. Mapped to HTTP 422.
    #[error("backend `{backend}` does not support this request: {message}")]
    Unsupported {
        /// Identifier of the selected backend.
        backend: String,
        /// Contract limitation that rejected the request.
        message: String,
    },

    /// Backend admission queue is full. Mapped to HTTP 529 / gRPC unavailable.
    #[error("backend `{backend}` is overloaded; retry after {retry_after_ms} ms")]
    Overloaded {
        /// Identifier of the selected backend.
        backend: String,
        /// Suggested caller backoff.
        retry_after_ms: u64,
    },

    /// Underlying backend driver encountered an internal execution failure. Mapped to HTTP 500.
    #[error("backend `{backend}` failed: {message}")]
    Backend {
        /// Identifier of the failing backend.
        backend: String,
        /// Descriptive failure message.
        message: String,
    },
}

/// Specialized Result alias for engine operations returning an [`EngineError`].
pub type EngineResult<T> = Result<T, EngineError>;
