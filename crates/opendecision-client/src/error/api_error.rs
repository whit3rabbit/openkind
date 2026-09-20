//! Definition of `ApiError` and `ApiErrorKind` taxonomies.

use std::time::Duration;

use serde_json::Value;

use super::envelope::{extract_code, extract_message, MAX_ERROR_BODY_LENGTH};

/// A single unsuccessful HTTP response, with everything a caller needs to
/// react: status, parsed error envelope, request id for support tickets, and
/// the retry delay the server asked for.
#[derive(Debug, Clone)]
pub struct ApiError {
    /// HTTP status code (e.g. `429`, `529`).
    pub status: u16,
    /// Machine-readable `code` from the error envelope (`rate_limited`, ...).
    pub code: Option<String>,
    /// Human-readable message from the error envelope, if any.
    pub message: Option<String>,
    /// The parsed JSON error body, when the body was valid JSON.
    pub body: Option<Value>,
    /// The `x-typesafe-request-id` response header, for support requests.
    pub request_id: Option<String>,
    /// Delay the server asked the client to wait via `retry-after-ms` /
    /// `Retry-After`, if sent.
    pub retry_after: Option<Duration>,
    /// `METHOD path` of the failed request, e.g. `POST /v1/systemone`.
    pub endpoint: Option<String>,
}

impl ApiError {
    pub(crate) fn from_response(
        status: u16,
        request_id: Option<String>,
        retry_after: Option<Duration>,
        body: &[u8],
        endpoint: String,
    ) -> Self {
        let parsed: Option<Value> = if body.is_empty() {
            None
        } else {
            serde_json::from_slice(body).ok()
        };
        let message = extract_message(parsed.as_ref()).or_else(|| {
            if body.is_empty() {
                return None;
            }
            // Fall back to a truncated slice of whatever the body was.
            let raw = String::from_utf8_lossy(body);
            let mut text = raw.chars().take(MAX_ERROR_BODY_LENGTH).collect::<String>();
            if raw.chars().count() > MAX_ERROR_BODY_LENGTH {
                text.push('…');
            }
            Some(text)
        });
        Self {
            status,
            code: extract_code(parsed.as_ref()),
            message,
            body: parsed,
            request_id,
            retry_after,
            endpoint: Some(endpoint),
        }
    }

    /// Classify this error by HTTP status. See [`ApiErrorKind`].
    pub fn kind(&self) -> ApiErrorKind {
        ApiErrorKind::from_status(self.status)
    }

    /// True when this is 429 Too Many Requests.
    pub fn is_rate_limit(&self) -> bool {
        self.status == 429
    }

    /// True when this is 529 Overloaded (the opendecision/TypeSafe overload status).
    pub fn is_overloaded(&self) -> bool {
        self.status == 529
    }

    /// True for 429 and 529 — the statuses that mean "retry later".
    pub fn is_retryable_later(&self) -> bool {
        self.is_rate_limit() || self.is_overloaded()
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.status)?;
        if let Some(code) = &self.code {
            write!(f, " {code}")?;
        }
        if let Some(message) = &self.message {
            write!(f, ": {message}")?;
        }
        if let Some(endpoint) = &self.endpoint {
            write!(f, " ({endpoint})")?;
        }
        if let Some(request_id) = &self.request_id {
            write!(f, " (request_id={request_id})")?;
        }
        Ok(())
    }
}

impl std::error::Error for ApiError {}

/// Coarse classification of an [`ApiError`] by status, mirroring the Python
/// SDK exception subclasses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApiErrorKind {
    /// 400 — request body was not valid JSON (`bad_json`).
    BadRequest,
    /// 401 — missing or invalid API key (`unauthorized`).
    Authentication,
    /// 403 — key valid but not permitted for this resource
    /// (mirrors the Python SDK's `TypeSafePermissionDeniedError`).
    PermissionDenied,
    /// 404 — unknown model alias (`unknown_model`).
    NotFound,
    /// 413 — request payload exceeded the server limit (`payload_too_large`).
    PayloadTooLarge,
    /// 422 — body failed schema validation (`invalid_body`).
    UnprocessableEntity,
    /// 429 — rate limited (`rate_limited`); carries [`ApiError::retry_after`].
    RateLimit,
    /// 529 — server overloaded (`overloaded`); carries [`ApiError::retry_after`].
    Overloaded,
    /// 5xx — engine/backend failure (`backend_error`, `internal_error`).
    InternalServer,
    /// Any other status.
    Other,
}

impl ApiErrorKind {
    /// Convert an HTTP status code into an [`ApiErrorKind`].
    pub fn from_status(status: u16) -> Self {
        match status {
            400 => Self::BadRequest,
            401 => Self::Authentication,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            413 => Self::PayloadTooLarge,
            422 => Self::UnprocessableEntity,
            429 => Self::RateLimit,
            529 => Self::Overloaded,
            s if (500..600).contains(&s) => Self::InternalServer,
            _ => Self::Other,
        }
    }
}
