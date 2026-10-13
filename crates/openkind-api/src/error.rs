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
use openkind_engine::EngineError;
use serde_json::json;
use std::time::Duration;

const RETRY_AFTER_MS: HeaderName = HeaderName::from_static("retry-after-ms");
const RETRY_AFTER: HeaderName = HeaderName::from_static("retry-after");

/// API transport and protocol errors.
///
/// Mapped to canonical HTTP status codes and JSON error envelopes matching the Python SDK taxonomy.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// Request failed validation or syntactic constraints (HTTP 422 Unprocessable Entity, `invalid_body`).
    #[error("invalid request body: {0}")]
    InvalidBody(String),

    /// Payload contains malformed JSON syntax (HTTP 400 Bad Request, `bad_json`).
    #[error("invalid JSON: {0}")]
    BadJson(String),

    /// Request payload exceeds maximum allowed size (HTTP 413 Payload Too Large, `payload_too_large`).
    #[error("payload too large: {0}")]
    PayloadTooLarge(String),

    /// Missing or invalid Bearer authentication token (HTTP 401 Unauthorized, `unauthorized`).
    #[error("missing or invalid API key")]
    Unauthorized,

    /// Request exceeded rate limits (HTTP 429 Too Many Requests, `rate_limited`).
    #[error("rate limited; retry after {retry_after_ms} ms")]
    RateLimited {
        /// Suggested backoff period in milliseconds before retrying.
        retry_after_ms: u64,
    },

    /// Server is temporarily overloaded (HTTP 529 API Overloaded, `overloaded`).
    #[error("server overloaded; retry after {retry_after_ms} ms")]
    Overloaded {
        /// Suggested backoff period in milliseconds before retrying.
        retry_after_ms: u64,
    },

    /// Error propagated from underlying decision engine dispatch.
    #[error("engine error: {0}")]
    Engine(#[from] EngineError),

    /// Upstream (proxied) service failed or was unreachable
    /// (HTTP 502 Bad Gateway, `bad_gateway`).
    #[error("bad gateway: {0}")]
    BadGateway(String),

    /// Unexpected internal server error (HTTP 500 Internal Server Error, `internal_error`).
    #[error("internal error: {0}")]
    Internal(String),
}

/// Transport-neutral classification of an error: the one table both the HTTP
/// and gRPC layers read, so the two cannot drift apart. Message text is
/// deliberately not part of it (the transports word messages differently).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Classified {
    /// Stable machine-readable code in the HTTP JSON envelope.
    pub(crate) code: &'static str,
    /// HTTP status.
    pub(crate) http: StatusCode,
    /// gRPC status code. Variants that never occur over gRPC (for example
    /// `BadGateway`) carry the closest equivalent.
    pub(crate) grpc: tonic::Code,
    /// Suggested backoff in milliseconds, emitted on both transports.
    pub(crate) retry_after_ms: Option<u64>,
}

impl Classified {
    fn new(code: &'static str, http: StatusCode, grpc: tonic::Code) -> Self {
        Self {
            code,
            http,
            grpc,
            retry_after_ms: None,
        }
    }

    fn with_retry(mut self, retry_after_ms: u64) -> Self {
        self.retry_after_ms = Some(retry_after_ms);
        self
    }
}

/// HTTP 529 (non-standard "API overloaded", part of the SDK taxonomy).
fn status_overloaded() -> StatusCode {
    StatusCode::from_u16(529).expect("529 is a valid status code")
}

/// `Retry-After` seconds for a millisecond backoff, rounded up so callers
/// never retry early.
pub(crate) fn retry_after_secs(ms: u64) -> u64 {
    ms.div_ceil(1000)
}

pub(crate) fn classify_engine(e: &EngineError) -> Classified {
    use tonic::Code;
    match e {
        EngineError::Invalid(_) | EngineError::Unsupported { .. } => Classified::new(
            "invalid_body",
            StatusCode::UNPROCESSABLE_ENTITY,
            Code::InvalidArgument,
        ),
        EngineError::UnknownModel(_) => {
            Classified::new("unknown_model", StatusCode::NOT_FOUND, Code::NotFound)
        }
        EngineError::Overloaded { retry_after_ms, .. } => {
            Classified::new("overloaded", status_overloaded(), Code::Unavailable)
                .with_retry(*retry_after_ms)
        }
        EngineError::DeadlineExceeded { .. } => Classified::new(
            "deadline_exceeded",
            StatusCode::GATEWAY_TIMEOUT,
            Code::DeadlineExceeded,
        ),
        EngineError::Backend { .. } | EngineError::BackendValidation { .. } => Classified::new(
            "backend_error",
            StatusCode::INTERNAL_SERVER_ERROR,
            Code::Internal,
        ),
    }
}

impl ApiError {
    pub(crate) fn classify(&self) -> Classified {
        use tonic::Code;
        match self {
            ApiError::InvalidBody(_) => Classified::new(
                "invalid_body",
                StatusCode::UNPROCESSABLE_ENTITY,
                Code::InvalidArgument,
            ),
            ApiError::BadJson(_) => {
                Classified::new("bad_json", StatusCode::BAD_REQUEST, Code::InvalidArgument)
            }
            ApiError::PayloadTooLarge(_) => Classified::new(
                "payload_too_large",
                StatusCode::PAYLOAD_TOO_LARGE,
                Code::ResourceExhausted,
            ),
            ApiError::Unauthorized => Classified::new(
                "unauthorized",
                StatusCode::UNAUTHORIZED,
                Code::Unauthenticated,
            ),
            ApiError::RateLimited { retry_after_ms } => Classified::new(
                "rate_limited",
                StatusCode::TOO_MANY_REQUESTS,
                Code::ResourceExhausted,
            )
            .with_retry(*retry_after_ms),
            ApiError::Overloaded { retry_after_ms } => {
                Classified::new("overloaded", status_overloaded(), Code::Unavailable)
                    .with_retry(*retry_after_ms)
            }
            ApiError::Engine(e) => classify_engine(e),
            ApiError::BadGateway(_) => {
                Classified::new("bad_gateway", StatusCode::BAD_GATEWAY, Code::Unavailable)
            }
            ApiError::Internal(_) => Classified::new(
                "internal_error",
                StatusCode::INTERNAL_SERVER_ERROR,
                Code::Internal,
            ),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let Classified {
            code,
            http: status,
            retry_after_ms,
            ..
        } = self.classify();

        let mut headers = HeaderMap::new();
        if let Some(ms) = retry_after_ms {
            // Both: SDK reads either.
            headers.insert(RETRY_AFTER_MS, HeaderValue::from(ms));
            headers.insert(RETRY_AFTER, HeaderValue::from(retry_after_secs(ms)));
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

/// Body-extraction failures keep their distinct statuses: oversize (413),
/// syntax (400), and everything else, including schema mismatches (422).
impl From<axum::extract::rejection::JsonRejection> for ApiError {
    fn from(rejection: axum::extract::rejection::JsonRejection) -> Self {
        use axum::extract::rejection::JsonRejection;
        match rejection {
            JsonRejection::BytesRejection(e) => ApiError::PayloadTooLarge(e.to_string()),
            JsonRejection::JsonSyntaxError(e) => ApiError::BadJson(e.to_string()),
            JsonRejection::JsonDataError(e) => ApiError::InvalidBody(e.to_string()),
            other => ApiError::InvalidBody(other.to_string()),
        }
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
            retry_after_ms: openkind_core::duration_millis_saturating(d),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use http_body_util::BodyExt;
    use openkind_core::ValidationError;

    async fn extract_body_json(resp: Response) -> (StatusCode, HeaderMap, serde_json::Value) {
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, headers, val)
    }

    #[tokio::test]
    async fn rate_limited_error_into_response() {
        let err = ApiError::RateLimited {
            retry_after_ms: 1500,
        };
        let (status, headers, body) = extract_body_json(err.into_response()).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(headers.get("retry-after-ms").unwrap(), "1500");
        assert_eq!(headers.get("retry-after").unwrap(), "2"); // 1500.div_ceil(1000) = 2
        assert_eq!(body["error"]["code"], "rate_limited");
    }

    #[tokio::test]
    async fn bad_gateway_error_into_response() {
        let err = ApiError::BadGateway("upstream unreachable".into());
        let (status, _headers, body) = extract_body_json(err.into_response()).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["error"]["code"], "bad_gateway");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("upstream unreachable"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn overloaded_error_into_response() {
        let err = ApiError::Overloaded {
            retry_after_ms: 500,
        };
        let (status, headers, body) = extract_body_json(err.into_response()).await;
        assert_eq!(status, StatusCode::from_u16(529).unwrap());
        assert_eq!(headers.get("retry-after-ms").unwrap(), "500");
        assert_eq!(headers.get("retry-after").unwrap(), "1");
        assert_eq!(body["error"]["code"], "overloaded");
    }

    #[tokio::test]
    async fn unauthorized_error_into_response() {
        let err = ApiError::Unauthorized;
        let (status, headers, body) = extract_body_json(err.into_response()).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(headers.get("www-authenticate").unwrap(), "Bearer");
        assert_eq!(body["error"]["code"], "unauthorized");
    }

    #[tokio::test]
    async fn internal_error_into_response() {
        let err = ApiError::Internal("db crashed".into());
        let (status, _, body) = extract_body_json(err.into_response()).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["code"], "internal_error");
    }

    #[tokio::test]
    async fn bad_json_and_model_not_found_into_response() {
        let err_json = ApiError::BadJson("syntax error".into());
        let (status, _, body) = extract_body_json(err_json.into_response()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "bad_json");

        let err_model = ApiError::Engine(EngineError::UnknownModel("gpt-5".into()));
        let (status, _, body) = extract_body_json(err_model.into_response()).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "unknown_model");
    }

    #[tokio::test]
    async fn validation_and_backend_errors_into_response() {
        let err_val = ApiError::Engine(EngineError::Invalid(ValidationError::NoQuestions));
        let (status, _, body) = extract_body_json(err_val.into_response()).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["error"]["code"], "invalid_body");

        let err_validation = ApiError::Engine(EngineError::BackendValidation {
            backend: "mock".into(),
            source: ValidationError::MissingAnswer("q".into()),
        });
        let (status, _, body) = extract_body_json(err_validation.into_response()).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["code"], "backend_error");

        let err_backend = ApiError::Engine(EngineError::Backend {
            backend: "mock".into(),
            message: "simulated failure".into(),
        });
        let (status, _, body) = extract_body_json(err_backend.into_response()).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["code"], "backend_error");
    }

    #[tokio::test]
    async fn unsupported_and_engine_overload_keep_transport_semantics() {
        let unsupported = ApiError::Engine(EngineError::Unsupported {
            backend: "native".into(),
            message: "explicit semantic none required".into(),
        });
        let (status, _, body) = extract_body_json(unsupported.into_response()).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["error"]["code"], "invalid_body");

        let overloaded = ApiError::Engine(EngineError::Overloaded {
            backend: "native".into(),
            retry_after_ms: 750,
        });
        let (status, headers, body) = extract_body_json(overloaded.into_response()).await;
        assert_eq!(status, StatusCode::from_u16(529).unwrap());
        assert_eq!(headers["retry-after-ms"], "750");
        assert_eq!(body["error"]["code"], "overloaded");

        let deadline = ApiError::Engine(EngineError::DeadlineExceeded {
            backend: "native".into(),
            timeout_ms: 30_000,
        });
        let (status, _, body) = extract_body_json(deadline.into_response()).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(body["error"]["code"], "deadline_exceeded");
    }

    #[test]
    fn duration_conversion_to_rate_limited() {
        let err: ApiError = Duration::from_millis(2500).into();
        match err {
            ApiError::RateLimited { retry_after_ms } => assert_eq!(retry_after_ms, 2500),
            _ => panic!("expected RateLimited"),
        }
    }

    #[test]
    fn duration_conversion_saturates_instead_of_wrapping() {
        let err: ApiError = Duration::from_secs(u64::MAX).into();
        assert!(matches!(
            err,
            ApiError::RateLimited {
                retry_after_ms: u64::MAX
            }
        ));
    }
}
