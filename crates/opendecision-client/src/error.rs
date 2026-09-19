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

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::HeaderMap;
use serde_json::Value;

pub(crate) const REQUEST_ID_HEADER: &str = "x-typesafe-request-id";
pub(crate) const RETRY_AFTER_HEADER: &str = "retry-after";
pub(crate) const RETRY_AFTER_MS_HEADER: &str = "retry-after-ms";

/// Matches the Python SDK's `MAX_ERROR_BODY_LENGTH`: raw bodies longer than
/// this are truncated in error messages.
const MAX_ERROR_BODY_LENGTH: usize = 200;

/// Parse the retry delay the server asked for, in order of preference:
///
/// 1. `retry-after-ms` — milliseconds (the opendecision server sends this).
/// 2. `retry-after` — seconds (integer or decimal), or an RFC 7231 HTTP-date
///    such as `Sun, 06 Nov 1994 08:49:37 GMT`.
///
/// Returns `None` when neither header is present, or their values are
/// negative, non-finite, or unparseable.
pub fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    if let Some(raw) = header_str(headers, RETRY_AFTER_MS_HEADER) {
        if let Some(delay) = finite_millis(raw, 1.0) {
            return Some(delay);
        }
    }
    if let Some(raw) = header_str(headers, RETRY_AFTER_HEADER) {
        if let Some(delay) = finite_millis(raw, 1000.0) {
            return Some(delay);
        }
        if let Some(delay) = parse_http_date(raw) {
            return Some(delay);
        }
    }
    None
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
}

/// Parse a header value as a finite, non-negative number and scale it to
/// milliseconds (`multiplier` is the value's unit in milliseconds). An empty
/// value parses as zero, matching the Python SDK (`float(raw or "0")`); any
/// other unparseable value returns `None`.
fn finite_millis(raw: &str, multiplier: f64) -> Option<Duration> {
    let value: f64 = if raw.is_empty() {
        0.0
    } else {
        raw.parse().ok()?
    };
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let millis = value * multiplier;
    if !millis.is_finite() {
        return None;
    }
    Some(Duration::from_millis(millis as u64))
}

/// Parse an IMF-fixdate HTTP-date (`Sun, 06 Nov 1994 08:49:37 GMT`) into a
/// delay from now. Obsolete RFC 850 / asctime forms are not supported; the
/// opendecision server never emits them.
fn parse_http_date(raw: &str) -> Option<Duration> {
    // Strip the leading weekday name ("Sun, ") if present.
    let rest = raw.split_once(", ").map(|(_, r)| r).unwrap_or(raw);
    let mut fields = rest.split_ascii_whitespace();
    let day: i64 = fields.next()?.parse().ok()?;
    let month = month_number(fields.next()?)?;
    let year: i64 = fields.next()?.parse().ok()?;
    let mut clock = fields.next()?.split(':');
    let hour: i64 = clock.next()?.parse().ok()?;
    let minute: i64 = clock.next()?.parse().ok()?;
    let second: i64 = clock.next()?.parse().ok()?;
    if !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let target = days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
    // Dates in the past mean "retry now" (clamped to zero), matching the
    // Python SDK's `max(0.0, delta)`.
    let delta = (target - now).max(0);
    Some(Duration::from_secs(delta as u64))
}

fn month_number(name: &str) -> Option<i64> {
    let month = match name {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    Some(month)
}

/// Days since 1970-01-01 from a civil date (Howard Hinnant's algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Extract the machine-readable error code the server puts in
/// `{"error": {"code": ...}}`, with lenient fallbacks for proxies that
/// reshape the envelope.
pub(crate) fn extract_code(body: Option<&Value>) -> Option<String> {
    let body = body?;
    if let Some(code) = str_field(&body["error"], "code") {
        return Some(code);
    }
    if let Some(code) = str_scalar(&body["error"]) {
        return Some(code);
    }
    str_field(body, "code")
}

/// Extract a human-readable message from an error body, accepting the
/// server's `error.message` shape plus the lenient fallbacks the Python SDK
/// tolerates (`message`, `detail` as string, object with `message`, or
/// FastAPI-style detail list).
pub(crate) fn extract_message(body: Option<&Value>) -> Option<String> {
    let body = body?;
    if let Some(error) = str_scalar(&body["error"]) {
        return Some(error);
    }
    if let Some(message) = str_field(&body["error"], "message") {
        return Some(message);
    }
    if let Some(message) = str_field(body, "message") {
        return Some(message);
    }
    if let Some(detail) = str_scalar(&body["detail"]) {
        return Some(detail);
    }
    if let Some(message) = str_field(&body["detail"], "message") {
        return Some(message);
    }
    if let Some(detail) = body["detail"].as_array() {
        // FastAPI-style validation errors: [{"loc": [...], "msg": "..."}, ...]
        let parts: Vec<String> = detail
            .iter()
            .filter_map(|entry| {
                let msg = str_field(entry, "msg")?;
                Some(match entry["loc"].as_array() {
                    Some(loc) => {
                        let path = loc
                            .iter()
                            .filter(|&seg| seg != "body")
                            .map(|seg| match seg {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .collect::<Vec<_>>()
                            .join(".");
                        if path.is_empty() {
                            msg
                        } else {
                            format!("{path}: {msg}")
                        }
                    }
                    None => msg,
                })
            })
            .collect();
        if !parts.is_empty() {
            return Some(parts.join("; "));
        }
    }
    None
}

fn str_scalar(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

fn str_field(value: &Value, field: &str) -> Option<String> {
    value[field].as_str().map(str::to_owned)
}

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
    fn from_status(status: u16) -> Self {
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
    Connection(#[source] reqwest::Error),

    /// The request exceeded its per-attempt timeout.
    #[error("request timed out after {timeout:?}")]
    Timeout {
        /// The timeout that was in effect for the failed attempt.
        timeout: Duration,
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

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (k, v) in pairs {
            map.insert(
                reqwest::header::HeaderName::from_lowercase(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        map
    }

    #[test]
    fn retry_after_ms_preferred_over_retry_after() {
        let h = headers(&[("retry-after", "99"), ("retry-after-ms", "250")]);
        assert_eq!(parse_retry_after(&h), Some(Duration::from_millis(250)));
    }

    #[test]
    fn retry_after_seconds_parsing() {
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "2")])),
            Some(Duration::from_secs(2))
        );
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "0.5")])),
            Some(Duration::from_millis(500))
        );
    }

    #[test]
    fn retry_after_invalid_values_ignored() {
        assert_eq!(parse_retry_after(&headers(&[("retry-after", "-1")])), None);
        assert_eq!(parse_retry_after(&headers(&[("retry-after", "abc")])), None);
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after-ms", "NaN")])),
            None
        );
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after-ms", "-5")])),
            None
        );
        assert_eq!(parse_retry_after(&headers(&[])), None);
    }

    /// Ported matrix from the Python SDK's `test_parse_retry_after`:
    /// empty values are zero, invalid `retry-after-ms` falls through to
    /// `Retry-After`, and overflow yields `None`.
    #[test]
    fn retry_after_python_sdk_parity_matrix() {
        // {"Retry-After": ""} → 0 (Python: float("" or "0")).
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "")])),
            Some(Duration::ZERO)
        );
        // {"retry-after-ms": ""} → 0 as well.
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after-ms", "")])),
            Some(Duration::ZERO)
        );
        // {"retry-after-ms": "NaN", "Retry-After": "1.5"} → 1500 ms.
        assert_eq!(
            parse_retry_after(&headers(&[
                ("retry-after-ms", "NaN"),
                ("retry-after", "1.5")
            ])),
            Some(Duration::from_millis(1500))
        );
        // {"retry-after-ms": "-1", "Retry-After": "2"} → 2000 ms.
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after-ms", "-1"), ("retry-after", "2")])),
            Some(Duration::from_secs(2))
        );
        // {"retry-after-ms": "inf"} → None.
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after-ms", "inf")])),
            None
        );
        // {"retry-after-ms": "bad", "Retry-After": "2"} → 2000 ms.
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after-ms", "bad"), ("retry-after", "2")])),
            Some(Duration::from_secs(2))
        );
        // {"Retry-After": "1e308"} → overflow to non-finite ms → None.
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "1e308")])),
            None
        );
    }

    #[test]
    fn retry_after_http_date_in_past_clamps_to_zero() {
        // Python SDK: max(0.0, delta) — past dates mean "retry immediately".
        let got = parse_retry_after(&headers(&[(
            "retry-after",
            "Sun, 06 Nov 1994 08:49:37 GMT",
        )]));
        assert_eq!(got, Some(Duration::ZERO));
    }

    #[test]
    fn retry_after_http_date() {
        // "expires" header 60s in the future, formatted per RFC 7231.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 60;
        let date = http_date_from_epoch(now);
        let got = parse_retry_after(&headers(&[("retry-after", &date)])).unwrap();
        assert!((50..=60).contains(&got.as_secs()), "got {got:?}");
    }

    fn http_date_from_epoch(secs: u64) -> String {
        // Format `secs` as IMF-fixdate using the civil-from-days inverse.
        let days = (secs / 86_400) as i64;
        let rem = secs % 86_400;
        let (y, m, d) = civil_from_days(days);
        let weekday =
            ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"][(days.rem_euclid(7)) as usize];
        let month = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][(m - 1) as usize];
        format!(
            "{weekday}, {d:02} {month} {y} {:02}:{:02}:{:02} GMT",
            rem / 3600,
            (rem % 3600) / 60,
            rem % 60
        )
    }

    /// Inverse of `days_from_civil`, also Hinnant's.
    fn civil_from_days(z: i64) -> (i64, i64, i64) {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        (if m <= 2 { y + 1 } else { y }, m, d)
    }

    #[test]
    fn extract_code_from_server_envelope() {
        let body = serde_json::json!({"error": {"code": "rate_limited", "message": "slow down"}});
        assert_eq!(extract_code(Some(&body)).as_deref(), Some("rate_limited"));
        assert_eq!(extract_message(Some(&body)).as_deref(), Some("slow down"));
    }

    #[test]
    fn extract_message_lenient_shapes() {
        let plain_error = serde_json::json!({"error": "unauthorized"});
        assert_eq!(
            extract_message(Some(&plain_error)).as_deref(),
            Some("unauthorized")
        );

        let message = serde_json::json!({"message": "boom"});
        assert_eq!(extract_message(Some(&message)).as_deref(), Some("boom"));

        let detail = serde_json::json!({"detail": "nope"});
        assert_eq!(extract_message(Some(&detail)).as_deref(), Some("nope"));

        let fastapi = serde_json::json!({"detail": [
            {"loc": ["body", "state"], "msg": "field required"},
            {"loc": ["body"], "msg": "invalid"},
        ]});
        assert_eq!(
            extract_message(Some(&fastapi)).as_deref(),
            Some("state: field required; invalid")
        );
    }

    #[test]
    fn extract_message_includes_detail_object_message() {
        // Python SDK matrix: {"detail": {"message": "nested detail"}}.
        let nested = serde_json::json!({"detail": {"message": "nested detail"}});
        assert_eq!(
            extract_message(Some(&nested)).as_deref(),
            Some("nested detail")
        );
    }

    #[test]
    fn kind_maps_permission_denied() {
        // Python SDK: TypeSafePermissionDeniedError.
        assert_eq!(
            ApiErrorKind::from_status(403),
            ApiErrorKind::PermissionDenied
        );
    }

    #[test]
    fn api_error_display_includes_context() {
        let body = serde_json::json!({"error": {"code": "rate_limited", "message": "too many"}});
        let err = ApiError::from_response(
            429,
            Some("req-123".into()),
            Some(Duration::from_millis(1500)),
            serde_json::to_vec(&body).unwrap().as_slice(),
            "POST /v1/systemone".into(),
        );
        let displayed = err.to_string();
        assert!(
            displayed.contains("429 rate_limited: too many"),
            "{displayed}"
        );
        assert!(displayed.contains("POST /v1/systemone"), "{displayed}");
        assert!(displayed.contains("request_id=req-123"), "{displayed}");
        assert_eq!(err.retry_after, Some(Duration::from_millis(1500)));
        assert_eq!(err.kind(), ApiErrorKind::RateLimit);
        assert!(err.is_rate_limit());
    }

    #[test]
    fn api_error_non_json_body_falls_back_to_text() {
        let err = ApiError::from_response(
            502,
            None,
            None,
            b"<html>Bad Gateway</html>",
            "POST /v1/systemone".into(),
        );
        assert_eq!(err.kind(), ApiErrorKind::InternalServer);
        assert_eq!(err.code, None);
        assert_eq!(err.message.as_deref(), Some("<html>Bad Gateway</html>"));
    }

    #[test]
    fn api_error_empty_body() {
        let err = ApiError::from_response(500, None, None, b"", "GET /v1/models".into());
        assert_eq!(err.kind(), ApiErrorKind::InternalServer);
        assert_eq!(err.message, None);
        assert_eq!(err.body, None);
    }

    #[test]
    fn kind_covers_server_status_matrix() {
        assert_eq!(ApiErrorKind::from_status(400), ApiErrorKind::BadRequest);
        assert_eq!(ApiErrorKind::from_status(401), ApiErrorKind::Authentication);
        assert_eq!(ApiErrorKind::from_status(404), ApiErrorKind::NotFound);
        assert_eq!(
            ApiErrorKind::from_status(413),
            ApiErrorKind::PayloadTooLarge
        );
        assert_eq!(
            ApiErrorKind::from_status(422),
            ApiErrorKind::UnprocessableEntity
        );
        assert_eq!(ApiErrorKind::from_status(429), ApiErrorKind::RateLimit);
        assert_eq!(ApiErrorKind::from_status(529), ApiErrorKind::Overloaded);
        assert_eq!(ApiErrorKind::from_status(500), ApiErrorKind::InternalServer);
        assert_eq!(ApiErrorKind::from_status(503), ApiErrorKind::InternalServer);
        assert_eq!(ApiErrorKind::from_status(418), ApiErrorKind::Other);
    }
}
