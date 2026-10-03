//! Unit tests for openkind-client error handling and header parsing.

use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue};

use super::api_error::{ApiError, ApiErrorKind};
use super::envelope::{extract_code, extract_message};
use super::retry_after::parse_retry_after;

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
    let integer = headers(&[("retry-after", "3")]);
    assert_eq!(parse_retry_after(&integer), Some(Duration::from_secs(3)));

    let decimal = headers(&[("retry-after", "1.5")]);
    assert_eq!(
        parse_retry_after(&decimal),
        Some(Duration::from_millis(1500))
    );
}

#[test]
fn retry_after_http_date() {
    // A fixed date well into the future so this test remains deterministic.
    let h = headers(&[("retry-after", "Fri, 31 Dec 2099 23:59:59 GMT")]);
    let delay = parse_retry_after(&h).expect("valid future date should parse");
    assert!(delay > Duration::from_secs(86_400));
}

#[test]
fn retry_after_http_date_in_past_clamps_to_zero() {
    let h = headers(&[("retry-after", "Sun, 06 Nov 1994 08:49:37 GMT")]);
    assert_eq!(parse_retry_after(&h), Some(Duration::ZERO));
}

#[test]
fn retry_after_invalid_values_ignored() {
    for bad in ["not-a-number", "-5", "NaN", "inf", ""] {
        let h = headers(&[("retry-after", bad)]);
        assert_eq!(
            parse_retry_after(&h),
            None,
            "expected bad value {bad:?} to return None"
        );
    }
}

#[test]
fn malformed_http_dates_cannot_overflow_or_normalize_invalid_fields() {
    for date in [
        "Sun, 06 Nov 9223372036854775807 08:49:37 GMT",
        "Sun, 06 Jan -9223372036854775808 08:49:37 GMT",
        "Sun, 31 Feb 2026 08:49:37 GMT",
        "Sun, 06 Nov 2026 -1:49:37 GMT",
        "Sun, 06 Nov 2026 08:49:37 EST",
        "Sun, 06 Nov 2026 08:49:37:12 GMT",
    ] {
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", date)])),
            None,
            "{date}"
        );
    }
}

/// Port of the Python SDK's `test_retry_after` parameter matrix:
/// [None, "0", "1.5", "invalid", "-1", "Fri, 31 Dec 2099..."]
#[test]
fn retry_after_python_sdk_parity_matrix() {
    assert_eq!(parse_retry_after(&headers(&[])), None);
    assert_eq!(
        parse_retry_after(&headers(&[("retry-after", "0")])),
        Some(Duration::ZERO)
    );
    assert_eq!(
        parse_retry_after(&headers(&[("retry-after", "1.5")])),
        Some(Duration::from_millis(1500))
    );
    assert_eq!(
        parse_retry_after(&headers(&[("retry-after", "invalid")])),
        None
    );
    assert_eq!(parse_retry_after(&headers(&[("retry-after", "-1")])), None);
}

#[test]
fn extract_code_from_server_envelope() {
    let body: serde_json::Value =
        serde_json::json!({"error": {"code": "rate_limited", "message": "calm down"}});
    assert_eq!(extract_code(Some(&body)), Some("rate_limited".into()));

    let flat: serde_json::Value = serde_json::json!({"code": "unknown_model"});
    assert_eq!(extract_code(Some(&flat)), Some("unknown_model".into()));

    let string_error: serde_json::Value = serde_json::json!({"error": "unauthorized"});
    assert_eq!(
        extract_code(Some(&string_error)),
        Some("unauthorized".into())
    );
}

#[test]
fn extract_message_includes_detail_object_message() {
    // Regression test for the Python SDK pattern: detail={"message": "..."}
    let body: serde_json::Value = serde_json::json!({"detail": {"message": "structured problem"}});
    assert_eq!(
        extract_message(Some(&body)),
        Some("structured problem".into())
    );
}

#[test]
fn extract_message_lenient_shapes() {
    let server: serde_json::Value =
        serde_json::json!({"error": {"code": "c", "message": "the message"}});
    assert_eq!(extract_message(Some(&server)), Some("the message".into()));

    let flat_msg: serde_json::Value = serde_json::json!({"message": "flat"});
    assert_eq!(extract_message(Some(&flat_msg)), Some("flat".into()));

    let flat_detail: serde_json::Value = serde_json::json!({"detail": "from detail"});
    assert_eq!(
        extract_message(Some(&detail(&flat_detail))),
        Some("from detail".into())
    );

    let fastapi: serde_json::Value = serde_json::json!({
        "detail": [
            {"loc": ["body", "state"], "msg": "field required"},
            {"loc": ["body", "questions", "0"], "msg": "invalid shape"}
        ]
    });
    assert_eq!(
        extract_message(Some(&fastapi)),
        Some("state: field required; questions.0: invalid shape".into())
    );
}

fn detail(v: &serde_json::Value) -> serde_json::Value {
    v.clone()
}

#[test]
fn api_error_display_includes_context() {
    let err = ApiError {
        status: 429,
        code: Some("rate_limited".into()),
        message: Some("too fast".into()),
        body: None,
        request_id: Some("req-42".into()),
        retry_after: Some(Duration::from_millis(500)),
        endpoint: Some("POST /v1/systemone".into()),
    };
    let rendered = err.to_string();
    assert!(
        rendered.contains("429 rate_limited: too fast"),
        "{rendered}"
    );
    assert!(rendered.contains("(POST /v1/systemone)"), "{rendered}");
    assert!(rendered.contains("(request_id=req-42)"), "{rendered}");
}

#[test]
fn api_error_non_json_body_falls_back_to_text() {
    let raw = b"<!DOCTYPE html><html>502 Bad Gateway</html>";
    let err = ApiError::from_response(502, None, None, raw, "POST /v1/systemone".into());
    assert_eq!(
        err.message,
        Some("<!DOCTYPE html><html>502 Bad Gateway</html>".into())
    );
    assert_eq!(err.kind(), ApiErrorKind::InternalServer);
}

#[test]
fn api_error_empty_body() {
    let err = ApiError::from_response(404, None, None, b"", "GET /v1/models".into());
    assert_eq!(err.message, None);
    assert_eq!(err.code, None);
    assert_eq!(err.kind(), ApiErrorKind::NotFound);
}

#[test]
fn kind_covers_server_status_matrix() {
    assert_eq!(ApiErrorKind::from_status(400), ApiErrorKind::BadRequest);
    assert_eq!(ApiErrorKind::from_status(401), ApiErrorKind::Authentication);
    assert_eq!(
        ApiErrorKind::from_status(403),
        ApiErrorKind::PermissionDenied
    );
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

#[test]
fn kind_maps_permission_denied() {
    let err = ApiError::from_response(403, None, None, b"{}", "GET /v1/models".into());
    assert_eq!(err.kind(), ApiErrorKind::PermissionDenied);
}

#[test]
fn api_error_predicates_classify_retryable_later_statuses() {
    let error = |status| ApiError::from_response(status, None, None, b"{}", "e".into());
    assert!(error(429).is_rate_limit());
    assert!(!error(429).is_overloaded());
    assert!(error(529).is_overloaded());
    assert!(!error(529).is_rate_limit());
    assert!(error(429).is_retryable_later());
    assert!(error(529).is_retryable_later());
    for status in [400, 401, 404, 500, 502, 503] {
        assert!(
            !error(status).is_retryable_later(),
            "{status} is not a retry-later status"
        );
    }
}

#[test]
fn non_json_bodies_truncate_at_two_hundred_chars_with_an_ellipsis() {
    let body = "x".repeat(300).into_bytes();
    let error = ApiError::from_response(500, None, None, &body, "e".into());
    let message = error.message.expect("plain body becomes the message");
    assert_eq!(message.chars().count(), 201, "200 chars plus the ellipsis");
    assert!(message.ends_with('…'));

    let short = ApiError::from_response(500, None, None, b"boom", "e".into());
    assert_eq!(short.message.as_deref(), Some("boom"));
}

#[test]
fn retry_after_ms_falls_through_to_retry_after_when_unparseable() {
    let lookup = |name: &str| match name {
        "retry-after-ms" => Some("garbage"),
        "retry-after" => Some("3"),
        _ => None,
    };
    assert_eq!(
        super::parse_retry_after_with(lookup),
        Some(Duration::from_secs(3))
    );
}
