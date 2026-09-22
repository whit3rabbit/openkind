use std::time::Duration;

use super::*;
use crate::error::{ApiError, ApiErrorKind, Error};

fn api_error(status: u16) -> Error {
    Error::Api(Box::new(ApiError {
        status,
        code: None,
        message: None,
        body: None,
        request_id: None,
        retry_after: None,
        endpoint: None,
    }))
}

fn rate_limited(retry_after: Option<Duration>) -> Error {
    Error::Api(Box::new(ApiError {
        status: 429,
        code: Some("rate_limited".into()),
        message: None,
        body: None,
        request_id: None,
        retry_after,
        endpoint: None,
    }))
}

#[test]
fn defaults_match_python_sdk() {
    let p = RetryPolicy::default();
    assert_eq!(p.max_retries, 2);
    assert_eq!(p.backoff_initial, Duration::from_millis(500));
    assert_eq!(p.backoff_max, Duration::from_secs(5));
    assert!((p.backoff_jitter - 0.25).abs() < f64::EPSILON);
    assert_eq!(p.retry_statuses, vec![408, 429]);
    assert!(p.retry_server_errors);
    assert!(p.respect_retry_after);
    assert_eq!(p.total_timeout, Some(Duration::from_secs(30)));
}

#[test]
fn retryable_status_matrix() {
    let p = RetryPolicy::default();
    // 529 is in the 5xx range and 429/408 are explicit.
    assert!(p.is_retryable(&api_error(429)));
    assert!(p.is_retryable(&api_error(408)));
    assert!(p.is_retryable(&api_error(500)));
    assert!(p.is_retryable(&api_error(529)));
    assert!(p.is_retryable(&api_error(503)));
    // Client errors are not retried.
    assert!(!p.is_retryable(&api_error(400)));
    assert!(!p.is_retryable(&api_error(401)));
    assert!(!p.is_retryable(&api_error(404)));
    assert!(!p.is_retryable(&api_error(422)));
    // Neither are non-API failures. Connection/timeout retryability is
    // covered by the retry_behavior integration tests (reqwest errors
    // cannot be synthesized here).
    assert!(!p.is_retryable(&Error::Config("x".into())));
}

#[test]
fn disabling_rules_disables_retry() {
    let p = RetryPolicy::default()
        .retry_server_errors(false)
        .retry_connection_errors(false)
        .retry_timeout_errors(false)
        .max_retries(0);
    // Still "retryable" per the status rules, but max_retries gates attempts.
    assert!(p.is_retryable(&api_error(503)) == p.retry_server_errors);
    assert!(!p.retry_server_errors);
}

#[test]
fn extra_status_can_be_added() {
    let p = RetryPolicy::default().retry_status(425);
    assert!(p.is_retryable(&api_error(425)));
    assert!(!RetryPolicy::default().is_retryable(&api_error(425)));
}

#[test]
fn backoff_doubles_and_caps() {
    let p = RetryPolicy::new()
        .backoff_initial(Duration::from_millis(100))
        .backoff_max(Duration::from_millis(400))
        .backoff_jitter(0.0);
    assert_eq!(p.backoff_delay(0), Duration::from_millis(100));
    assert_eq!(p.backoff_delay(1), Duration::from_millis(200));
    assert_eq!(p.backoff_delay(2), Duration::from_millis(400));
    assert_eq!(
        p.backoff_delay(30),
        Duration::from_millis(400),
        "caps at max"
    );
}

#[test]
fn backoff_zero_disables_delay() {
    let p = RetryPolicy::new().backoff_initial(Duration::ZERO);
    assert_eq!(p.backoff_delay(5), Duration::ZERO);
}

#[test]
fn jitter_stays_within_fraction() {
    let p = RetryPolicy::new()
        .backoff_initial(Duration::from_secs(1))
        .backoff_max(Duration::from_secs(8));
    for i in 0..5u32 {
        let d = p.backoff_delay(i);
        let base = (1000u64 << i).min(8000);
        assert!(d.as_millis() as u64 >= base - base / 4, "{i} -> {d:?}");
        assert!(d.as_millis() as u64 <= base, "{i} -> {d:?}");
    }
}

#[test]
fn retry_after_takes_precedence() {
    let p = RetryPolicy::new().backoff_initial(Duration::from_secs(10));
    let err = rate_limited(Some(Duration::from_millis(25)));
    assert_eq!(p.delay_for(&err, 0), Duration::from_millis(25));
    // With respect_retry_after off, backoff is used.
    let p2 = RetryPolicy::new()
        .backoff_initial(Duration::from_millis(40))
        .backoff_jitter(0.0)
        .respect_retry_after(false);
    assert_eq!(p2.delay_for(&err, 0), Duration::from_millis(40));
    // No Retry-After on the error → backoff even when respecting it.
    let p3 = RetryPolicy::new()
        .backoff_initial(Duration::from_millis(7))
        .backoff_jitter(0.0);
    assert_eq!(
        p3.delay_for(&rate_limited(None), 0),
        Duration::from_millis(7)
    );
}

#[test]
fn validate_rejects_bad_jitter() {
    assert!(RetryPolicy::new().backoff_jitter(1.5).validate().is_err());
    assert!(RetryPolicy::new().backoff_jitter(-0.1).validate().is_err());
    assert!(RetryPolicy::new()
        .backoff_jitter(f64::NAN)
        .validate()
        .is_err());
    assert!(RetryPolicy::new().backoff_jitter(0.5).validate().is_ok());
}

#[test]
fn kinds_are_reported_through_error() {
    let err = rate_limited(None);
    assert_eq!(err.kind(), Some(ApiErrorKind::RateLimit));
    assert_eq!(err.status(), Some(429));
    assert_eq!(Error::Config("x".into()).kind(), None);
}

/// Port of the Python SDK's `test_backoff_extreme_values`, adjusted for
/// `Duration` (nanosecond precision, saturating at u64 ns).
#[test]
fn backoff_extreme_values() {
    // Sub-millisecond initial delay keeps its precision.
    let p = RetryPolicy::new()
        .backoff_initial(Duration::from_nanos(1))
        .backoff_max(Duration::from_secs(10))
        .backoff_jitter(0.0);
    assert_eq!(p.backoff_delay(0), Duration::from_nanos(1));
    assert_eq!(p.backoff_delay(1), Duration::from_nanos(2));
    // Cap smaller than the initial delay binds immediately.
    let p2 = RetryPolicy::new()
        .backoff_initial(Duration::from_millis(500))
        .backoff_max(Duration::from_micros(600))
        .backoff_jitter(0.0);
    assert_eq!(p2.backoff_delay(0), Duration::from_micros(600));
    // Very large values saturate instead of overflowing.
    let p3 = RetryPolicy::new()
        .backoff_initial(Duration::from_secs(60))
        .backoff_max(Duration::MAX)
        .backoff_jitter(0.0);
    assert_eq!(p3.backoff_delay(0), Duration::from_secs(60));
    assert_eq!(
        p3.backoff_delay(40),
        Duration::from_nanos(u64::MAX),
        "saturates at the widest representable duration"
    );
}

/// Port of `test_retry_policy_custom_statuses`: a replacing status set
/// plus `retry_server_errors(false)` retries only the listed codes.
#[test]
fn custom_statuses_replace_defaults() {
    let p = RetryPolicy::new()
        .with_retry_statuses([409])
        .retry_server_errors(false);
    assert!(p.is_retryable(&api_error(409)));
    assert!(
        !p.is_retryable(&api_error(408)),
        "default 408 must be replaced"
    );
    assert!(
        !p.is_retryable(&api_error(429)),
        "default 429 must be replaced"
    );
    assert!(
        !p.is_retryable(&api_error(500)),
        "5xx needs retry_server_errors"
    );
}

/// Port of `test_retry_policy_exceptions_and_predicate`: a predicate opts
/// non-transient statuses (404) into retrying.
#[test]
fn predicate_opts_errors_into_retry() {
    let p = RetryPolicy::new().retry_predicate(Some(std::sync::Arc::new(|error: &Error| {
        error.status() == Some(404)
    })));
    assert!(p.is_retryable(&api_error(404)));
    assert!(!RetryPolicy::new().is_retryable(&api_error(404)));
    // Without the predicate the defaults stand.
    assert!(!p.is_retryable(&api_error(401)));
}

#[test]
fn long_server_delays_are_honored_without_cap() {
    // Python: "Server-requested delays are always honored, however long."
    let p = RetryPolicy::new().backoff_max(Duration::from_secs(5));
    assert_eq!(
        p.delay_for(&rate_limited(Some(Duration::from_secs(61))), 0),
        Duration::from_secs(61)
    );
    assert_eq!(
        p.delay_for(&rate_limited(Some(Duration::from_millis(60_001))), 0),
        Duration::from_millis(60_001)
    );
    // Unparseable header falls back to backoff (500ms default, jitter 0).
    let fallback = RetryPolicy::new().backoff_jitter(0.0);
    assert_eq!(
        fallback.delay_for(&rate_limited(None), 0),
        Duration::from_millis(500)
    );
}
