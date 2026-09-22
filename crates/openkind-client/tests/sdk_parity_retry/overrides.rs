use std::sync::Arc;
use std::time::Duration;

use openkind_client::{Error, RequestOptions, RetryPolicy};
use serde_json::json;

use super::common::*;

// ---------------------------------------------------------------------
// test_retry_policy_per_call_override — call policy beats client policy
// ---------------------------------------------------------------------
#[tokio::test]
async fn per_call_retry_override_wins() {
    let (url, requests) =
        spawn(move |_| Outcome::error(429, json!({})).with_retry_after_ms(0)).await;
    let client = client(&url, RetryPolicy::new().max_retries(2));

    // Per-call policy disables retries: one attempt.
    let err = client
        .evaluate_with(
            evaluate_request(),
            &RequestOptions::new().retry(RetryPolicy::new().max_retries(0)),
        )
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(429));
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn per_call_retry_override_can_extend_retries() {
    let (url, requests) =
        spawn(move |_| Outcome::error(429, json!({})).with_retry_after_ms(0)).await;
    // Client policy allows 2 retries; the call asks for 4.
    let client = client(&url, RetryPolicy::new().max_retries(2));
    client
        .evaluate_with(
            evaluate_request(),
            &RequestOptions::new().retry(RetryPolicy::new().max_retries(4)),
        )
        .await
        .unwrap_err();
    assert_eq!(requests.len(), 5);
}

// ---------------------------------------------------------------------
// test_retry_policy_exceptions_and_predicate — 404 opted into retrying
// ---------------------------------------------------------------------
#[tokio::test]
async fn retry_predicate_opts_404_into_retries() {
    let (url, requests) =
        spawn(move |_| Outcome::error(404, json!({"message": "gone"})).with_retry_after_ms(0))
            .await;
    let policy = RetryPolicy::new()
        .max_retries(1)
        .retry_predicate(Some(Arc::new(|error: &Error| error.status() == Some(404))));
    let client = client(&url, policy);
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert_eq!(err.status(), Some(404));
    assert_eq!(requests.len(), 2, "predicate opts the 404 into one retry");
}

// ---------------------------------------------------------------------
// test_zero_backoff_retries — zero backoff still retries, either outcome
// ---------------------------------------------------------------------
#[tokio::test]
async fn zero_backoff_still_retries() {
    for (initial_ms, _max_ms, recover) in [
        (0, 5_000, false),
        (500, 0, false),
        (0, 0, false),
        (0, 0, true),
        (500, 0, true),
    ] {
        let (url, requests) = spawn(move |captured: &Captured| {
            if recover && captured.retry_count.as_deref() == Some("1") {
                Outcome::success(json!({"models": []}))
            } else {
                Outcome::error(503, json!({"message": "temporarily unavailable"}))
            }
        })
        .await;
        let policy = RetryPolicy::new()
            .max_retries(1)
            .backoff_initial(Duration::from_millis(initial_ms))
            .backoff_jitter(0.0);
        let client = client(&url, policy);
        let result = client.list_models().await;
        assert_eq!(requests.len(), 2, "one retry regardless of backoff");
        assert_eq!(requests.retry_sequence(), vec![None, Some("1".into())]);
        if recover {
            assert!(result.is_ok(), "recover case must succeed");
        } else {
            assert_eq!(result.unwrap_err().status(), Some(503));
        }
    }
}
