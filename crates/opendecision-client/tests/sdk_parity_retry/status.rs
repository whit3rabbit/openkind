use opendecision_client::RetryPolicy;
use serde_json::json;

use super::common::*;

// ---------------------------------------------------------------------
// test_default_retry_statuses — status matrix
// ---------------------------------------------------------------------
#[tokio::test]
async fn default_retry_statuses_match_python_matrix() {
    // (status, expected attempts): 408/429/5xx retried by default (529 is in
    // the 5xx band); every other code surfaces after a single attempt.
    let cases: &[(u16, usize)] = &[
        (408, 3),
        (429, 3),
        (500, 3),
        (503, 3),
        (599, 3),
        (529, 3),
        (400, 1),
        (401, 1),
        (403, 1),
        (404, 1),
        (409, 1),
        (422, 1),
        (302, 1),
    ];
    for &(status, attempts) in cases {
        let (url, requests) = spawn(move |_| {
            Outcome::error(status, json!({"message": "failed"})).with_retry_after_ms(0)
        })
        .await;
        let client = client(&url, RetryPolicy::new());
        let err = client.evaluate(evaluate_request()).await.unwrap_err();
        assert_eq!(err.status(), Some(status));
        assert_eq!(requests.len(), attempts, "status {status}");
    }
}

#[tokio::test]
async fn retried_requests_carry_python_retry_count_sequence() {
    let (url, requests) =
        spawn(move |_| Outcome::error(429, json!({"message": "slow"})).with_retry_after_ms(0))
            .await;
    let client = client(&url, RetryPolicy::new());
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert_eq!(err.status(), Some(429));
    // The Python suite pins exactly this header sequence for 2 retries.
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests.retry_sequence(),
        vec![None, Some("1".into()), Some("2".into())]
    );
}

// ---------------------------------------------------------------------
// test_retry_policy_max_retries — (0, 1), (1, 2), (4, 5)
// ---------------------------------------------------------------------
#[tokio::test]
async fn max_retries_table() {
    for (max_retries, attempts) in [(0, 1), (1, 2), (4, 5)] {
        let (url, requests) =
            spawn(move |_| Outcome::error(429, json!({"message": "slow"})).with_retry_after_ms(0))
                .await;
        let client = client(&url, RetryPolicy::new().max_retries(max_retries));
        let err = client.evaluate(evaluate_request()).await.unwrap_err();
        assert_eq!(err.status(), Some(429));
        assert_eq!(requests.len(), attempts, "max_retries={max_retries}");
    }
}

// ---------------------------------------------------------------------
// test_retry_policy_custom_statuses — replacing set: 409 retried, 500 not
// ---------------------------------------------------------------------
#[tokio::test]
async fn custom_statuses_replace_defaults() {
    let policy = RetryPolicy::new()
        .with_retry_statuses([409])
        .retry_server_errors(false);
    for (status, attempts) in [(409, 3), (500, 1)] {
        let (url, requests) =
            spawn(move |_| Outcome::error(status, json!({"message": "x"})).with_retry_after_ms(0))
                .await;
        let client = client(&url, policy.clone());
        client.evaluate(evaluate_request()).await.unwrap_err();
        assert_eq!(requests.len(), attempts, "status {status}");
    }
}
