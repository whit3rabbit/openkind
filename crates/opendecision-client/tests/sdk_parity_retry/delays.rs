use std::time::{Duration, Instant};

use opendecision_client::RetryPolicy;
use serde_json::json;

use super::common::*;

// ---------------------------------------------------------------------
// test_server_delay_through_tenacity — server delay wins over backoff
// ---------------------------------------------------------------------
#[tokio::test]
async fn retry_after_ms_honored_over_long_backoff() {
    let (url, requests) = spawn(move |captured: &Captured| {
        if captured.retry_count.is_none() {
            Outcome::error(429, json!({})).with_retry_after_ms(20)
        } else {
            Outcome::success(json!({"models": []}))
        }
    })
    .await;
    // Backoff would be 10s; the server asks for 20ms.
    let client = client(
        &url,
        RetryPolicy::new()
            .max_retries(3)
            .backoff_initial(Duration::from_secs(10)),
    );
    let start = Instant::now();
    client.list_models().await.unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "server delay must override backoff, took {:?}",
        start.elapsed()
    );
    assert_eq!(requests.len(), 2);
}

#[tokio::test]
async fn retry_after_raw_seconds_honored() {
    let (url, requests) = spawn(move |captured: &Captured| {
        if captured.retry_count.is_none() {
            Outcome::error(429, json!({})).with_retry_after_raw("1")
        } else {
            Outcome::success(json!({"models": []}))
        }
    })
    .await;
    let client = client(
        &url,
        RetryPolicy::new()
            .max_retries(3)
            .backoff_initial(Duration::from_secs(60)),
    );
    let start = Instant::now();
    client.list_models().await.unwrap();
    // 1s wait < the 60s backoff it replaced; allow generous CI headroom.
    assert!(
        start.elapsed() <= Duration::from_secs(4),
        "{:?}",
        start.elapsed()
    );
    assert_eq!(requests.len(), 2);
}

// ---------------------------------------------------------------------
// test_retry_policy_timeout_budget — budget stops retries, fresh per call
// ---------------------------------------------------------------------
#[tokio::test]
async fn budget_stops_retrying() {
    let (url, requests) =
        spawn(move |_| Outcome::error(429, json!({})).with_retry_after_ms(25)).await;
    let client = client(
        &url,
        RetryPolicy::new()
            .max_retries(100)
            .backoff_initial(Duration::from_millis(1))
            .backoff_jitter(0.0)
            .total_timeout(Some(Duration::from_millis(100))),
    );
    let start = Instant::now();
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert_eq!(err.status(), Some(429));
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(
        requests.len() < 6,
        "budget must cap attempts: {}",
        requests.len()
    );
}

#[tokio::test]
async fn budget_is_fresh_per_call() {
    let (url, requests) =
        spawn(move |_| Outcome::error(429, json!({})).with_retry_after_ms(0)).await;
    let client = client(
        &url,
        RetryPolicy::new()
            .max_retries(100)
            .backoff_initial(Duration::from_millis(1))
            .backoff_jitter(0.0)
            .total_timeout(Some(Duration::from_millis(30))),
    );
    let mut used_per_call = Vec::new();
    for _ in 0..2 {
        let before = requests.len();
        let err = client.evaluate(evaluate_request()).await.unwrap_err();
        assert_eq!(err.status(), Some(429));
        used_per_call.push(requests.len() - before);
    }
    // Fresh budget per call: both calls make (approximately) the same number
    // of attempts rather than the second resuming the first's spent budget.
    let first = used_per_call[0];
    assert!(first >= 5, "budget must allow many retries, used {first}");
    // The second call must start from a fresh budget: it retries roughly as
    // many times as the first (exact counts jitter with scheduler timing).
    assert!(
        used_per_call[1] >= first / 2 && used_per_call[1] <= first * 3 / 2 + 1,
        "attempt counts must be comparable across calls: {used_per_call:?}"
    );
}
