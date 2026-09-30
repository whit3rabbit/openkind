use std::time::{Duration, Instant};

use openkind_client::RetryPolicy;
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
    let (url, requests) = spawn(|captured| {
        if captured.retry_count.is_none() {
            Outcome::error(429, json!({})).with_retry_after_ms(0)
        } else {
            Outcome::success(result_for(captured))
        }
    })
    .await;
    let budget = Duration::from_secs(1);
    let client = client(
        &url,
        RetryPolicy::new()
            .max_retries(1)
            .backoff_initial(Duration::ZERO)
            .backoff_jitter(0.0)
            .total_timeout(Some(budget)),
    );
    client.evaluate(evaluate_request()).await.unwrap();
    assert_eq!(requests.retry_sequence(), vec![None, Some("1".into())]);

    // A reused timer would be expired before the second call's required retry.
    tokio::time::sleep(budget + Duration::from_millis(100)).await;
    client.evaluate(evaluate_request()).await.unwrap();
    assert_eq!(
        requests.retry_sequence(),
        vec![None, Some("1".into()), None, Some("1".into())]
    );
}
