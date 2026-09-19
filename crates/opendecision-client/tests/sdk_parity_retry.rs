//! Ports of the behavioral tests from the TypeSafe Python SDK's
//! `tests/test_retry.py` (`github.com/typesafe-ai/typesafe-sdk-python`).
//!
//! Where the Python suite asserts values handed to a mocked sleep, these
//! ports assert wall-clock bounds with millisecond-scale delays; semantics
//! (attempt counts, retry-count header sequences, precedence rules) are
//! asserted exactly.

mod common;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use common::*;
use opendecision_client::{
    question, Client, Error, RequestOptions, RetryPolicy, State, SystemRequest,
};
use serde_json::json;

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

// ---------------------------------------------------------------------
// test_exhausted_retry_preserves_final_http_error
// ---------------------------------------------------------------------
#[tokio::test]
async fn exhausted_retry_preserves_final_http_error() {
    let (url, requests) = spawn(move |captured: &Captured| {
        let attempt = captured
            .retry_count
            .as_deref()
            .map_or(1, |c| c.parse().unwrap())
            + 1;
        let (status, id): (u16, String) = match attempt {
            1 => (429, "request-1".into()),
            2 => (500, "request-2".into()),
            _ => (503, "request-3".into()),
        };
        Outcome::error(status, json!({"message": format!("attempt {attempt}")}))
            .with_retry_after_ms(0)
            .with_request_id(id)
    })
    .await;
    let client = client(&url, RetryPolicy::new());
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert_eq!(requests.len(), 3);
    // The LAST response is what the caller sees, not the first.
    assert_eq!(err.status(), Some(503));
    assert_eq!(err.request_id(), Some("request-3"));
    match &err {
        Error::Api(api) => {
            assert_eq!(api.message.as_deref(), Some("attempt 3"));
            assert_eq!(api.body.as_ref().unwrap()["message"], "attempt 3");
            assert_eq!(
                api.to_string(),
                "503: attempt 3 (POST /v1/systemone) (request_id=request-3)"
            );
        }
        other => panic!("expected Api error, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// test_connection_retry_recovers — timeout flavor: retried then recovers
// ---------------------------------------------------------------------
#[tokio::test]
async fn timeout_errors_are_retried_then_recover() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_handler = attempts.clone();
    let app = Router::new().route(
        "/v1/systemone",
        post(move |_req: axum::http::Request<axum::body::Body>| {
            let n = attempts_for_handler.fetch_add(1, Ordering::SeqCst);
            async move {
                if n == 0 {
                    // First attempt stalls past the client timeout.
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Json(json!({
                    "model": "client-model",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "answers": {"q": {"type": "noul", "noul": 0.5}}
                }))
                .into_response()
            }
        }),
    );
    let url = spawn_router(app).await;
    let client = client(
        &url,
        RetryPolicy::new()
            .max_retries(3)
            .backoff_initial(Duration::from_millis(1))
            .backoff_jitter(0.0),
    );
    let started = Instant::now();
    let response = client
        .evaluate_with(
            evaluate_request(),
            &RequestOptions::new().timeout(Duration::from_millis(50)),
        )
        .await
        .expect("timeout error must be retried into success");
    assert_eq!(response.answers.len(), 1);
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert!(
        started.elapsed() >= Duration::from_millis(50),
        "first attempt really timed out"
    );
}

// ---------------------------------------------------------------------
// test_async_concurrent_retry_state — independent state across tasks
// ---------------------------------------------------------------------
#[tokio::test]
async fn concurrent_calls_have_independent_retry_state() {
    let counts: Arc<Mutex<HashMap<String, u32>>> = Arc::new(Mutex::new(HashMap::new()));
    let (url, requests) = spawn(move |captured: &Captured| {
        let key = captured.call.clone().unwrap_or_default();
        let mut counts = counts.lock().unwrap();
        let seen = counts.entry(key).or_insert(0);
        *seen += 1;
        if *seen == 1 {
            Outcome::error(429, json!({})).with_retry_after_ms(0)
        } else {
            Outcome::success(json!({"models": []}))
        }
    })
    .await;
    let client = client(
        &url,
        RetryPolicy::new()
            .max_retries(3)
            .backoff_initial(Duration::from_millis(1))
            .backoff_jitter(0.0),
    );

    async fn one_call(
        client: &opendecision_client::Client,
        i: u8,
    ) -> Result<opendecision_client::ModelsResponse, Error> {
        let opts = RequestOptions::new().header(
            axum::http::HeaderName::from_static("x-call"),
            axum::http::HeaderValue::from_str(&i.to_string()).unwrap(),
        );
        client.list_models_with(&opts).await
    }
    let (r0, r1, r2, r3) = tokio::join!(
        one_call(&client, 0),
        one_call(&client, 1),
        one_call(&client, 2),
        one_call(&client, 3)
    );
    let results = [r0, r1, r2, r3];
    assert!(results.iter().all(|r| r.is_ok()));
    for i in 0..4u8 {
        let key = i.to_string();
        let seq: Vec<Option<String>> = requests
            .calls(&key)
            .iter()
            .map(|r| r.retry_count.clone())
            .collect();
        assert_eq!(seq, vec![None, Some("1".into())], "call {key}");
    }
}

// ---------------------------------------------------------------------
// test_concurrent_system_one_overrides — per-call isolation
// ---------------------------------------------------------------------
#[tokio::test]
async fn concurrent_calls_with_distinct_policies_and_models() {
    let (url, requests) =
        spawn(move |_| Outcome::error(429, json!({"message": "retry"})).with_retry_after_ms(0))
            .await;
    let client = client(
        &url,
        RetryPolicy::new()
            .max_retries(1)
            .backoff_initial(Duration::from_millis(1))
            .backoff_jitter(0.0),
    );

    let mk = |name: &'static str, retries: u32, timeout_ms: u64| {
        let client = client.clone();
        async move {
            let request = SystemRequest {
                state: State::Text(name.into()),
                model: name.into(),
                questions: std::collections::HashMap::from([(
                    "q".to_string(),
                    question::noul("?"),
                )]),
            };
            let opts = RequestOptions::new()
                .retry(RetryPolicy::new().max_retries(retries).backoff_jitter(0.0))
                .timeout(Duration::from_millis(timeout_ms))
                .header(
                    axum::http::HeaderName::from_static("x-call"),
                    axum::http::HeaderValue::from_str(name).unwrap(),
                );
            let err = client.evaluate_with(request, &opts).await.unwrap_err();
            (name, err.status().unwrap())
        }
    };

    let (one, three, default) = tokio::join!(
        mk("one", 0, 1_000),
        mk("three", 2, 3_000),
        mk("default", 1, 2_000)
    );
    assert_eq!(one, ("one", 429));
    assert_eq!(three, ("three", 429));
    assert_eq!(default, ("default", 429));

    assert_eq!(requests.calls("one").len(), 1);
    assert_eq!(requests.calls("three").len(), 3);
    assert_eq!(requests.calls("default").len(), 2);

    // Each call's requests carry its own model/state and retry sequence.
    for (name, count) in [("one", 1), ("three", 3), ("default", 2)] {
        let group = requests.calls(name);
        let seq: Vec<Option<String>> = group.iter().map(|r| r.retry_count.clone()).collect();
        let expected = (0..count)
            .map(|i| if i == 0 { None } else { Some(i.to_string()) })
            .collect::<Vec<_>>();
        assert_eq!(seq, expected, "{name}");
        for captured in &group {
            let body = captured.body.as_ref().unwrap();
            assert_eq!(body["model"], name);
            assert_eq!(body["state"], name);
        }
    }
}

// ---------------------------------------------------------------------
// test_system_one_retry_recovers_with_overrides — per-attempt wire asserts
// ---------------------------------------------------------------------
#[tokio::test]
async fn retry_recovery_with_full_wire_assertions() {
    let (url, requests) = spawn(move |captured: &Captured| {
        if captured.retry_count.is_none() {
            Outcome::error(429, json!({"message": "slow down"})).with_retry_after_ms(1)
        } else if captured.retry_count.as_deref() == Some("1") {
            Outcome::error(500, json!({"message": "backend hiccup"}))
        } else {
            Outcome::success(result())
        }
    })
    .await;
    let client = Client::builder()
        .api_key("test-key")
        .base_url(&url)
        .default_model("client-model")
        .timeout(Duration::from_secs(5))
        .retry(RetryPolicy::new().max_retries(5).backoff_jitter(0.0))
        .build()
        .unwrap();

    let request = SystemRequest {
        state: State::Object(json!({"document": "hello"}).as_object().unwrap().clone()),
        model: "call-model".into(),
        questions: std::collections::HashMap::from([("q".to_string(), question::noul("?"))]),
    };
    let opts = RequestOptions::new()
        .timeout(Duration::from_millis(3_000))
        .header(
            axum::http::HeaderName::from_static("x-default"),
            axum::http::HeaderValue::from_static("kept"),
        )
        .header(
            axum::http::HeaderName::from_static("x-call"),
            axum::http::HeaderValue::from_static("override"),
        );

    let response = client.evaluate_with(request, &opts).await.unwrap();
    assert_eq!(response.model, "jev-latest");
    assert_eq!(response.usage.input_tokens, 12);

    // Every attempt sent the identical wire body and headers.
    let expected = json!({
        "state": {"document": "hello"},
        "model": "call-model",
        "questions": {"q": {"type": "noul", "instructions": "?"}}
    });
    assert_eq!(requests.len(), 3);
    for captured in requests.snapshot() {
        assert_eq!(captured.body.as_ref().unwrap(), &expected);
        assert_eq!(captured.header("authorization"), Some("Bearer test-key"));
        assert_eq!(captured.header("x-default"), Some("kept"));
        assert_eq!(captured.header("x-call"), Some("override"));
        assert_eq!(captured.header("content-type"), Some("application/json"));
    }
    assert_eq!(
        requests.retry_sequence(),
        vec![None, Some("1".into()), Some("2".into())]
    );

    // A plain follow-up call uses the client defaults again: default model,
    // no per-call headers, no retry-count on the initial attempt.
    drop(requests);
    let (url2, requests2) = spawn(move |_| Outcome::success(result())).await;
    let client2 = Client::builder()
        .api_key("test-key")
        .base_url(&url2)
        .default_model("client-model")
        .timeout(Duration::from_secs(5))
        .retry(RetryPolicy::new().max_retries(0))
        .build()
        .unwrap();
    let response = client2
        .system_one("next", [("q", question::noul("?"))])
        .await
        .unwrap();
    assert_eq!(response.model, "jev-latest");
    let captured = requests2.last().unwrap();
    assert_eq!(captured.body.as_ref().unwrap()["model"], "client-model");
    assert!(captured.header("x-call").is_none());
    assert!(captured.retry_count.is_none());
}

// ---------------------------------------------------------------------
// test_cancel_pending_retry — cancellation during the retry sleep
// ---------------------------------------------------------------------
#[tokio::test]
async fn cancel_pending_retry() {
    let (url, _requests) =
        spawn(move |_| Outcome::error(429, json!({})).with_retry_after_ms(60_000)).await;
    // No total budget: the 60s server-requested delay becomes a real sleep,
    // which is what we cancel (the Python suite blocks inside a mocked sleep).
    let client = client(&url, RetryPolicy::new().max_retries(5).total_timeout(None));

    let task = tokio::spawn(async move { client.evaluate(evaluate_request()).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    task.abort();
    let joined = task.await;
    assert!(
        matches!(joined, Err(join_err) if join_err.is_cancelled()),
        "pending retry must be cancellable"
    );
}
