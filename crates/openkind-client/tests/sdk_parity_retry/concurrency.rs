use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openkind_client::{question, Client, Error, RequestOptions, RetryPolicy, State, SystemRequest};
use serde_json::json;

use super::common::*;

// ---------------------------------------------------------------------
// test_async_concurrent_retry_state — independent state across tasks
// ---------------------------------------------------------------------
#[tokio::test]
async fn concurrent_calls_have_independent_retry_state() {
    let counts: Arc<Mutex<HashMap<String, u32>>> = Arc::new(Mutex::new(HashMap::default()));
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
        client: &openkind_client::Client,
        i: u8,
    ) -> Result<openkind_client::ModelsResponse, Error> {
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
                questions: std::collections::HashMap::from_iter([(
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
        questions: std::collections::HashMap::from_iter([("q".to_string(), question::noul("?"))]),
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
