//! Provider routing contracts for the shared Jev question and answer subset.

mod common;

use common::{spawn, Outcome};
use openkind_client::{question, Client, Error};
use serde_json::json;

#[tokio::test]
async fn openrouter_uses_system_one_path_and_accepts_provider_metadata() {
    let (url, requests) = spawn(|_| {
        Outcome::success(json!({
            "id": "gen-dec-example",
            "model": "typesafe/jev-1.13-20260917",
            "provider": "TypeSafe",
            "answers": {"refund": {"type": "noul", "noul": 0.98}},
            "usage": {"input_tokens": 275, "output_tokens": 20, "cost": 0.00003}
        }))
    })
    .await;
    let client = Client::builder()
        .api_key("openrouter-test-key")
        .base_url(url)
        .default_model("jev-1.13")
        .build()
        .unwrap();

    let response = client
        .system_one(
            "I was charged twice.",
            [("refund", question::noul("Is a refund requested?"))],
        )
        .await
        .unwrap();
    let captured = requests.last().unwrap();
    assert_eq!(captured.path, "/v1/systemone");
    assert_eq!(
        captured.header("authorization"),
        Some("Bearer openrouter-test-key")
    );
    assert_eq!(captured.body.as_ref().unwrap()["model"], "jev-1.13");
    assert_eq!(response.model, "typesafe/jev-1.13-20260917");
    assert!(response.answers.contains_key("refund"));
}

#[tokio::test]
async fn cloudflare_wraps_input_and_unwraps_result() {
    let (url, requests) = spawn(|_| {
        Outcome::success(json!({
            "success": true,
            "result": {
                "model": "jev-1.13.0",
                "answers": {"refund": {"type": "noul", "noul": 0.95}},
                "usage": {"input_tokens": 426, "output_tokens": 73}
            }
        }))
    })
    .await;
    let client = Client::builder()
        .api_key("cloudflare-test-token")
        .cloudflare_account("0123456789abcdef")
        .base_url(url)
        .build()
        .unwrap();

    let response = client
        .system_one(
            "I was charged twice.",
            [("refund", question::noul("Is a refund requested?"))],
        )
        .await
        .unwrap();
    let captured = requests.last().unwrap();
    assert_eq!(captured.path, "/ai/run");
    assert_eq!(captured.body.as_ref().unwrap()["model"], "typesafe/jev");
    assert_eq!(
        captured.body.as_ref().unwrap()["input"]["state"],
        "I was charged twice."
    );
    assert_eq!(
        captured.body.as_ref().unwrap()["input"]["questions"]["refund"]["type"],
        "noul"
    );
    assert!(captured.body.as_ref().unwrap().get("questions").is_none());
    assert_eq!(response.model, "jev-1.13.0");
    assert!(matches!(client.list_models().await, Err(Error::Config(_))));
}

#[test]
fn cloudflare_account_must_be_one_path_segment() {
    assert!(Client::builder()
        .api_key("test")
        .cloudflare_account("../other")
        .build()
        .is_err());
}

#[test]
fn cloudflare_uses_account_url_and_jev_model_by_default() {
    let client = Client::builder()
        .api_key("test")
        .cloudflare_account("0123456789abcdef")
        .build()
        .unwrap();
    assert_eq!(
        client.base_url(),
        "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef"
    );
    assert_eq!(client.default_model(), "typesafe/jev");
}
