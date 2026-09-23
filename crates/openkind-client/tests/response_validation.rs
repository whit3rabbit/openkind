//! A successful HTTP status is insufficient for a request-bound decision.

mod common;

use common::*;
use openkind_client::{question, Error, RetryPolicy};
use openkind_core::ValidationError;
use serde_json::json;

#[tokio::test]
async fn client_rejects_out_of_list_choice_without_retry_or_response_leak() {
    let (url, requests) = spawn(|_| {
        Outcome::success(json!({
            "model": "jev-latest",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "secret-unrequested-action",
                    "probabilities": {"inspect": 0.6, "verify": 0.4},
                    "confidence": 0.6
                }
            }
        }))
    })
    .await;
    let client = client(&url, RetryPolicy::new().max_retries(3));
    let err = client
        .system_one(
            "state",
            [(
                "route",
                question::choice("Next?", [("inspect", None), ("verify", None)]),
            )],
        )
        .await
        .unwrap_err();

    assert!(matches!(
        &err,
        Error::InvalidResponse {
            source: ValidationError::ChoiceNotInCriteria(_)
        }
    ));
    assert!(!err.to_string().contains("secret-unrequested-action"));
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn client_rejects_answer_type_mismatch() {
    let (url, requests) = spawn(|_| {
        Outcome::success(json!({
            "model": "jev-latest",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"route": {"type": "noul", "noul": 0.5}}
        }))
    })
    .await;
    let client = client(&url, RetryPolicy::new());
    let err = client
        .system_one(
            "state",
            [("route", question::choice("Next?", [("inspect", None)]))],
        )
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidResponse {
            source: ValidationError::AnswerTypeMismatch {
                expected: "choice",
                actual: "noul",
                ..
            }
        }
    ));
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn client_accepts_matching_choice_response() {
    let (url, requests) = spawn(|captured| Outcome::success(result_for(captured))).await;
    let client = client(&url, RetryPolicy::new().max_retries(0));
    let response = client
        .system_one(
            "state",
            [(
                "route",
                question::choice("Next?", [("inspect", None), ("verify", None)]),
            )],
        )
        .await
        .unwrap();
    assert_eq!(response.answers.len(), 1);
    assert_eq!(requests.len(), 1);
}
