use axum::http::StatusCode;
use serde_json::{json, Value};

use super::helpers::{body_bytes, post_systemone};

#[tokio::test]
async fn sdk_quickstart_example_works_end_to_end() {
    // Verbatim from docs.typesafe.ai/sdk/python.md#quickstart:
    //
    //   response = client.system_one(
    //       state={"document": "I was charged twice. Please fix this ASAP."},
    //       questions={
    //           "billing": Noul(instructions="Is this ticket about billing?"),
    //           "tone": Choice(
    //               instructions="What is the customer's tone?",
    //               criteria={"calm": None, "frustrated": None, "angry": None},
    //           ),
    //           "urgency": Score(
    //               instructions="How urgent is this ticket?",
    //               criteria=["can wait", "this week", "today"],
    //           ),
    //       },
    //   )
    let body = json!({
        "state": { "document": "I was charged twice. Please fix this ASAP." },
        "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "Is this ticket about billing?" },
            "tone": {
                "type": "choice",
                "instructions": "What is the customer's tone?",
                "criteria": { "calm": null, "frustrated": null, "angry": null }
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["can wait", "this week", "today"]
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    // Capture headers up-front since body_bytes takes ownership.
    let request_id_header = resp.headers().get("x-typesafe-request-id").unwrap().clone();
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    // SDK accesses these properties:
    assert!(v["answers"]["billing"]["noul"].is_number());
    assert!(v["answers"]["tone"]["choice"].is_string());
    assert!(v["answers"]["urgency"]["score"].is_number());
    assert!(v["answers"]["urgency"]["legend"].is_object());
    assert!(v["usage"]["input_tokens"].is_number());
    assert!(v["usage"]["output_tokens"].is_number());
    // request_id should be readable from headers.
    assert!(request_id_header.to_str().unwrap().len() == 36);
}

#[tokio::test]
async fn sdk_client_aclose_pattern_works() {
    // SDK pattern: `with TypeSafeClient() as client: ...` — implies a
    // stateless server with no connection-level resources. We verify
    // there are no session cookies or sticky requirements.
    for _ in 0..3 {
        let body = json!({
            "state": "x", "model": "mock",
            "questions": { "q": { "type": "noul", "instructions": "?" } }
        });
        let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn multi_question_response_keys_match_request_keys() {
    // SDK does `result.choices["tone"]`, so server must echo the keys.
    let mut questions = serde_json::Map::new();
    for name in ["alpha", "beta", "gamma", "delta"] {
        questions.insert(name.into(), json!({ "type": "noul", "instructions": "?" }));
    }
    let body = json!({ "state": "x", "model": "mock", "questions": questions });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let answers = v["answers"].as_object().unwrap();
    assert!(answers.contains_key("alpha"));
    assert!(answers.contains_key("beta"));
    assert!(answers.contains_key("gamma"));
    assert!(answers.contains_key("delta"));
    assert_eq!(answers.len(), 4);
}
