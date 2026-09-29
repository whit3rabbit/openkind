//! Ports of the wire-conformance tests from the TypeSafe Python SDK:
//! `test_clients.py::test_round_trip` / `test_models_shape` /
//! `test_models_ignore_unknown_fields` / `test_invalid_models_response`,
//! `test_responses.py` (tolerated extras, malformed bodies, typed answers),
//! `test_questions.py` (discriminators, omitted defaults), and
//! `test_types.py` (array/object states).

mod common;

use std::collections::HashMap;

use common::*;
use openkind_client::{
    question, Answer, ChoiceAnswer, Error, ModelsResponse, NoulAnswer, RetryPolicy, ScoreAnswer,
    State, SystemRequest, Usage,
};
use serde_json::json;

fn no_retries() -> RetryPolicy {
    RetryPolicy::new().max_retries(0)
}

// ---------------------------------------------------------------------
// test_round_trip — exact wire body and fully-typed response
// ---------------------------------------------------------------------
#[tokio::test]
async fn round_trip_exact_wire_body_and_typed_response() {
    let (url, requests) = spawn(move |_| Outcome::success(result())).await;
    let client = client(&url, no_retries());

    let request = SystemRequest {
        state: State::Object(json!({"document": "Hello 🌍"}).as_object().unwrap().clone()),
        model: "jev-latest".into(),
        questions: HashMap::from_iter([
            ("spam".to_string(), question::noul("Spam?")),
            (
                "tone".to_string(),
                question::choice("Tone?", [("friendly", None), ("hostile", None)]),
            ),
            (
                "quality".to_string(),
                question::score("Quality?", ["bad", "ok", "great"]),
            ),
        ]),
    };
    let response = client.evaluate(request).await.unwrap();

    // The exact body the Python suite asserts (order-insensitive JSON).
    assert_eq!(requests.len(), 1);
    let sent = requests.last().unwrap();
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/v1/systemone");
    assert_eq!(sent.header("content-type"), Some("application/json"));
    assert_eq!(
        sent.body.as_ref().unwrap(),
        &json!({
            "state": {"document": "Hello 🌍"},
            "model": "jev-latest",
            "questions": {
                "spam": {"type": "noul", "instructions": "Spam?"},
                "tone": {
                    "type": "choice",
                    "instructions": "Tone?",
                    "criteria": {"friendly": null, "hostile": null}
                },
                "quality": {"type": "score", "instructions": "Quality?", "criteria": ["bad", "ok", "great"]}
            }
        })
    );

    // Fully-typed response access, matching the Python assertions.
    assert_eq!(response.model, "jev-latest");
    assert_eq!(
        response.usage,
        Usage {
            input_tokens: 12,
            output_tokens: 3
        }
    );
    assert_eq!(response.answers.len(), 3);
    match &response.answers["spam"] {
        Answer::Noul(a) => assert_eq!(a.noul, 0.98),
        other => panic!("expected noul, got {other:?}"),
    }
    match &response.answers["tone"] {
        Answer::Choice(a) => {
            assert_eq!(a.choice, "friendly");
            assert_eq!(a.confidence, 0.9);
            assert_eq!(
                a.probabilities,
                HashMap::from_iter([("friendly".to_string(), 0.9), ("hostile".to_string(), 0.1)])
            );
        }
        other => panic!("expected choice, got {other:?}"),
    }
    match &response.answers["quality"] {
        Answer::Score(a) => {
            assert_eq!(a.score, 1.7);
            assert_eq!(a.confidence, 0.8);
            // Rust keeps the wire's string keys; Python's public model uses ints.
            assert_eq!(
                a.legend,
                HashMap::from_iter([
                    ("0".to_string(), "bad".to_string()),
                    ("1".to_string(), "ok".to_string()),
                    ("2".to_string(), "great".to_string())
                ])
            );
            assert_eq!(
                a.probabilities,
                HashMap::from_iter([
                    ("0".to_string(), 0.1),
                    ("1".to_string(), 0.1),
                    ("2".to_string(), 0.8)
                ])
            );
        }
        other => panic!("expected score, got {other:?}"),
    }
    // The inner answer structs deserialize without the "type" tag too —
    // the tag belongs to the enum, the payload to the struct.
    let noul_payload: NoulAnswer = serde_json::from_value(json!({"noul": 0.98})).unwrap();
    assert_eq!(noul_payload.noul, 0.98);
    let _choice_payload: ChoiceAnswer = serde_json::from_value(json!({
        "choice": "friendly", "confidence": 0.9,
        "probabilities": {"friendly": 0.9, "hostile": 0.1}
    }))
    .unwrap();
    let _score_payload: ScoreAnswer = serde_json::from_value(json!({
        "score": 1.7, "confidence": 0.8,
        "legend": {"0": "bad", "1": "ok", "2": "great"},
        "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}
    }))
    .unwrap();
}

// ---------------------------------------------------------------------
// test_models_shape + test_models_ignore_unknown_fields
// ---------------------------------------------------------------------
#[tokio::test]
async fn models_response_shape_and_unknown_fields() {
    let (url, requests) = spawn(move |_| {
        Outcome::success(json!({
            "models": [{
                "name": "jev-latest",
                "description": "Fast model",
                "release_date": "2026-08-01",
                "unknown_future_field": {"nested": true}
            }],
            "also_unknown": 1
        }))
    })
    .await;
    let client = client(&url, no_retries());
    let models: ModelsResponse = client.list_models().await.unwrap();

    assert_eq!(requests.last().unwrap().path, "/v1/models");
    assert_eq!(requests.last().unwrap().method, "GET");
    assert_eq!(models.models.len(), 1);
    let card = &models.models[0];
    assert_eq!(card.name, "jev-latest");
    assert_eq!(card.description, "Fast model");
    assert_eq!(card.release_date, "2026-08-01");
}

// ---------------------------------------------------------------------
// test_invalid_models_response — missing required field is a decode error
// ---------------------------------------------------------------------
#[tokio::test]
async fn invalid_models_response_is_decode_error() {
    // (body, expected message fragment) — the fragment pins which field the
    // decode error names, the closest Rust behavior to the Python SDK's
    // `test_nested_missing_field_path` field-path assertion.
    for (body, message_fragment) in [
        (json!({}), None), // missing "models"
        (
            json!({"models": [{"description": "x"}]}),
            Some("missing field `name`"),
        ), // missing "name"
        (json!({"models": {"name": "x"}}), None), // wrong shape
        (json!({"models": [{"name": 1}]}), None), // wrong type
    ] {
        let expected = body.clone();
        let (url, _requests) = spawn(move |_| Outcome::success(expected.clone())).await;
        let client = client(&url, no_retries());
        match client.list_models().await.unwrap_err() {
            err @ Error::Decode { status, .. } => {
                assert_eq!(status, 200);
                if let Some(fragment) = message_fragment {
                    assert!(
                        err.to_string().contains(fragment),
                        "expected `{fragment}` in {err}"
                    );
                }
            }
            other => panic!("expected Decode for {body}, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------
// test_unknown_extra_fields_tolerated — response extras are ignored
// ---------------------------------------------------------------------
#[tokio::test]
async fn unknown_extra_fields_tolerated() {
    let (url, _requests) = spawn(move |captured| {
        let mut response = result_for(captured);
        response["new_top_level"] = json!({"anything": [1, 2]});
        response["answers"]["q"]["extra_answer_field"] = json!(true);
        Outcome::success(response)
    })
    .await;
    let client = client(&url, no_retries());
    let response = client.evaluate(evaluate_request()).await.unwrap();
    assert_eq!(response.answers.len(), 1);
}

// ---------------------------------------------------------------------
// Unknown answer types: intentional strict-mode divergence
// ---------------------------------------------------------------------
#[tokio::test]
async fn unknown_answer_type_is_strict() {
    // The Python SDK silently drops answers with unknown "type" tags; the
    // Rust client deliberately fails closed (the wire contract adds a type,
    // and silently losing judgments is worse than erroring). Document the
    // divergence with a test so a future change is conscious.
    let body = json!({
        "model": "jev-latest",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "answers": {
            "known": {"type": "noul", "noul": 0.5},
            "mystery": {"type": "vibes", "vibe": "immaculate"}
        }
    });
    let (url, _requests) = spawn(move |_| Outcome::success(body.clone())).await;
    let client = client(&url, no_retries());
    let err = client.evaluate(evaluate_request()).await.unwrap_err();
    assert!(matches!(err, Error::Decode { .. }), "got {err:?}");
}

// ---------------------------------------------------------------------
// test_responses.py::test_malformed_response_raises_validation_error
// ---------------------------------------------------------------------
#[tokio::test]
async fn malformed_responses_are_decode_errors() {
    let cases = vec![
        // noul out of spec: missing value
        json!({"type": "noul"}),
        // noul value must be a number, not a string
        json!({"type": "noul", "noul": "high"}),
        // choice missing probabilities
        json!({"type": "choice", "choice": "friendly", "confidence": 0.9}),
        // score missing legend
        json!({
            "type": "score",
            "score": 1.7,
            "confidence": 0.8,
            "probabilities": {"0": 1.0}
        }),
    ];
    for bad_answer in cases {
        let body = json!({
            "model": "jev-latest",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"q": bad_answer}
        });
        let (url, _requests) = spawn(move |_| Outcome::success(body.clone())).await;
        let client = client(&url, no_retries());
        let err = client.evaluate(evaluate_request()).await.unwrap_err();
        assert!(
            matches!(err, Error::Decode { .. }),
            "expected Decode for {bad_answer}, got {err:?}"
        );
    }
}

// ---------------------------------------------------------------------
// test_types.py::test_array_inputs + object states
// ---------------------------------------------------------------------
#[tokio::test]
async fn array_and_object_states_wire_through() {
    let (url, requests) = spawn(move |captured| Outcome::success(result_for(captured))).await;
    let client = client(&url, no_retries());

    // Array state via the direct State variant…
    let request = SystemRequest {
        state: State::Array(vec![json!("hello"), json!({"role": "user"})]),
        model: "client-model".into(),
        questions: HashMap::from_iter([("q".to_string(), question::noul("?"))]),
    };
    client.evaluate(request).await.unwrap();
    assert_eq!(
        requests.last().unwrap().body.as_ref().unwrap()["state"],
        json!(["hello", {"role": "user"}])
    );

    // …and via a raw Vec<Value> through system_one (IntoState).
    client
        .system_one(
            vec![json!("entry-1"), json!(2)],
            [("q", question::noul("?"))],
        )
        .await
        .unwrap();
    assert_eq!(
        requests.last().unwrap().body.as_ref().unwrap()["state"],
        json!(["entry-1", 2])
    );

    // Structured object state through system_one.
    let mut object = serde_json::Map::new();
    object.insert("user".into(), json!({"id": 7}));
    client
        .system_one(object, [("q", question::noul("?"))])
        .await
        .unwrap();
    assert_eq!(
        requests.last().unwrap().body.as_ref().unwrap()["state"],
        json!({"user": {"id": 7}})
    );
}

// ---------------------------------------------------------------------
// test_clients.py::test_rich_descriptions — object/array instructions
// ---------------------------------------------------------------------
#[tokio::test]
async fn rich_instructions_wire_through() {
    let (url, requests) = spawn(move |captured| Outcome::success(result_for(captured))).await;
    let client = client(&url, no_retries());

    let request = SystemRequest {
        state: State::Text("hello".into()),
        model: "client-model".into(),
        questions: HashMap::from_iter([
            (
                "obj".to_string(),
                question::noul(json!({"en": "Is it spam?", "de": "Ist das Spam?"})),
            ),
            (
                "arr".to_string(),
                question::score(
                    json!(["Rate the message:", "5 = excellent"]),
                    ["bad", "great"],
                ),
            ),
        ]),
    };
    client.evaluate(request).await.unwrap();

    let questions = requests.last().unwrap().body.as_ref().unwrap()["questions"].clone();
    assert_eq!(
        questions["obj"]["instructions"],
        json!({"en": "Is it spam?", "de": "Ist das Spam?"})
    );
    assert_eq!(
        questions["arr"]["instructions"],
        json!(["Rate the message:", "5 = excellent"])
    );
    assert_eq!(questions["arr"]["criteria"], json!(["bad", "great"]));
}

// ---------------------------------------------------------------------
// test_questions.py — discriminators are automatic, defaults omitted
// ---------------------------------------------------------------------
#[test]
fn question_discriminators_and_omitted_defaults() {
    let noul = serde_json::to_value(question::noul("Spam?")).unwrap();
    assert_eq!(noul, json!({"type": "noul", "instructions": "Spam?"}));

    let choice = serde_json::to_value(question::choice("Tone?", [("a", None)])).unwrap();
    assert_eq!(
        choice,
        json!({"type": "choice", "instructions": "Tone?", "criteria": {"a": null}})
    );

    let score = serde_json::to_value(question::score("Q?", ["bad", "ok"])).unwrap();
    assert_eq!(
        score,
        json!({"type": "score", "instructions": "Q?", "criteria": ["bad", "ok"]})
    );

    // Noul criteria serialize under the reserved "true"/"false" keys.
    let with_criteria =
        serde_json::to_value(question::noul_with("Q?", "yes means", "no means")).unwrap();
    assert_eq!(
        with_criteria["criteria"],
        json!({"true": "yes means", "false": "no means"})
    );
}
