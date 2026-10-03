use axum::http::StatusCode;
use serde_json::{json, Value};

use super::helpers::{body_bytes, post_systemone};

#[tokio::test]
async fn system_one_accepts_plain_text_state() {
    // SDK example: `client.system_one(state="I was charged twice.", ...)`
    let body = json!({
        "state": "I was charged twice. Please help.",
        "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "Is this about billing?" }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["model"], "mock");
    assert!(v["answers"]["billing"]["noul"].is_number());
}

#[tokio::test]
async fn system_one_accepts_object_state() {
    // Spec: state may be string | object | array.
    let body = json!({
        "state": { "document": "x", "messages": [] },
        "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "?" }
        }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_accepts_array_state() {
    let body = json!({
        "state": ["turn 1", "turn 2"],
        "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "?" }
        }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_returns_noul_answer_shape() {
    // SDK type: NoulAnswer { noul: float }
    let body = json!({
        "state": "x", "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let ans = &v["answers"]["q"];
    assert_eq!(ans["type"], "noul");
    assert!(ans["noul"].is_number());
    assert!(
        ans.get("confidence").is_none(),
        "noul answer MUST NOT have confidence"
    );
    let noul = ans["noul"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&noul));
}

#[tokio::test]
async fn system_one_returns_choice_answer_shape() {
    // SDK type: ChoiceAnswer { choice, probabilities, confidence }
    let body = json!({
        "state": "x", "model": "mock",
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "?",
                "criteria": { "billing": "pay", "tech": "bugs" }
            }
        }
    });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let ans = &v["answers"]["dept"];
    assert_eq!(ans["type"], "choice");
    assert!(ans["choice"].is_string());
    assert!(ans["probabilities"].is_object());
    assert!(ans["confidence"].is_number());
    // probabilities sum to 1.
    let sum: f64 = ans["probabilities"]
        .as_object()
        .unwrap()
        .values()
        .map(|x| x.as_f64().unwrap())
        .sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "probabilities must sum to 1, got {sum}"
    );
}

#[tokio::test]
async fn system_one_returns_score_answer_shape() {
    // SDK type: ScoreAnswer { score, legend, probabilities, confidence }
    let body = json!({
        "state": "x", "model": "mock",
        "questions": {
            "frust": {
                "type": "score",
                "instructions": "?",
                "criteria": ["Calm", "Frustrated", "Angry"]
            }
        }
    });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let ans = &v["answers"]["frust"];
    assert_eq!(ans["type"], "score");
    assert!(ans["score"].is_number());
    assert!(ans["legend"].is_object());
    assert!(ans["probabilities"].is_object());
    assert!(ans["confidence"].is_number());
    // legend and probabilities share the same keys.
    let lkeys: std::collections::HashSet<&str> = ans["legend"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let pkeys: std::collections::HashSet<&str> = ans["probabilities"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(lkeys, pkeys, "legend and probabilities must share keys");
}

#[tokio::test]
async fn system_one_choice_criteria_value_may_be_null() {
    // Spec: "use null when an option needs no extra detail"
    let body = json!({
        "state": "x", "model": "mock",
        "questions": {
            "tag": {
                "type": "choice",
                "instructions": "?",
                "criteria": { "alpha": null, "beta": "second" }
            }
        }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_instructions_may_be_string_object_or_array() {
    for instr in [
        json!("string"),
        json!({"role": "system", "content": "structured"}),
        json!(["line 1", "line 2"]),
    ] {
        let body = json!({
            "state": "x", "model": "mock",
            "questions": { "q": { "type": "noul", "instructions": instr } }
        });
        let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
        assert_eq!(status, StatusCode::OK, "instructions={instr}");
    }
}

#[tokio::test]
async fn system_one_returns_usage_block() {
    // SDK type: Usage { input_tokens, output_tokens } — both optional ints.
    let body = json!({
        "state": "x", "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["usage"].is_object());
    assert!(v["usage"]["input_tokens"].is_number());
    assert!(v["usage"]["output_tokens"].is_number());
}

#[tokio::test]
async fn system_one_default_model_jev_latest_works() {
    // Spec constant: DEFAULT_MODEL = 'jev-latest'. SDK sends this
    // when the user doesn't override.
    let body = json!({
        "state": "x",
        "model": "jev-latest",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_accepts_question_dict_with_type_field() {
    // SDK example uses both Question objects and dicts with `type` field.
    let body = json!({
        "state": "x", "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "?" },
            "tone": {
                "type": "choice",
                "instructions": "?",
                "criteria": { "calm": null, "angry": null }
            },
            "urgency": {
                "type": "score",
                "instructions": "?",
                "criteria": ["can wait", "this week", "today"]
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["answers"].as_object().unwrap().len(), 3);
}

#[tokio::test]
async fn extra_body_fields_are_ignored_not_rejected() {
    // The SDK sends `extra_body` to merge in extra top-level fields.
    // We don't implement merge semantics in Phase 1, but the server
    // shouldn't reject unknown top-level fields by default — strict
    // mode would need an opt-in flag.
    let body = json!({
        "state": "x", "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } },
        "metadata": { "trace_id": "abc" }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    // Note: serde's default is to ignore unknown fields. If we ever
    // tighten this, this test guards the SDK compatibility.
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_alias_route_works() {
    // Both POST /v1/systemone (canonical per spec) and /v1/system_one (alias) work identically.
    let body = json!({
        "state": "Test state",
        "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status, resp) = post_systemone("/v1/system_one", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["model"], "mock");
    assert!(v["answers"]["q"]["noul"].is_number());
}

#[tokio::test]
async fn choice_with_none_option_reports_its_semantic_none_mass() {
    // AGENTS.md invariant 7: native Choice questions must include a
    // non-empty `__none__` option and report its semantic-none probability
    // mass on the wire — exercised here through the HTTP contract.
    let body = json!({
        "state": "x", "model": "mock",
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "?",
                "criteria": {
                    "billing": "pay",
                    "tech": "bugs",
                    "__none__": "none of the offered departments apply"
                }
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let answer = &v["answers"]["dept"];
    assert_eq!(answer["type"], "choice");
    let probabilities = answer["probabilities"].as_object().expect("probabilities");
    let none = probabilities
        .get("__none__")
        .and_then(Value::as_f64)
        .expect("the semantic-none key must carry a probability");
    assert!(
        (0.0..=1.0).contains(&none),
        "semantic-none mass must be a probability, got {none}"
    );
    let sum: f64 = probabilities
        .values()
        .map(Value::as_f64)
        .map(Option::unwrap_or_default)
        .sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "probabilities must sum to 1, got {sum}"
    );
}
