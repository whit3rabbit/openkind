use axum::http::StatusCode;
use serde_json::{json, Value};

use super::helpers::{body_bytes, post_systemone};

// --- Types: Common (JSONContent arbitrary JSON values, structured instructions) ---

#[tokio::test]
async fn common_types_state_supports_deep_nesting_and_nulls() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/common
    // JSONContent: dict[str, Any] | list[Any] | str | int | float | bool | None
    let body = json!({
        "state": {
            "session": null,
            "cart": {
                "items": [
                    {"id": 1, "price": 29.99, "in_stock": true},
                    {"id": 2, "price": null, "notes": ["backorder", null, 42]}
                ],
                "discount": null
            },
            "user_authenticated": false
        },
        "model": "mock",
        "questions": {
            "has_discount": { "type": "noul", "instructions": "Is there a discount?" }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["answers"]["has_discount"]["noul"].is_number());
}

#[tokio::test]
async fn common_types_instructions_as_structured_object() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/common
    // JSONContent can be a structured instructions object
    let body = json!({
        "state": "Customer support ticket #4432",
        "model": "mock",
        "questions": {
            "policy_check": {
                "type": "noul",
                "instructions": {
                    "role": "tier-3 auditor",
                    "guidelines": ["Check SLA compliance", "Verify identity"],
                    "threshold_hours": 24
                }
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["answers"]["policy_check"]["noul"].is_number());
}

// --- Types: Questions (NoulModel, ChoiceModel, ScoreModel) ---

#[tokio::test]
async fn question_types_noul_with_explicit_true_and_false_criteria() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/questions
    // NoulModel: criteria can be dict[bool, str] mapping true and false to descriptions
    let body = json!({
        "state": "User requested account deletion.",
        "model": "mock",
        "questions": {
            "destructive": {
                "type": "noul",
                "instructions": "Is this action destructive to user data?",
                "criteria": {
                    "true": "Irreversible loss of user account or records.",
                    "false": "Standard operational change, fully recoverable."
                }
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let noul = v["answers"]["destructive"]["noul"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&noul));
}

#[tokio::test]
async fn question_types_noul_without_criteria() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/questions
    // NoulModel: criteria is optional (None)
    let body = json!({
        "state": "The door is locked.",
        "model": "mock",
        "questions": {
            "locked": {
                "type": "noul",
                "instructions": "Is the door locked?"
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["answers"]["locked"]["noul"].is_number());
}

#[tokio::test]
async fn question_types_choice_with_mixed_string_and_null_criteria() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/questions
    // ChoiceModel: criteria dict[str, str | None]
    let body = json!({
        "state": "Alert triggered: CPU at 98%",
        "model": "mock",
        "questions": {
            "severity": {
                "type": "choice",
                "instructions": "What is the severity level?",
                "criteria": {
                    "low": null,
                    "medium": "Normal operational spike",
                    "high": null,
                    "critical": "System is degraded or unresponsive"
                }
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let choice_ans = &v["answers"]["severity"];
    let choice = choice_ans["choice"].as_str().unwrap();
    assert!(["low", "medium", "high", "critical"].contains(&choice));
    let probs = choice_ans["probabilities"].as_object().unwrap();
    assert_eq!(probs.len(), 4);
    assert!(choice_ans["confidence"].is_number());
}

#[tokio::test]
async fn question_types_score_ordered_rubric_levels() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/questions
    // In SDK client: ScoreModel(criteria={0: "Unsatisfactory", 1: "Developing", ...})
    // On the wire: criteria is an ordered array of level descriptions [str, str, ...]
    let body = json!({
        "state": "Essay submission for grading.",
        "model": "mock",
        "questions": {
            "rubric_score": {
                "type": "score",
                "instructions": "Grade the essay according to the rubric.",
                "criteria": [
                    "Unsatisfactory: failing to meet core requirements",
                    "Developing: partially meets requirements",
                    "Proficient: meets all rubric requirements",
                    "Advanced: exceeds rubric requirements"
                ]
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let score_ans = &v["answers"]["rubric_score"];
    let score = score_ans["score"].as_f64().unwrap();
    assert!((0.0..=3.0).contains(&score));
    assert!(score_ans["legend"].is_object());
    assert!(score_ans["probabilities"].is_object());
    assert!(score_ans["confidence"].is_number());
    // SDK deserializes legend string keys ("0", "1", "2", "3") to int keys
    assert_eq!(
        score_ans["legend"]["0"],
        "Unsatisfactory: failing to meet core requirements"
    );
    assert_eq!(
        score_ans["legend"]["3"],
        "Advanced: exceeds rubric requirements"
    );
}

// --- Types: Responses (SystemOneResponse structure, UsageInfo, Answer models) ---

#[tokio::test]
async fn response_structure_matches_sdk_system_one_response_model() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/responses
    // SystemOneResponse:
    //   request_id: str (from header)
    //   model: str
    //   usage: UsageInfo { input_tokens: int, output_tokens: int }
    //   answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    let body = json!({
        "state": "Response structure verification",
        "model": "mock",
        "questions": {
            "noul_q": { "type": "noul", "instructions": "noul check" },
            "choice_q": {
                "type": "choice",
                "instructions": "choice check",
                "criteria": { "a": "alpha", "b": "beta" }
            },
            "score_q": {
                "type": "score",
                "instructions": "score check",
                "criteria": ["low", "high"]
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);

    // 1. Header request_id matches UUID format
    let req_id = resp
        .headers()
        .get("x-typesafe-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(req_id.split('-').count(), 5);

    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();

    // 2. model is string
    assert_eq!(v["model"], "mock");

    // 3. usage has positive integers
    let input_tokens = v["usage"]["input_tokens"].as_u64().unwrap();
    let output_tokens = v["usage"]["output_tokens"].as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // 4. answers contains each question key
    let answers = v["answers"].as_object().unwrap();
    assert_eq!(answers.len(), 3);

    // 5. NoulAnswer strictly contains `noul` (and NOT `confidence`)
    let noul_ans = answers["noul_q"].as_object().unwrap();
    assert!(noul_ans.contains_key("noul"));
    assert!(
        !noul_ans.contains_key("confidence"),
        "SDK NoulAnswer has no confidence"
    );

    // 6. ChoiceAnswer contains `choice`, `probabilities`, `confidence`
    let choice_ans = answers["choice_q"].as_object().unwrap();
    assert!(choice_ans.contains_key("choice"));
    assert!(choice_ans.contains_key("probabilities"));
    assert!(choice_ans.contains_key("confidence"));

    // 7. ScoreAnswer contains `score`, `legend`, `probabilities`, `confidence`
    let score_ans = answers["score_q"].as_object().unwrap();
    assert!(score_ans.contains_key("score"));
    assert!(score_ans.contains_key("legend"));
    assert!(score_ans.contains_key("probabilities"));
    assert!(score_ans.contains_key("confidence"));
}
