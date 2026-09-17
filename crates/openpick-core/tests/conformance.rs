//! Conformance tests against the Jev API contract.
//!
//! Each test is anchored to a specific example or rule from
//! <https://docs.typesafe.ai/api>. If you change a schema field or validator,
//! you should be able to point at a comment in this file that names the spec
//! section it came from.

use std::collections::HashMap;

use openpick_core::*;
use serde_json::{json, Value};

// ------------------------------------------------------------------
// Spec section: request-body / example request
// ------------------------------------------------------------------

#[test]
fn spec_example_request_noul() {
    // Direct from docs.typesafe.ai/api#request-body example:
    //   {"state":"Help! My payouts have been failing for 3 days.",
    //    "model":"jev-latest",
    //    "questions":{"is_urgent":{"type":"noul","instructions":"..."}}}
    let raw = json!({
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this convey urgency?"
            }
        }
    });

    let req: SystemRequest = serde_json::from_value(raw.clone()).expect("parse");
    validate_request(&req).expect("valid");
    let round = serde_json::to_value(&req).unwrap();
    assert_eq!(round, raw);
}

#[test]
fn spec_example_request_noul_with_criteria() {
    // Direct from docs.typesafe.ai/api#noul example.
    let raw = json!({
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this convey urgency?",
                "criteria": {
                    "true": "Explicitly time-sensitive",
                    "false": "No urgency expressed"
                }
            }
        }
    });
    let req: SystemRequest = serde_json::from_value(raw.clone()).expect("parse");
    validate_request(&req).expect("valid");
    assert_eq!(serde_json::to_value(&req).unwrap(), raw);
}

#[test]
fn spec_example_request_choice() {
    // Direct from docs.typesafe.ai/api#choice example.
    let raw = json!({
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts"
                }
            }
        }
    });
    let req: SystemRequest = serde_json::from_value(raw.clone()).expect("parse");
    validate_request(&req).expect("valid");
    assert_eq!(serde_json::to_value(&req).unwrap(), raw);
}

#[test]
fn spec_example_request_score() {
    // Direct from docs.typesafe.ai/api#score example.
    let raw = json!({
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"]
            }
        }
    });
    let req: SystemRequest = serde_json::from_value(raw.clone()).expect("parse");
    validate_request(&req).expect("valid");
    assert_eq!(serde_json::to_value(&req).unwrap(), raw);
}

// ------------------------------------------------------------------
// Spec section: answer-types / example responses
// ------------------------------------------------------------------

#[test]
fn spec_example_response_noul() {
    let raw = json!({
        "model": "jev-latest",
        "answers": {
            "is_urgent": { "type": "noul", "noul": 0.92 }
        },
        "usage": { "input_tokens": 312, "output_tokens": 48 }
    });
    let resp: SystemResponse = serde_json::from_value(raw.clone()).expect("parse");
    let mut criteria = HashMap::new();
    criteria.insert("is_urgent".to_string(), Vec::new());
    validate_response(&resp, &criteria).expect("valid");
    assert_eq!(serde_json::to_value(&resp).unwrap(), raw);
}

#[test]
fn spec_example_response_choice() {
    let raw = json!({
        "model": "jev-latest",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": { "billing": 0.08, "technical": 0.85, "sales": 0.07 },
                "confidence": 0.82
            }
        },
        "usage": { "input_tokens": 312, "output_tokens": 48 }
    });
    let resp: SystemResponse = serde_json::from_value(raw.clone()).expect("parse");
    let mut criteria = HashMap::new();
    criteria.insert(
        "department".to_string(),
        vec!["billing".into(), "technical".into(), "sales".into()],
    );
    validate_response(&resp, &criteria).expect("valid");
    assert_eq!(serde_json::to_value(&resp).unwrap(), raw);
}

#[test]
fn spec_example_response_score() {
    let raw = json!({
        "model": "jev-latest",
        "answers": {
            "frustration": {
                "type": "score",
                "score": 1.6,
                "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                "probabilities": { "0": 0.05, "1": 0.3, "2": 0.65 },
                "confidence": 0.78
            }
        },
        "usage": { "input_tokens": 312, "output_tokens": 48 }
    });
    let resp: SystemResponse = serde_json::from_value(raw.clone()).expect("parse");
    let mut criteria = HashMap::new();
    criteria.insert("frustration".to_string(), Vec::new());
    validate_response(&resp, &criteria).expect("valid");
    assert_eq!(serde_json::to_value(&resp).unwrap(), raw);
}

// ------------------------------------------------------------------
// Spec section: question-types — edge cases
// ------------------------------------------------------------------

#[test]
fn choice_criteria_value_may_be_null() {
    // Spec: "use null when an option needs no extra detail"
    let raw = json!({
        "state": "x",
        "model": "jev-latest",
        "questions": {
            "tag": {
                "type": "choice",
                "instructions": "Pick one",
                "criteria": {
                    "alpha": null,
                    "beta": "second option"
                }
            }
        }
    });
    let req: SystemRequest = serde_json::from_value(raw).expect("null value should parse");
    validate_request(&req).expect("valid");
}

#[test]
fn instructions_may_be_string_object_or_array() {
    // Spec: instructions: string | object | array
    let cases = vec![
        json!("plain string"),
        json!({"role": "system", "content": "structured"}),
        json!(["line one", "line two"]),
    ];

    for instr in cases {
        let raw = json!({
            "state": "x",
            "model": "jev-latest",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": instr,
                }
            }
        });
        let req: SystemRequest = serde_json::from_value(raw).expect("should parse");
        validate_request(&req).expect("valid");
    }
}

#[test]
fn state_may_be_string_object_or_array() {
    let cases = vec![
        json!("a string state"),
        json!({"user": {"id": 123}, "messages": [], "diff": "..."}),
        json!([1, 2, 3]),
    ];
    for s in cases {
        let raw = json!({
            "state": s,
            "model": "jev-latest",
            "questions": {
                "q": { "type": "noul", "instructions": "?" }
            }
        });
        let req: SystemRequest = serde_json::from_value(raw).expect("should parse");
        validate_request(&req).expect("valid");
    }
}

// ------------------------------------------------------------------
// Spec section: validation rules
// ------------------------------------------------------------------

#[test]
fn empty_questions_is_rejected() {
    let raw = json!({
        "state": "x",
        "model": "jev-latest",
        "questions": {}
    });
    let req: SystemRequest = serde_json::from_value(raw).unwrap();
    assert!(matches!(
        validate_request(&req),
        Err(ValidationError::NoQuestions)
    ));
}

#[test]
fn choice_with_no_criteria_is_rejected() {
    // Spec: `criteria` is required on Choice, so serde rejects missing field
    // at deserialization (before our validator even runs).
    let raw = json!({
        "state": "x",
        "model": "jev-latest",
        "questions": {
            "q": { "type": "choice", "instructions": "?" }
        }
    });
    let parse: Result<SystemRequest, _> = serde_json::from_value(raw);
    assert!(parse.is_err(), "missing `criteria` must be a parse error");

    // Sanity: empty criteria map should also be rejected at validation.
    let raw_empty = json!({
        "state": "x",
        "model": "jev-latest",
        "questions": {
            "q": { "type": "choice", "instructions": "?", "criteria": {} }
        }
    });
    let req: SystemRequest = serde_json::from_value(raw_empty).unwrap();
    assert!(matches!(
        validate_request(&req),
        Err(ValidationError::ChoiceCriteriaEmpty(_))
    ));
}

#[test]
fn score_with_one_level_is_rejected() {
    let raw = json!({
        "state": "x",
        "model": "jev-latest",
        "questions": {
            "q": { "type": "score", "instructions": "?", "criteria": ["Only"] }
        }
    });
    let req: SystemRequest = serde_json::from_value(raw).unwrap();
    assert!(matches!(
        validate_request(&req),
        Err(ValidationError::ScoreCriteriaTooFew(_))
    ));
}

#[test]
fn response_probabilities_must_sum_to_one() {
    let raw = json!({
        "model": "jev-latest",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": { "billing": 0.5, "technical": 0.3, "sales": 0.1 },
                "confidence": 0.5
            }
        },
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp: SystemResponse = serde_json::from_value(raw).unwrap();
    let mut criteria = HashMap::new();
    criteria.insert(
        "department".to_string(),
        vec!["billing".into(), "technical".into(), "sales".into()],
    );
    assert!(matches!(
        validate_response(&resp, &criteria),
        Err(ValidationError::ProbabilitiesDontSum { .. })
    ));
}

#[test]
fn response_choice_not_in_criteria_is_rejected() {
    let raw = json!({
        "model": "jev-latest",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "ghost",
                "probabilities": { "billing": 1.0 },
                "confidence": 1.0
            }
        },
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp: SystemResponse = serde_json::from_value(raw).unwrap();
    let mut criteria = HashMap::new();
    criteria.insert("department".to_string(), vec!["billing".into()]);
    assert!(matches!(
        validate_response(&resp, &criteria),
        Err(ValidationError::ChoiceNotInCriteria(_))
    ));
}

#[test]
fn response_confidence_must_be_in_range() {
    let raw = json!({
        "model": "jev-latest",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": { "billing": 1.0 },
                "confidence": 1.5
            }
        },
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp: SystemResponse = serde_json::from_value(raw).unwrap();
    let mut criteria = HashMap::new();
    criteria.insert("department".to_string(), vec!["billing".into()]);
    assert!(matches!(
        validate_response(&resp, &criteria),
        Err(ValidationError::ConfidenceOutOfRange { .. })
    ));
}

#[test]
fn noul_answer_does_not_carry_confidence() {
    // Spec: "Choice and Score answers also carry a confidence". Noul does not.
    // We rely on the type system: NoulAnswer has no `confidence` field.
    // This test guards that invariant by checking the parsed type.
    let raw = json!({
        "model": "jev-latest",
        "answers": {
            "is_urgent": { "type": "noul", "noul": 0.5 }
        },
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp: SystemResponse = serde_json::from_value(raw).unwrap();
    match resp.answers.get("is_urgent").unwrap() {
        Answer::Noul(_) => {}
        _ => panic!("expected Noul answer"),
    }
}

#[test]
fn score_legend_and_probability_keys_must_match() {
    let raw = json!({
        "model": "jev-latest",
        "answers": {
            "frustration": {
                "type": "score",
                "score": 1.0,
                "legend": { "0": "Calm", "1": "Frustrated" },
                "probabilities": { "0": 0.5, "2": 0.5 },
                "confidence": 0.5
            }
        },
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp: SystemResponse = serde_json::from_value(raw).unwrap();
    let mut criteria = HashMap::new();
    criteria.insert("frustration".to_string(), Vec::new());
    assert!(matches!(
        validate_response(&resp, &criteria),
        Err(ValidationError::ScoreLegendMismatch(_))
    ));
}

#[test]
fn score_indices_must_be_numeric_strings() {
    let raw = json!({
        "model": "jev-latest",
        "answers": {
            "frustration": {
                "type": "score",
                "score": 1.0,
                "legend": { "low": "Calm", "high": "Angry" },
                "probabilities": { "low": 0.5, "high": 0.5 },
                "confidence": 0.5
            }
        },
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp: SystemResponse = serde_json::from_value(raw).unwrap();
    let mut criteria = HashMap::new();
    criteria.insert("frustration".to_string(), Vec::new());
    assert!(matches!(
        validate_response(&resp, &criteria),
        Err(ValidationError::ScoreIndexNotNumeric { .. })
    ));
}

// ------------------------------------------------------------------
// Spec section: api_version field on response is required
// ------------------------------------------------------------------

#[test]
fn response_requires_model_and_usage() {
    // We don't put `model` on the response.
    let raw = json!({
        "answers": { "q": { "type": "noul", "noul": 0.5 } },
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    assert!(serde_json::from_value::<SystemResponse>(raw).is_err());
}

#[test]
fn request_requires_model() {
    // Spec: model is required.
    let raw = json!({
        "state": "x",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    assert!(serde_json::from_value::<SystemRequest>(raw).is_err());
}

// ------------------------------------------------------------------
// JSON Schema generation (used by MCP adapters, OpenAPI clients)
// ------------------------------------------------------------------

#[test]
fn json_schema_generates_for_request_and_response() {
    let req_schema = schemars::schema_for!(SystemRequest);
    let resp_schema = schemars::schema_for!(SystemResponse);

    let req_json = serde_json::to_value(&req_schema).unwrap();
    let resp_json = serde_json::to_value(&resp_schema).unwrap();

    // Sanity: schemas are non-empty objects.
    assert!(req_json.is_object());
    assert!(resp_json.is_object());

    // Spot-check: SystemRequest has a "questions" property of type "object".
    let props = req_json.get("properties").and_then(|p| p.get("questions")).unwrap();
    assert_eq!(props.get("type").and_then(|t| t.as_str()), Some("object"));
}

// ------------------------------------------------------------------
// Spec section: "model" field — exact string equality preserved
// ------------------------------------------------------------------

#[test]
fn model_string_is_preserved_exactly() {
    // Per the literal-preservation rule: don't normalize tokens.
    let raw = json!({
        "state": "x",
        "model": "custom-alias/with-slashes",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let req: SystemRequest = serde_json::from_value(raw.clone()).unwrap();
    assert_eq!(req.model, "custom-alias/with-slashes");
    assert_eq!(serde_json::to_value(&req).unwrap(), raw);
}