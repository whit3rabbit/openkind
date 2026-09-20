use std::collections::HashMap;

use opendecision_core::*;
use serde_json::json;

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
