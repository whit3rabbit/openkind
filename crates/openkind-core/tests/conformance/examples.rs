use std::collections::HashMap;

use openkind_core::*;
use serde_json::json;

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
    let mut criteria = HashMap::default();
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
    let mut criteria = HashMap::default();
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
    let mut criteria = HashMap::default();
    criteria.insert("frustration".to_string(), Vec::new());
    validate_response(&resp, &criteria).expect("valid");
    assert_eq!(serde_json::to_value(&resp).unwrap(), raw);
}
