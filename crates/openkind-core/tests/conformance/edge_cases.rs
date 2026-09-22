use openkind_core::*;
use serde_json::json;

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
