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

// ------------------------------------------------------------------
// Wire-shape robustness for the streaming tagged deserializers
// ------------------------------------------------------------------

#[test]
fn question_fields_parse_in_any_order() {
    // `criteria` arriving before `type` exercises the buffered path of the
    // streaming deserializer; the result must match the canonical order.
    let canonical = json!({
        "type": "choice",
        "instructions": "Pick one",
        "criteria": {"alpha": null, "beta": "second"}
    });
    let reordered = json!({
        "criteria": {"alpha": null, "beta": "second"},
        "instructions": "Pick one",
        "type": "choice"
    });
    let a: Question = serde_json::from_value(canonical).expect("canonical order");
    let b: Question = serde_json::from_value(reordered).expect("reordered fields");
    assert_eq!(a, b);

    let noul_canonical = json!({"type": "noul", "instructions": "?", "criteria": null});
    let noul_reordered = json!({"criteria": null, "instructions": "?", "type": "noul"});
    assert_eq!(
        serde_json::from_value::<Question>(noul_canonical).unwrap(),
        serde_json::from_value::<Question>(noul_reordered).unwrap()
    );

    let score_reordered = json!({
        "criteria": ["low", "high"],
        "type": "score",
        "instructions": "Rate"
    });
    match serde_json::from_value::<Question>(score_reordered).unwrap() {
        Question::Score(s) => assert_eq!(s.criteria, vec!["low".to_string(), "high".to_string()]),
        other => panic!("expected score, got {other:?}"),
    }
}

#[test]
fn question_unknown_fields_are_ignored() {
    let raw = json!({
        "type": "noul",
        "instructions": "?",
        "future-extension": {"anything": true}
    });
    assert!(serde_json::from_value::<Question>(raw).is_ok());
}

#[test]
fn question_rejects_bad_tags_and_missing_fields() {
    let bad_tag = json!({"type": "mystery", "instructions": "?"});
    assert!(serde_json::from_value::<Question>(bad_tag).is_err());

    let missing_tag = json!({"instructions": "?"});
    assert!(serde_json::from_value::<Question>(missing_tag).is_err());

    let missing_instructions = json!({"type": "noul"});
    assert!(serde_json::from_value::<Question>(missing_instructions).is_err());

    let missing_choice_criteria = json!({"type": "choice", "instructions": "?"});
    assert!(serde_json::from_value::<Question>(missing_choice_criteria).is_err());

    let wrong_criteria_type = json!({"type": "choice", "instructions": "?", "criteria": 7});
    assert!(serde_json::from_value::<Question>(wrong_criteria_type).is_err());
}

#[test]
fn answer_fields_parse_in_any_order() {
    let canonical = json!({
        "type": "choice",
        "choice": "alpha",
        "probabilities": {"alpha": 0.7, "beta": 0.3},
        "confidence": 0.4
    });
    let reordered = json!({
        "confidence": 0.4,
        "probabilities": {"alpha": 0.7, "beta": 0.3},
        "choice": "alpha",
        "type": "choice"
    });
    assert_eq!(
        serde_json::from_value::<Answer>(canonical).unwrap(),
        serde_json::from_value::<Answer>(reordered).unwrap()
    );

    let score_reordered = json!({
        "confidence": 0.9,
        "probabilities": {"0": 1.0},
        "legend": {"0": "calm"},
        "score": 0.0,
        "type": "score"
    });
    match serde_json::from_value::<Answer>(score_reordered).unwrap() {
        Answer::Score(s) => {
            assert_eq!(s.score, 0.0);
            assert_eq!(s.confidence, 0.9);
        }
        other => panic!("expected score, got {other:?}"),
    }
}

#[test]
fn answer_rejects_bad_shapes() {
    assert!(serde_json::from_value::<Answer>(json!({"type": "nope"})).is_err());
    assert!(serde_json::from_value::<Answer>(json!({"type": "noul"})).is_err());
    assert!(serde_json::from_value::<Answer>(json!({
        "type": "noul", "noul": "not a number"
    }))
    .is_err());
}

#[test]
fn answer_ignores_fields_from_other_variants_in_any_order() {
    // Serde treats another variant's fields as unknown fields. Their shapes
    // and duplicate occurrences must not change the selected answer.
    for raw in [
        r#"{"type":"noul","noul":0.5,"confidence":null,"choice":{},"legend":false}"#,
        r#"{"confidence":null,"choice":{},"legend":false,"noul":0.5,"type":"noul"}"#,
        r#"{"confidence":null,"confidence":false,"type":"noul","noul":0.5}"#,
        r#"{"confidence":null,"type":"noul","noul":0.5,"confidence":false}"#,
        r#"{"type":"noul","noul":0.5,"confidence":null,"confidence":false}"#,
        r#"{"type":"choice","choice":"a","probabilities":{"a":1.0},"confidence":1.0,"noul":{},"score":false,"legend":null}"#,
        r#"{"noul":{},"score":false,"legend":null,"type":"choice","choice":"a","probabilities":{"a":1.0},"confidence":1.0}"#,
        r#"{"type":"score","score":0.0,"legend":{"0":"low"},"probabilities":{"0":1.0},"confidence":1.0,"choice":false,"noul":null}"#,
        r#"{"choice":false,"noul":null,"type":"score","score":0.0,"legend":{"0":"low"},"probabilities":{"0":1.0},"confidence":1.0}"#,
    ] {
        serde_json::from_str::<Answer>(raw).unwrap_or_else(|err| panic!("{raw}: {err}"));
    }
}

#[test]
fn answer_rejects_duplicate_variant_fields_in_any_order() {
    for raw in [
        r#"{"type":"noul","noul":0.5,"noul":0.8}"#,
        r#"{"noul":0.5,"noul":0.8,"type":"noul"}"#,
        r#"{"noul":0.5,"type":"noul","noul":0.8}"#,
        r#"{"confidence":0.5,"confidence":0.8,"type":"choice","choice":"a","probabilities":{"a":1.0}}"#,
    ] {
        let err = serde_json::from_str::<Answer>(raw).unwrap_err();
        assert!(err.to_string().contains("duplicate field"), "{raw}: {err}");
    }
}

#[test]
fn escaped_tag_keys_still_resolve_the_variant() {
    // A tag key written with JSON escapes (`\u0074ype`) must behave like `type`.
    let raw = r#"{"\u0074ype": "noul", "instructions": "?"}"#;
    let q: Question = serde_json::from_str(raw).expect("escaped tag key parses");
    assert!(matches!(q, Question::Noul(_)));
}

#[test]
fn state_and_answers_round_trip_through_wire_bytes() {
    let req = SystemRequest {
        state: State::Object(serde_json::Map::from_iter([(
            "k".to_string(),
            serde_json::json!([1, true, null, "s"]),
        )])),
        model: "mock".to_string(),
        questions: std::collections::HashMap::from_iter([
            (
                "q1".to_string(),
                Question::Noul(NoulQuestion {
                    instructions: serde_json::json!("?"),
                    criteria: Some(NoulCriteria {
                        r#true: "yes".to_string(),
                        r#false: "no".to_string(),
                    }),
                }),
            ),
            (
                "q2".to_string(),
                Question::Choice(ChoiceQuestion {
                    instructions: serde_json::json!(["a", "b"]),
                    criteria: std::collections::HashMap::from_iter([
                        ("x".to_string(), None),
                        ("y".to_string(), Some("why".to_string())),
                    ]),
                }),
            ),
        ]),
    };
    let bytes = serde_json::to_vec(&req).unwrap();
    let parsed: SystemRequest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(req, parsed);
}
