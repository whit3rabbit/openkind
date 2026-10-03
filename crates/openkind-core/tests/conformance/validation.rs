use std::collections::HashMap;

use openkind_core::*;
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
fn instructions_reject_unsupported_scalar_shapes_for_every_question_kind() {
    // The protocol permits strings, objects, and arrays, while the public
    // JSON-value alias also lets callers construct unsupported scalar values.
    for instructions in [json!(true), json!(false), json!(1), json!(0.5)] {
        for question in [
            json!({"type": "noul", "instructions": instructions}),
            json!({"type": "choice", "instructions": instructions, "criteria": {"a": null}}),
            json!({"type": "score", "instructions": instructions, "criteria": ["Low", "High"]}),
        ] {
            let request: SystemRequest = serde_json::from_value(json!({
                "state": "x", "model": "mock", "questions": {"q": question}
            }))
            .unwrap();
            assert!(matches!(
                validate_request(&request),
                Err(ValidationError::MissingInstructions(_))
            ));
            assert!(matches!(
                ResponseContract::from_request(&request),
                Err(ValidationError::MissingInstructions(_))
            ));
        }
    }
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
    let mut criteria = HashMap::default();
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
    let mut criteria = HashMap::default();
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
    let mut criteria = HashMap::default();
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
    let mut criteria = HashMap::default();
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
    let mut criteria = HashMap::default();
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

#[test]
fn non_finite_wire_floats_fail_closed_everywhere() {
    // Infinities never survive a JSON round trip (they encode as null), so
    // the guards are exercised on programmatically constructed responses —
    // exactly what a backend that computes inf/nan would hand to the
    // validator.
    let response = |answer: Answer| -> SystemResponse {
        let mut answers = HashMap::default();
        answers.insert("q".to_string(), answer);
        SystemResponse {
            model: "m".into(),
            answers,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
            },
        }
    };

    for noul in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let resp = response(Answer::Noul(NoulAnswer { noul }));
        assert!(
            matches!(
                validate_response(&resp, &HashMap::new()),
                Err(ValidationError::NoulOutOfRange { .. })
            ),
            "noul {noul} must be rejected"
        );
    }

    for score in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let mut probabilities = HashMap::default();
        probabilities.insert("0".to_string(), 1.0);
        let mut legend = HashMap::default();
        legend.insert("0".to_string(), "low".to_string());
        let resp = response(Answer::Score(ScoreAnswer {
            score,
            probabilities,
            legend,
            confidence: 0.5,
        }));
        assert!(
            matches!(
                validate_response(&resp, &HashMap::new()),
                Err(ValidationError::ScoreOutOfRange { .. })
            ),
            "score {score} must be rejected"
        );
    }

    for confidence in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let mut probabilities = HashMap::default();
        probabilities.insert("a".to_string(), 1.0);
        let resp = response(Answer::Choice(ChoiceAnswer {
            choice: "a".into(),
            probabilities,
            confidence,
        }));
        let mut criteria = HashMap::default();
        criteria.insert("q".to_string(), vec!["a".to_string()]);
        assert!(
            matches!(
                validate_response(&resp, &criteria),
                Err(ValidationError::ConfidenceOutOfRange { .. })
            ),
            "confidence {confidence} must be rejected"
        );
    }

    for probability in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let mut probabilities = HashMap::default();
        probabilities.insert("a".to_string(), probability);
        probabilities.insert("b".to_string(), 1.0 - probability.min(1.0));
        let resp = response(Answer::Choice(ChoiceAnswer {
            choice: "a".into(),
            probabilities,
            confidence: 0.5,
        }));
        let mut criteria = HashMap::default();
        criteria.insert("q".to_string(), vec!["a".to_string(), "b".to_string()]);
        assert!(
            matches!(
                validate_response(&resp, &criteria),
                Err(ValidationError::ProbabilityOutOfRange { .. })
            ),
            "probability {probability} must be rejected"
        );
    }
}

#[test]
fn probability_sums_honor_the_one_part_in_a_thousand_tolerance() {
    // Per-key range checks run first, so tolerance probes spread the drift
    // across two in-range keys whose sum crosses the 1e-3 boundary.
    let response = |first: f64, second: f64| -> SystemResponse {
        let mut probabilities = HashMap::default();
        probabilities.insert("a".to_string(), first);
        probabilities.insert("b".to_string(), second);
        let mut answers = HashMap::default();
        answers.insert(
            "q".to_string(),
            Answer::Choice(ChoiceAnswer {
                choice: "a".into(),
                probabilities,
                confidence: 0.5,
            }),
        );
        SystemResponse {
            model: "m".into(),
            answers,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
            },
        }
    };
    let criteria = || {
        let mut criteria = HashMap::default();
        criteria.insert("q".to_string(), vec!["a".to_string(), "b".to_string()]);
        criteria
    };

    for (first, second) in [
        (0.5, 0.5),
        (0.5, 0.4995),
        (0.5005, 0.5),
        (0.9991, 0.0),
        (0.5004, 0.5),
    ] {
        validate_response(&response(first, second), &criteria()).unwrap_or_else(|error| {
            panic!(
                "sum {} sits inside the 1e-3 tolerance: {error}",
                first + second
            )
        });
    }
    for (first, second) in [(0.5, 0.4985), (0.5015, 0.5), (0.0, 0.0)] {
        assert!(
            matches!(
                validate_response(&response(first, second), &criteria()),
                Err(ValidationError::ProbabilitiesDontSum { .. })
            ),
            "sum {} must be rejected",
            first + second
        );
    }

    // An empty distribution cannot sum to one.
    let mut probabilities: HashMap<String, f64> = HashMap::default();
    probabilities.insert("a".to_string(), 1.0);
    let mut answers = HashMap::default();
    answers.insert(
        "q".to_string(),
        Answer::Choice(ChoiceAnswer {
            choice: "a".into(),
            probabilities,
            confidence: 0.5,
        }),
    );
    let mut empty_answers = answers.clone();
    if let Answer::Choice(choice) = &mut empty_answers.get_mut("q").unwrap() {
        choice.probabilities.clear();
    }
    let empty = SystemResponse {
        model: "m".into(),
        answers: empty_answers,
        usage: Usage {
            input_tokens: 1,
            output_tokens: 1,
        },
    };
    assert!(matches!(
        validate_response(&empty, &criteria()),
        Err(ValidationError::ProbabilityKeysMismatch(_))
            | Err(ValidationError::ProbabilitiesDontSum { .. })
    ));
}

#[test]
fn score_at_the_top_level_boundary_validates() {
    // `score == max_level` is the inclusive upper boundary; only values
    // beyond the highest level key are out of range.
    let raw = json!({
        "model": "m",
        "answers": {"q": {"type": "score", "score": 2.0,
                          "probabilities": {"2": 1.0}, "legend": {"2": "high"},
                          "confidence": 1.0}},
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let resp: SystemResponse = serde_json::from_value(raw).unwrap();
    validate_response(&resp, &HashMap::new()).expect("the highest level is a valid score");
}

#[test]
fn response_contract_captures_request_bounds_directly() {
    let request = |count: usize| -> SystemRequest {
        let mut questions = HashMap::default();
        for index in 0..count {
            questions.insert(
                format!("q{index}"),
                Question::Noul(NoulQuestion {
                    instructions: json!("?"),
                    criteria: None,
                }),
            );
        }
        SystemRequest {
            state: State::Text("x".into()),
            model: "m".into(),
            questions,
        }
    };

    assert!(matches!(
        ResponseContract::from_request(&request(0)),
        Err(ValidationError::NoQuestions)
    ));
    assert!(matches!(
        ResponseContract::from_request(&request(MAX_QUESTIONS_PER_REQUEST + 1)),
        Err(ValidationError::TooManyQuestions { .. })
    ));
    ResponseContract::from_request(&request(MAX_QUESTIONS_PER_REQUEST))
        .expect("the limit boundary itself is accepted");
}

#[test]
fn answer_type_mismatch_reports_both_directions() {
    let request = |kind: &str| -> SystemRequest {
        let question = match kind {
            "noul" => json!({"type": "noul", "instructions": "?"}),
            "score" => json!({"type": "score", "instructions": "?", "criteria": ["a", "b"]}),
            _ => unreachable!(),
        };
        serde_json::from_value(json!({
            "state": "x", "model": "m",
            "questions": {"q": question}
        }))
        .unwrap()
    };
    let response = |kind: &str| -> SystemResponse {
        let answer = match kind {
            "noul" => json!({"type": "noul", "noul": 0.5}),
            "choice" => json!({"type": "choice", "choice": "a",
                             "probabilities": {"a": 1.0}, "confidence": 0.5}),
            "score" => json!({"type": "score", "score": 0.0,
                            "probabilities": {"0": 1.0}, "legend": {"0": "a"},
                            "confidence": 0.5}),
            _ => unreachable!(),
        };
        serde_json::from_value(json!({
            "model": "m",
            "answers": {"q": answer},
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .unwrap()
    };

    for (question_kind, answer_kind) in [("noul", "choice"), ("noul", "score"), ("score", "noul")] {
        let error = validate_response_for_request(&response(answer_kind), &request(question_kind))
            .expect_err("mismatched answer types must fail");
        assert!(
            matches!(&error, ValidationError::AnswerTypeMismatch { expected, actual, .. }
                if *expected == question_kind && *actual == answer_kind),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn state_array_serialization_round_trips() {
    let state = State::Array(vec![json!(1), json!("two"), json!(null)]);
    let encoded = serde_json::to_value(&state).unwrap();
    assert_eq!(encoded, json!([1, "two", null]));
    let decoded: State = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, state);
}
