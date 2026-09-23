//! Unit tests for request validation rules in openkind-core.

use std::collections::HashMap;

use crate::error::types::ValidationError;
use crate::error::validate::{validate_request, MAX_CRITERIA_OPTIONS, MAX_QUESTIONS_PER_REQUEST};
use crate::question::{ChoiceQuestion, NoulCriteria, NoulQuestion, Question, ScoreQuestion};
use crate::request::SystemRequest;
use crate::state::State;

#[test]
fn validate_request_happy_path_all_three_types() {
    let mut questions = HashMap::default();
    questions.insert(
        "noul".to_string(),
        Question::Noul(NoulQuestion {
            instructions: serde_json::json!("Is this fine?"),
            criteria: Some(NoulCriteria {
                r#true: "yes".into(),
                r#false: "no".into(),
            }),
        }),
    );
    let mut criteria = HashMap::default();
    criteria.insert("opt1".into(), Some("First".into()));
    criteria.insert("opt2".into(), None);
    questions.insert(
        "choice".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick one"),
            criteria,
        }),
    );
    questions.insert(
        "score".to_string(),
        Question::Score(ScoreQuestion {
            instructions: serde_json::json!("Score it"),
            criteria: vec!["Low".into(), "High".into()],
        }),
    );
    let req = SystemRequest {
        state: State::Text("test state".into()),
        model: "mock".into(),
        questions,
    };
    assert!(validate_request(&req).is_ok());
}

#[test]
fn validate_request_rejects_empty_score_level() {
    let mut questions = HashMap::default();
    questions.insert(
        "rating".to_string(),
        Question::Score(ScoreQuestion {
            instructions: serde_json::json!("Rate this"),
            criteria: vec!["Level 1".into(), "".into()],
        }),
    );
    let req = SystemRequest {
        state: State::Text("sample".into()),
        model: "mock".into(),
        questions,
    };
    let err = validate_request(&req).unwrap_err();
    assert!(matches!(
        err,
        ValidationError::ScoreLevelEmpty { index: 1, .. }
    ));
}

#[test]
fn validate_request_rejects_empty_noul_criteria_strings() {
    let mut questions = HashMap::default();
    questions.insert(
        "q1".to_string(),
        Question::Noul(NoulQuestion {
            instructions: serde_json::json!("Is this valid?"),
            criteria: Some(NoulCriteria {
                r#true: "".into(),
                r#false: "no".into(),
            }),
        }),
    );
    let req = SystemRequest {
        state: State::Text("sample".into()),
        model: "mock".into(),
        questions,
    };
    assert!(matches!(
        validate_request(&req).unwrap_err(),
        ValidationError::NoulTrueEmpty(_)
    ));

    let mut questions2 = HashMap::default();
    questions2.insert(
        "q2".to_string(),
        Question::Noul(NoulQuestion {
            instructions: serde_json::json!("Is this valid?"),
            criteria: Some(NoulCriteria {
                r#true: "yes".into(),
                r#false: "".into(),
            }),
        }),
    );
    let req2 = SystemRequest {
        state: State::Text("sample".into()),
        model: "mock".into(),
        questions: questions2,
    };
    assert!(matches!(
        validate_request(&req2).unwrap_err(),
        ValidationError::NoulFalseEmpty(_)
    ));
}

#[test]
fn validate_request_rejects_missing_instructions_shapes() {
    let empty_shapes = vec![
        serde_json::json!(""),
        serde_json::json!({}),
        serde_json::json!([]),
        serde_json::Value::Null,
    ];
    for empty_instr in empty_shapes {
        let mut questions = HashMap::default();
        questions.insert(
            "q".to_string(),
            Question::Noul(NoulQuestion {
                instructions: empty_instr.clone(),
                criteria: None,
            }),
        );
        let req = SystemRequest {
            state: State::Text("sample".into()),
            model: "mock".into(),
            questions,
        };
        assert!(
            matches!(
                validate_request(&req).unwrap_err(),
                ValidationError::MissingInstructions(_)
            ),
            "failed to reject missing instructions shape: {:?}",
            empty_instr
        );
    }
}

#[test]
fn validate_request_rejects_excessive_questions_and_criteria() {
    let mut questions = HashMap::default();
    for i in 0..=MAX_QUESTIONS_PER_REQUEST {
        questions.insert(
            format!("q_{i}"),
            Question::Noul(NoulQuestion {
                instructions: serde_json::json!("Test"),
                criteria: None,
            }),
        );
    }
    let req = SystemRequest {
        state: State::Text("state".into()),
        model: "mock".into(),
        questions,
    };
    assert!(matches!(
        validate_request(&req).unwrap_err(),
        ValidationError::TooManyQuestions { .. }
    ));
}

#[test]
fn validate_request_rejects_excessive_choice_criteria() {
    let mut questions = HashMap::default();
    questions.insert(
        "big_choice".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick one"),
            criteria: (0..=MAX_CRITERIA_OPTIONS)
                .map(|i| (format!("opt_{i}"), None))
                .collect(),
        }),
    );
    let req = SystemRequest {
        state: State::Text("state".into()),
        model: "mock".into(),
        questions,
    };
    let err = validate_request(&req).unwrap_err();
    assert!(matches!(
        err,
        ValidationError::TooManyCriteriaOptions {
            max: MAX_CRITERIA_OPTIONS,
            ..
        }
    ));
}

#[test]
fn validate_request_rejects_excessive_score_criteria() {
    let mut questions = HashMap::default();
    questions.insert(
        "big_score".to_string(),
        Question::Score(ScoreQuestion {
            instructions: serde_json::json!("Rate it"),
            criteria: (0..=MAX_CRITERIA_OPTIONS)
                .map(|i| format!("level_{i}"))
                .collect(),
        }),
    );
    let req = SystemRequest {
        state: State::Text("state".into()),
        model: "mock".into(),
        questions,
    };
    let err = validate_request(&req).unwrap_err();
    assert!(matches!(
        err,
        ValidationError::TooManyCriteriaOptions {
            max: MAX_CRITERIA_OPTIONS,
            ..
        }
    ));
}
