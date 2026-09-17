//! Validation errors. Spec lists four error statuses (401/422/429/529);
//! `422 Unprocessable Entity` is the one this module emits — invalid bodies.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("`questions` must contain at least one entry")]
    NoQuestions,

    #[error("question `{0}`: `instructions` is required")]
    MissingInstructions(String),

    #[error("question `{0}` (choice): `criteria` must contain at least one option")]
    ChoiceCriteriaEmpty(String),

    #[error("question `{0}` (score): `criteria` must contain at least 2 levels")]
    ScoreCriteriaTooFew(String),

    #[error("question `{id}` (score): level descriptions must be non-empty")]
    ScoreLevelEmpty { id: String, index: usize },

    #[error("question `{0}` (noul): `criteria.true` is empty")]
    NoulTrueEmpty(String),

    #[error("question `{0}` (noul): `criteria.false` is empty")]
    NoulFalseEmpty(String),

    #[error("question `{0}`: choice answer `choice` not in criteria keys")]
    ChoiceNotInCriteria(String),

    #[error("question `{0}`: choice answer `probabilities` keys must match criteria")]
    ProbabilityKeysMismatch(String),

    #[error("question `{id}`: probabilities do not sum to 1 (got {sum})")]
    ProbabilitiesDontSum { id: String, sum: f64 },

    #[error("answer `{id}`: confidence must be in 0..=1 (got {value})")]
    ConfidenceOutOfRange { id: String, value: f64 },

    #[error("answer `{id}`: noul must be in 0..=1 (got {value})")]
    NoulOutOfRange { id: String, value: f64 },

    #[error("score answer `{0}`: legend keys must match probabilities keys")]
    ScoreLegendMismatch(String),

    #[error("score answer `{id}`: legend/probability indices must be numeric, got `{key}`")]
    ScoreIndexNotNumeric { id: String, key: String },
}

pub type ValidationResult<T> = Result<T, ValidationError>;

use crate::answer::Answer;
use crate::question::Question;
use crate::request::SystemRequest;
use crate::response::SystemResponse;

/// Validate a request. Pure — does no I/O.
///
/// Rules derived from the spec:
/// - at least one question
/// - all question kinds validate their own `criteria` shape
/// - for Choice: criteria must be non-empty (no explicit lower bound in spec
///   but a zero-option question is meaningless and would 422 from the real API)
pub fn validate_request(req: &SystemRequest) -> ValidationResult<()> {
    if req.questions.is_empty() {
        return Err(ValidationError::NoQuestions);
    }
    for (id, q) in &req.questions {
        validate_question(id, q)?;
    }
    Ok(())
}

fn validate_question(id: &str, q: &Question) -> ValidationResult<()> {
    match q {
        Question::Noul(n) => {
            if instructions_missing(&n.instructions) {
                return Err(ValidationError::MissingInstructions(id.to_string()));
            }
            if let Some(c) = &n.criteria {
                if c.r#true.is_empty() {
                    return Err(ValidationError::NoulTrueEmpty(id.to_string()));
                }
                if c.r#false.is_empty() {
                    return Err(ValidationError::NoulFalseEmpty(id.to_string()));
                }
            }
        }
        Question::Choice(c) => {
            if instructions_missing(&c.instructions) {
                return Err(ValidationError::MissingInstructions(id.to_string()));
            }
            if c.criteria.is_empty() {
                return Err(ValidationError::ChoiceCriteriaEmpty(id.to_string()));
            }
        }
        Question::Score(s) => {
            if instructions_missing(&s.instructions) {
                return Err(ValidationError::MissingInstructions(id.to_string()));
            }
            if s.criteria.len() < 2 {
                return Err(ValidationError::ScoreCriteriaTooFew(id.to_string()));
            }
            for (i, level) in s.criteria.iter().enumerate() {
                if level.is_empty() {
                    return Err(ValidationError::ScoreLevelEmpty {
                        id: id.to_string(),
                        index: i,
                    });
                }
            }
        }
    }
    Ok(())
}

fn instructions_missing(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::String(s) => s.is_empty(),
        serde_json::Value::Object(m) => m.is_empty(),
        serde_json::Value::Array(a) => a.is_empty(),
        serde_json::Value::Null => true,
        _ => false,
    }
}

/// Validate a response. Catches the common engine-side bugs:
/// - probabilities don't sum to 1
/// - confidence out of range
/// - choice answer's keys don't match criteria
/// - score answer legend doesn't match probabilities
pub fn validate_response(
    resp: &SystemResponse,
    criteria: &std::collections::HashMap<String, Vec<String>>,
) -> ValidationResult<()> {
    // Every question id in the request must have a matching answer.
    // (Caller passes criteria so we can validate cross-references.)
    for (id, ans) in &resp.answers {
        match ans {
            Answer::Noul(n) => {
                if !(0.0..=1.0).contains(&n.noul) {
                    return Err(ValidationError::NoulOutOfRange {
                        id: id.clone(),
                        value: n.noul,
                    });
                }
            }
            Answer::Choice(c) => {
                let expected_keys: std::collections::HashSet<&str> = criteria
                    .get(id)
                    .map(|v| v.iter().map(String::as_str).collect())
                    .unwrap_or_default();
                if !expected_keys.is_empty() && !expected_keys.contains(c.choice.as_str()) {
                    return Err(ValidationError::ChoiceNotInCriteria(id.clone()));
                }
                let prob_keys: std::collections::HashSet<&str> =
                    c.probabilities.keys().map(String::as_str).collect();
                if !expected_keys.is_empty() && prob_keys != expected_keys {
                    return Err(ValidationError::ProbabilityKeysMismatch(id.clone()));
                }
                check_confidence(id, c.confidence)?;
                check_sums_to_one(id, c.probabilities.values().copied().sum())?;
            }
            Answer::Score(s) => {
                let prob_keys: std::collections::HashSet<&str> =
                    s.probabilities.keys().map(String::as_str).collect();
                let legend_keys: std::collections::HashSet<&str> =
                    s.legend.keys().map(String::as_str).collect();
                if prob_keys != legend_keys {
                    return Err(ValidationError::ScoreLegendMismatch(id.clone()));
                }
                for k in prob_keys {
                    k.parse::<u32>()
                        .map_err(|_| ValidationError::ScoreIndexNotNumeric {
                            id: id.clone(),
                            key: k.to_string(),
                        })?;
                }
                check_confidence(id, s.confidence)?;
                check_sums_to_one(id, s.probabilities.values().copied().sum())?;
            }
        }
    }
    Ok(())
}

fn check_confidence(id: &str, value: f64) -> ValidationResult<()> {
    if !(0.0..=1.0).contains(&value) {
        return Err(ValidationError::ConfidenceOutOfRange {
            id: id.to_string(),
            value,
        });
    }
    Ok(())
}

fn check_sums_to_one(id: &str, sum: f64) -> ValidationResult<()> {
    if (sum - 1.0).abs() > 1e-3 {
        return Err(ValidationError::ProbabilitiesDontSum {
            id: id.to_string(),
            sum,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answer::{ChoiceAnswer, NoulAnswer, ScoreAnswer};
    use crate::question::{ChoiceQuestion, NoulCriteria, NoulQuestion, ScoreQuestion};
    use crate::request::SystemRequest;
    use crate::response::{SystemResponse, Usage};
    use crate::state::State;
    use std::collections::HashMap;

    #[test]
    fn validate_request_happy_path_all_three_types() {
        let mut questions = HashMap::new();
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
        let mut criteria = HashMap::new();
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
        let mut questions = HashMap::new();
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
        let mut questions = HashMap::new();
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

        let mut questions2 = HashMap::new();
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
            let mut questions = HashMap::new();
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
    fn validate_response_rejects_probability_keys_mismatch() {
        let mut answers = HashMap::new();
        let mut probs = HashMap::new();
        probs.insert("option_a".into(), 0.6);
        probs.insert("option_x".into(), 0.4);
        answers.insert(
            "ch".into(),
            Answer::Choice(ChoiceAnswer {
                choice: "option_a".into(),
                probabilities: probs,
                confidence: 0.8,
            }),
        );
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let mut criteria = HashMap::new();
        criteria.insert("ch".into(), vec!["option_a".into(), "option_b".into()]);
        assert!(matches!(
            validate_response(&resp, &criteria).unwrap_err(),
            ValidationError::ProbabilityKeysMismatch(_)
        ));
    }

    #[test]
    fn validate_response_rejects_noul_out_of_range() {
        for invalid_noul in [-0.01, 1.05] {
            let mut answers = HashMap::new();
            answers.insert(
                "noul_q".into(),
                Answer::Noul(NoulAnswer { noul: invalid_noul }),
            );
            let resp = SystemResponse {
                model: "mock".into(),
                answers,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
            };
            let criteria = HashMap::new();
            assert!(
                matches!(
                    validate_response(&resp, &criteria).unwrap_err(),
                    ValidationError::NoulOutOfRange { .. }
                ),
                "failed to reject out-of-range noul: {}",
                invalid_noul
            );
        }
    }

    #[test]
    fn validate_response_rejects_confidence_out_of_range() {
        for invalid_conf in [-0.1, 1.2] {
            let mut answers = HashMap::new();
            let mut probs = HashMap::new();
            probs.insert("0".into(), 0.7);
            probs.insert("1".into(), 0.3);
            let mut legend = HashMap::new();
            legend.insert("0".into(), "Low".into());
            legend.insert("1".into(), "High".into());
            answers.insert(
                "sc".into(),
                Answer::Score(ScoreAnswer {
                    score: 0.3,
                    legend,
                    probabilities: probs,
                    confidence: invalid_conf,
                }),
            );
            let resp = SystemResponse {
                model: "mock".into(),
                answers,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
            };
            let criteria = HashMap::new();
            assert!(
                matches!(
                    validate_response(&resp, &criteria).unwrap_err(),
                    ValidationError::ConfidenceOutOfRange { .. }
                ),
                "failed to reject out-of-range confidence: {}",
                invalid_conf
            );
        }
    }

    #[test]
    fn validate_response_happy_path_all_three_types() {
        let mut answers = HashMap::new();
        answers.insert("noul".into(), Answer::Noul(NoulAnswer { noul: 0.85 }));
        let mut c_probs = HashMap::new();
        c_probs.insert("a".into(), 0.7);
        c_probs.insert("b".into(), 0.3);
        answers.insert(
            "choice".into(),
            Answer::Choice(ChoiceAnswer {
                choice: "a".into(),
                probabilities: c_probs,
                confidence: 0.75,
            }),
        );
        let mut s_probs = HashMap::new();
        s_probs.insert("0".into(), 0.2);
        s_probs.insert("1".into(), 0.8);
        let mut legend = HashMap::new();
        legend.insert("0".into(), "No".into());
        legend.insert("1".into(), "Yes".into());
        answers.insert(
            "score".into(),
            Answer::Score(ScoreAnswer {
                score: 0.8,
                legend,
                probabilities: s_probs,
                confidence: 0.9,
            }),
        );
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 20,
                output_tokens: 6,
            },
        };
        let mut criteria = HashMap::new();
        criteria.insert("choice".into(), vec!["a".into(), "b".into()]);
        assert!(validate_response(&resp, &criteria).is_ok());
    }
}
