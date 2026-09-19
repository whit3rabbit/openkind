//! Validation errors. Spec lists four error statuses (401/422/429/529);
//! `422 Unprocessable Entity` is the one this module emits — invalid bodies.

use thiserror::Error;

/// Validation errors representing violations of the Jev protocol specification.
///
/// When encountered in the API layer, these errors are translated directly into
/// HTTP 422 Unprocessable Entity responses (`invalid_body` code) with detailed diagnostic messages.
#[derive(Debug, Error)]
pub enum ValidationError {
    /// Emitted when a request contains an empty `questions` map.
    /// The Jev specification requires at least one question per evaluation request.
    #[error("`questions` must contain at least one entry")]
    NoQuestions,

    /// Emitted when a question does not provide non-empty `instructions`.
    /// Instructions must be a non-empty string, object, or array.
    #[error("question `{0}`: `instructions` is required")]
    MissingInstructions(String),

    /// Emitted when a `choice` question provides an empty `criteria` map.
    /// A choice question must have at least one selectable option.
    #[error("question `{0}` (choice): `criteria` must contain at least one option")]
    ChoiceCriteriaEmpty(String),

    /// Emitted when a `score` question provides fewer than 2 rubric levels in `criteria`.
    /// An ordinal rating rubric requires at least 2 distinct levels.
    #[error("question `{0}` (score): `criteria` must contain at least 2 levels")]
    ScoreCriteriaTooFew(String),

    /// Emitted when any rubric level description in a `score` question is empty.
    #[error("question `{id}` (score): level descriptions must be non-empty")]
    ScoreLevelEmpty {
        /// Identifier of the question with the empty level description.
        id: String,
        /// Zero-based index of the invalid empty level.
        index: usize,
    },

    /// Emitted when a `noul` question's criteria specifies an empty `true` description.
    #[error("question `{0}` (noul): `criteria.true` is empty")]
    NoulTrueEmpty(String),

    /// Emitted when a `noul` question's criteria specifies an empty `false` description.
    #[error("question `{0}` (noul): `criteria.false` is empty")]
    NoulFalseEmpty(String),

    /// Emitted when a choice answer's selected `choice` string does not match any key in the question's criteria.
    #[error("question `{0}`: choice answer `choice` not in criteria keys")]
    ChoiceNotInCriteria(String),

    /// Emitted when the keys in a choice answer's `probabilities` map do not exactly match the question's criteria keys.
    #[error("question `{0}`: choice answer `probabilities` keys must match criteria")]
    ProbabilityKeysMismatch(String),

    /// Emitted when probabilities in an answer distribution do not sum to 1.0 (within epsilon tolerance).
    #[error("question `{id}`: probabilities do not sum to 1 (got {sum})")]
    ProbabilitiesDontSum {
        /// Question identifier.
        id: String,
        /// Calculated sum of the probability distribution.
        sum: f64,
    },

    /// Emitted when any probability value in an answer distribution is outside the valid range [0.0, 1.0] or NaN.
    #[error("answer `{id}`: probability must be in 0..=1 (got {value})")]
    ProbabilityOutOfRange {
        /// Question identifier.
        id: String,
        /// Invalid probability value.
        value: f64,
    },

    /// Emitted when an answer's `confidence` score is outside [0.0, 1.0] or NaN.
    #[error("answer `{id}`: confidence must be in 0..=1 (got {value})")]
    ConfidenceOutOfRange {
        /// Question identifier.
        id: String,
        /// Invalid confidence value.
        value: f64,
    },

    /// Emitted when a `noul` answer value is outside [0.0, 1.0] or NaN.
    #[error("answer `{id}`: noul must be in 0..=1 (got {value})")]
    NoulOutOfRange {
        /// Question identifier.
        id: String,
        /// Invalid noul probability value.
        value: f64,
    },

    /// Emitted when the keys in a score answer's `legend` do not match the keys in its `probabilities` map.
    #[error("score answer `{0}`: legend keys must match probabilities keys")]
    ScoreLegendMismatch(String),

    /// Emitted when a score answer's legend or probability key cannot be parsed as a numeric index string (e.g. "0", "1").
    #[error("score answer `{id}`: legend/probability indices must be numeric, got `{key}`")]
    ScoreIndexNotNumeric {
        /// Question identifier.
        id: String,
        /// Non-numeric key string encountered.
        key: String,
    },

    /// Emitted when a score answer's `score` is outside [0.0, max_level], NaN, or infinite.
    #[error("score answer `{id}`: score must be finite in 0..={max} (got {value})")]
    ScoreOutOfRange {
        /// Question identifier.
        id: String,
        /// Maximum allowed score index.
        max: f64,
        /// Invalid score value.
        value: f64,
    },

    /// Emitted when a request contains more questions than the maximum allowed limit.
    #[error("request exceeds maximum question count limit (got {count}, max {max})")]
    TooManyQuestions {
        /// Number of questions in the request.
        count: usize,
        /// Maximum allowed questions.
        max: usize,
    },

    /// Emitted when a question's criteria options exceed the maximum allowed limit.
    #[error("question `{id}` exceeds maximum criteria options limit (got {count}, max {max})")]
    TooManyCriteriaOptions {
        /// Question identifier.
        id: String,
        /// Number of criteria options.
        count: usize,
        /// Maximum allowed criteria options.
        max: usize,
    },

    /// Emitted when a response lacks an answer for a question id the caller supplied.
    #[error("response is missing an answer for question `{0}`")]
    MissingAnswer(String),

    /// Emitted when a response contains an answer for a question id that was not requested.
    #[error("response contains an answer for unknown question `{0}`")]
    UnexpectedAnswer(String),
}

/// Specialized Result alias for operations returning a [`ValidationError`].
pub type ValidationResult<T> = Result<T, ValidationError>;

use crate::answer::Answer;
use crate::question::Question;
use crate::request::SystemRequest;
use crate::response::SystemResponse;

/// Maximum number of questions allowed in a single evaluation request to prevent DoS.
pub const MAX_QUESTIONS_PER_REQUEST: usize = 10_000;

/// Maximum number of criteria options allowed per question to prevent DoS.
pub const MAX_CRITERIA_OPTIONS: usize = 10_000;

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
    if req.questions.len() > MAX_QUESTIONS_PER_REQUEST {
        return Err(ValidationError::TooManyQuestions {
            count: req.questions.len(),
            max: MAX_QUESTIONS_PER_REQUEST,
        });
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
            if c.criteria.len() > MAX_CRITERIA_OPTIONS {
                return Err(ValidationError::TooManyCriteriaOptions {
                    id: id.to_string(),
                    count: c.criteria.len(),
                    max: MAX_CRITERIA_OPTIONS,
                });
            }
        }
        Question::Score(s) => {
            if instructions_missing(&s.instructions) {
                return Err(ValidationError::MissingInstructions(id.to_string()));
            }
            if s.criteria.len() < 2 {
                return Err(ValidationError::ScoreCriteriaTooFew(id.to_string()));
            }
            if s.criteria.len() > MAX_CRITERIA_OPTIONS {
                return Err(ValidationError::TooManyCriteriaOptions {
                    id: id.to_string(),
                    count: s.criteria.len(),
                    max: MAX_CRITERIA_OPTIONS,
                });
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
/// - missing or unexpected answers (when `criteria` covers the request's question ids)
/// - probabilities don't sum to 1
/// - confidence out of range
/// - choice answer's keys don't match criteria
/// - score answer legend doesn't match probabilities
///
/// `criteria` maps question id → the question's choice-criteria keys (empty for
/// noul/score questions). When the map is non-empty it is treated as the full
/// question-id set of the originating request, and answer coverage is enforced
/// both ways: every requested id must be answered, and no extra answers may
/// appear. An empty map skips coverage checks (per-answer checks still run).
pub fn validate_response(
    resp: &SystemResponse,
    criteria: &std::collections::HashMap<String, Vec<String>>,
) -> ValidationResult<()> {
    if !criteria.is_empty() {
        for id in criteria.keys() {
            if !resp.answers.contains_key(id) {
                return Err(ValidationError::MissingAnswer(id.clone()));
            }
        }
        for id in resp.answers.keys() {
            if !criteria.contains_key(id) {
                return Err(ValidationError::UnexpectedAnswer(id.clone()));
            }
        }
    }
    // Every question id in the request must have a matching answer.
    // (Caller passes criteria so we can validate cross-references.)
    for (id, ans) in &resp.answers {
        match ans {
            Answer::Noul(n) => {
                if n.noul.is_nan() || !n.noul.is_finite() || !(0.0..=1.0).contains(&n.noul) {
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
                check_probabilities(id, &c.probabilities)?;
            }
            Answer::Score(s) => {
                let prob_keys: std::collections::HashSet<&str> =
                    s.probabilities.keys().map(String::as_str).collect();
                let legend_keys: std::collections::HashSet<&str> =
                    s.legend.keys().map(String::as_str).collect();
                if prob_keys != legend_keys {
                    return Err(ValidationError::ScoreLegendMismatch(id.clone()));
                }
                let mut max_idx: u32 = 0;
                for k in &prob_keys {
                    let idx =
                        k.parse::<u32>()
                            .map_err(|_| ValidationError::ScoreIndexNotNumeric {
                                id: id.clone(),
                                key: k.to_string(),
                            })?;
                    max_idx = max_idx.max(idx);
                }
                let max_score = max_idx as f64;
                if s.score.is_nan() || !s.score.is_finite() || s.score < 0.0 || s.score > max_score
                {
                    return Err(ValidationError::ScoreOutOfRange {
                        id: id.clone(),
                        max: max_score,
                        value: s.score,
                    });
                }
                check_confidence(id, s.confidence)?;
                check_probabilities(id, &s.probabilities)?;
            }
        }
    }
    Ok(())
}

fn check_confidence(id: &str, value: f64) -> ValidationResult<()> {
    if value.is_nan() || !(0.0..=1.0).contains(&value) {
        return Err(ValidationError::ConfidenceOutOfRange {
            id: id.to_string(),
            value,
        });
    }
    Ok(())
}

fn check_probabilities(
    id: &str,
    probabilities: &std::collections::HashMap<String, f64>,
) -> ValidationResult<()> {
    for &p in probabilities.values() {
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return Err(ValidationError::ProbabilityOutOfRange {
                id: id.to_string(),
                value: p,
            });
        }
    }
    let sum: f64 = probabilities.values().copied().sum();
    check_sums_to_one(id, sum)
}

fn check_sums_to_one(id: &str, sum: f64) -> ValidationResult<()> {
    if sum.is_nan() || (sum - 1.0).abs() > 1e-3 {
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
        criteria.insert("noul".into(), Vec::new());
        criteria.insert("score".into(), Vec::new());
        assert!(validate_response(&resp, &criteria).is_ok());
    }

    #[test]
    fn validate_response_rejects_missing_answer() {
        let mut answers = HashMap::new();
        answers.insert("noul".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
            },
        };
        let mut criteria = HashMap::new();
        criteria.insert("noul".into(), Vec::new());
        criteria.insert("dropped".into(), Vec::new());
        assert!(matches!(
            validate_response(&resp, &criteria).unwrap_err(),
            ValidationError::MissingAnswer(ref id) if id == "dropped"
        ));
    }

    #[test]
    fn validate_response_rejects_unexpected_answer() {
        let mut answers = HashMap::new();
        answers.insert("noul".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
        answers.insert("ghost".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
            },
        };
        let mut criteria = HashMap::new();
        criteria.insert("noul".into(), Vec::new());
        assert!(matches!(
            validate_response(&resp, &criteria).unwrap_err(),
            ValidationError::UnexpectedAnswer(ref id) if id == "ghost"
        ));
    }

    #[test]
    fn validate_response_empty_criteria_skips_coverage_checks() {
        let mut answers = HashMap::new();
        answers.insert("anything".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
            },
        };
        assert!(validate_response(&resp, &HashMap::new()).is_ok());
    }

    #[test]
    fn validate_response_rejects_nan_probabilities() {
        let mut answers = HashMap::new();
        let mut c_probs = HashMap::new();
        c_probs.insert("a".into(), f64::NAN);
        c_probs.insert("b".into(), 0.5);
        answers.insert(
            "choice".into(),
            Answer::Choice(ChoiceAnswer {
                choice: "a".into(),
                probabilities: c_probs,
                confidence: 0.5,
            }),
        );
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 2,
            },
        };
        let mut criteria = HashMap::new();
        criteria.insert("choice".into(), vec!["a".into(), "b".into()]);
        assert!(matches!(
            validate_response(&resp, &criteria).unwrap_err(),
            ValidationError::ProbabilityOutOfRange { .. }
        ));
    }

    #[test]
    fn validate_response_rejects_negative_probabilities() {
        let mut answers = HashMap::new();
        let mut c_probs = HashMap::new();
        c_probs.insert("a".into(), -0.2);
        c_probs.insert("b".into(), 1.2);
        answers.insert(
            "choice".into(),
            Answer::Choice(ChoiceAnswer {
                choice: "b".into(),
                probabilities: c_probs,
                confidence: 0.5,
            }),
        );
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 2,
            },
        };
        let mut criteria = HashMap::new();
        criteria.insert("choice".into(), vec!["a".into(), "b".into()]);
        assert!(matches!(
            validate_response(&resp, &criteria).unwrap_err(),
            ValidationError::ProbabilityOutOfRange { .. }
        ));
    }

    #[test]
    fn validate_response_rejects_nan_and_out_of_range_score() {
        let mut legend = HashMap::new();
        legend.insert("0".into(), "Low".into());
        legend.insert("1".into(), "High".into());

        let mut score_probs = HashMap::new();
        score_probs.insert("0".into(), 0.5);
        score_probs.insert("1".into(), 0.5);

        // Test NaN score
        let mut answers_nan = HashMap::new();
        answers_nan.insert(
            "score_q".into(),
            Answer::Score(ScoreAnswer {
                score: f64::NAN,
                legend: legend.clone(),
                probabilities: score_probs.clone(),
                confidence: 0.8,
            }),
        );
        let resp_nan = SystemResponse {
            model: "mock".into(),
            answers: answers_nan,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 4,
            },
        };
        assert!(matches!(
            validate_response(&resp_nan, &HashMap::new()).unwrap_err(),
            ValidationError::ScoreOutOfRange { .. }
        ));

        // Test out of range score (score 2.5 when max index is 1)
        let mut answers_oor = HashMap::new();
        answers_oor.insert(
            "score_q".into(),
            Answer::Score(ScoreAnswer {
                score: 2.5,
                legend,
                probabilities: score_probs,
                confidence: 0.8,
            }),
        );
        let resp_oor = SystemResponse {
            model: "mock".into(),
            answers: answers_oor,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 4,
            },
        };
        assert!(matches!(
            validate_response(&resp_oor, &HashMap::new()).unwrap_err(),
            ValidationError::ScoreOutOfRange { .. }
        ));
    }

    #[test]
    fn validate_response_rejects_nan_noul() {
        let mut answers = HashMap::new();
        answers.insert("noul_q".into(), Answer::Noul(NoulAnswer { noul: f64::NAN }));
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 1,
            },
        };
        assert!(matches!(
            validate_response(&resp, &HashMap::new()).unwrap_err(),
            ValidationError::NoulOutOfRange { .. }
        ));
    }

    #[test]
    fn validate_request_rejects_excessive_questions_and_criteria() {
        let mut questions = HashMap::new();
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
        let mut questions = HashMap::new();
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
        let mut questions = HashMap::new();
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
}
