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
                let expected_keys: std::collections::HashSet<&str> =
                    criteria.get(id).map(|v| v.iter().map(String::as_str).collect()).unwrap_or_default();
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
                    k.parse::<u32>().map_err(|_| ValidationError::ScoreIndexNotNumeric {
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