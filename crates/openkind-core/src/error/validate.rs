//! Pure validation functions for requests and responses under the Jev protocol specification.

use std::collections::{HashMap, HashSet};

use crate::answer::Answer;
use crate::error::types::{ValidationError, ValidationResult};
use crate::question::Question;
use crate::request::{SystemRequest, WireHashState};
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
#[inline]
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

#[inline]
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

#[inline]
fn instructions_missing(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::String(s) => s.is_empty(),
        serde_json::Value::Object(m) => m.is_empty(),
        serde_json::Value::Array(a) => a.is_empty(),
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
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
    criteria: &HashMap<String, Vec<String>>,
) -> ValidationResult<()> {
    if !criteria.is_empty() {
        for id in criteria.keys() {
            if !resp.answers.contains_key(id) {
                return Err(ValidationError::MissingAnswer(id.clone()));
            }
        }
        if resp.answers.len() != criteria.len() {
            for id in resp.answers.keys() {
                if !criteria.contains_key(id) {
                    return Err(ValidationError::UnexpectedAnswer(id.clone()));
                }
            }
        }
    }
    validate_response_inner(resp, |id| criteria.get(id).map(Vec::as_slice))
}

fn validate_response_inner<'a>(
    resp: &SystemResponse,
    keys_for: impl Fn(&str) -> Option<&'a [String]>,
) -> ValidationResult<()> {
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
                let expected_keys: HashSet<&str> = keys_for(id)
                    .map(|v| v.iter().map(String::as_str).collect())
                    .unwrap_or_default();
                if !expected_keys.is_empty() && !expected_keys.contains(c.choice.as_str()) {
                    return Err(ValidationError::ChoiceNotInCriteria(id.clone()));
                }
                let prob_keys: HashSet<&str> = c.probabilities.keys().map(String::as_str).collect();
                if !expected_keys.is_empty() && prob_keys != expected_keys {
                    return Err(ValidationError::ProbabilityKeysMismatch(id.clone()));
                }
                check_confidence(id, c.confidence)?;
                check_probabilities(id, &c.probabilities)?;
            }
            Answer::Score(s) => {
                let prob_keys: HashSet<&str> = s.probabilities.keys().map(String::as_str).collect();
                let legend_keys: HashSet<&str> = s.legend.keys().map(String::as_str).collect();
                if prob_keys != legend_keys {
                    return Err(ValidationError::ScoreLegendMismatch(id.clone()));
                }
                if let Some(levels) = keys_for(id).filter(|levels| !levels.is_empty()) {
                    // A self-consistent returned legend can still invent or
                    // omit levels, or change the meaning of the requested rubric.
                    if s.legend.len() != levels.len()
                        || levels.iter().enumerate().any(|(index, description)| {
                            s.legend.get(&index.to_string()) != Some(description)
                        })
                    {
                        return Err(ValidationError::ScoreLegendMismatch(id.clone()));
                    }
                }
                let mut max_idx: u32 = 0;
                for k in &prob_keys {
                    let idx =
                        k.parse::<u32>()
                            .map_err(|_| ValidationError::ScoreIndexNotNumeric {
                                id: id.clone(),
                                key: k.to_string(),
                            })?;
                    if idx.to_string() != *k {
                        return Err(ValidationError::ScoreIndexNotNumeric {
                            id: id.clone(),
                            key: k.to_string(),
                        });
                    }
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
                let expected: f64 = s
                    .probabilities
                    .iter()
                    .map(|(key, probability)| {
                        key.parse::<u32>().expect("validated score index") as f64 * probability
                    })
                    .sum();
                // Scale the existing probability tolerance to the rubric's numeric range.
                if (s.score - expected).abs() > 1e-3 * max_score.max(1.0) {
                    return Err(ValidationError::ScoreExpectationMismatch {
                        id: id.clone(),
                        expected,
                        value: s.score,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Request-bound answer expectations retained while an engine consumes the request.
pub struct ResponseContract {
    questions: HashMap<String, (AnswerKind, Vec<String>), WireHashState>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AnswerKind {
    Noul,
    Choice,
    Score,
}

impl AnswerKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }
}

impl ResponseContract {
    /// Capture question IDs, types, Choice keys, and Score levels without copying state or instructions.
    pub fn from_request(req: &SystemRequest) -> ValidationResult<Self> {
        if req.questions.is_empty() {
            return Err(ValidationError::NoQuestions);
        }
        if req.questions.len() > MAX_QUESTIONS_PER_REQUEST {
            return Err(ValidationError::TooManyQuestions {
                count: req.questions.len(),
                max: MAX_QUESTIONS_PER_REQUEST,
            });
        }
        let mut questions =
            HashMap::with_capacity_and_hasher(req.questions.len(), WireHashState::default());
        for (id, question) in &req.questions {
            validate_question(id, question)?;
            let (kind, keys) = contract_entry(question);
            questions.insert(id.clone(), (kind, keys));
        }
        Ok(Self { questions })
    }

    /// Check that every answer matches the originating question and wire rules.
    pub fn validate(&self, resp: &SystemResponse) -> ValidationResult<()> {
        for (id, (expected, _)) in &self.questions {
            let answer = resp
                .answers
                .get(id)
                .ok_or_else(|| ValidationError::MissingAnswer(id.clone()))?;
            let actual = answer_kind(answer);
            if actual != *expected {
                return Err(ValidationError::AnswerTypeMismatch {
                    id: id.clone(),
                    expected: expected.as_str(),
                    actual: actual.as_str(),
                });
            }
        }
        if resp.answers.len() != self.questions.len() {
            for id in resp.answers.keys() {
                if !self.questions.contains_key(id) {
                    return Err(ValidationError::UnexpectedAnswer(id.clone()));
                }
            }
        }
        // Reuse the retained contract directly. Rebuilding a second map here
        // copied every question ID and Choice key on each response.
        validate_response_inner(resp, |id| {
            self.questions.get(id).map(|(_, keys)| keys.as_slice())
        })
    }
}

fn contract_entry(question: &Question) -> (AnswerKind, Vec<String>) {
    match question {
        Question::Noul(_) => (AnswerKind::Noul, Vec::new()),
        Question::Choice(choice) => (
            AnswerKind::Choice,
            choice.criteria.keys().cloned().collect(),
        ),
        Question::Score(score) => (AnswerKind::Score, score.criteria.clone()),
    }
}

fn answer_kind(answer: &Answer) -> AnswerKind {
    match answer {
        Answer::Noul(_) => AnswerKind::Noul,
        Answer::Choice(_) => AnswerKind::Choice,
        Answer::Score(_) => AnswerKind::Score,
    }
}

/// Validate a response against its originating request.
///
/// The request supplies the question types, Choice option set, and Score rubric
/// that cannot be established from a response alone. The older [`validate_response`]
/// remains available for callers that only have a criteria map.
pub fn validate_response_for_request(
    resp: &SystemResponse,
    req: &SystemRequest,
) -> ValidationResult<()> {
    ResponseContract::from_request(req)?.validate(resp)
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

fn check_probabilities(id: &str, probabilities: &HashMap<String, f64>) -> ValidationResult<()> {
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
