//! Validation error types and result definitions for the Jev protocol specification.

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
    #[error("question `{0}`: `instructions` must be a non-empty string, object, or array")]
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

    /// Emitted when a score answer's legend disagrees with its probability keys or the requested rubric.
    #[error("score answer `{0}`: legend must match probabilities keys and requested rubric")]
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

    /// Emitted when a score disagrees with its probability-weighted rubric levels.
    #[error("score answer `{id}`: expected probability-weighted score {expected} (got {value})")]
    ScoreExpectationMismatch {
        /// Question identifier.
        id: String,
        /// Probability-weighted mean of the returned level indices.
        expected: f64,
        /// Inconsistent returned score.
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

    /// Emitted when an answer uses a different primitive than its question.
    #[error("question `{id}`: expected {expected} answer, got {actual}")]
    AnswerTypeMismatch {
        /// Question identifier.
        id: String,
        /// Requested primitive.
        expected: &'static str,
        /// Returned primitive.
        actual: &'static str,
    },
}

/// Specialized Result alias for operations returning a [`ValidationError`].
pub type ValidationResult<T> = Result<T, ValidationError>;
