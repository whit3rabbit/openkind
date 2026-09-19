//! Answer types — one per Question type.
//!
//! Spec: <https://docs.typesafe.ai/api#answer-types>
//! > Every answer carries a `type` matching its question. Choice and Score
//! > answers also carry a `confidence` between 0 to 1, derived from the
//! > answer's probability distribution.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Tagged union of answer models matching the evaluated question types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// Boolean probability answer containing a single probability value.
    Noul(NoulAnswer),
    /// Categorical choice answer containing selected label, probability distribution, and confidence.
    Choice(ChoiceAnswer),
    /// Ordinal rating answer containing the evaluated score, rubric legend, probabilities, and confidence.
    Score(ScoreAnswer),
}

/// Spec: `noul` is a number 0..1. **No `confidence` field** — Noul answers
/// don't get one per the spec.
///
/// `f64` so wire-format precision matches the API spec exactly (a 0.92
/// going through f32 round-trips to 0.9200000166893005 — wrong).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NoulAnswer {
    /// Probability value in `[0.0, 1.0]` representing the likelihood that the answer is true/yes.
    pub noul: f64,
}

/// `probabilities` is a full distribution (sums to 1) over the criteria keys.
/// `confidence` is required (per spec). `f64` for wire-format precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChoiceAnswer {
    /// Selected option identifier matching one of the keys in question criteria.
    pub choice: String,
    /// Normalized probability distribution over all criteria options summing to 1.0.
    pub probabilities: HashMap<String, f64>,
    /// Model confidence score in `[0.0, 1.0]` derived from the probability distribution.
    pub confidence: f64,
}

/// `legend` is level-index (string) → level description.
/// `probabilities` is keyed by the same index strings. `confidence` required.
/// `f64` for wire-format precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScoreAnswer {
    /// Inferred expected score value along the ordinal rubric.
    pub score: f64,
    /// Mapping of numeric level index strings (`"0"`, `"1"`, ...) to rubric descriptions.
    pub legend: HashMap<String, String>,
    /// Normalized probability distribution over the level indices summing to 1.0.
    pub probabilities: HashMap<String, f64>,
    /// Model confidence score in `[0.0, 1.0]` derived from the probability distribution.
    pub confidence: f64,
}
