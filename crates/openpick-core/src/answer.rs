//! Answer types — one per Question type.
//!
//! Spec: <https://docs.typesafe.ai/api#answer-types>
//! > Every answer carries a `type` matching its question. Choice and Score
//! > answers also carry a `confidence` between 0 to 1, derived from the
//! > answer's probability distribution.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
}

/// Spec: `noul` is a number 0..1. **No `confidence` field** — Noul answers
/// don't get one per the spec.
///
/// `f64` so wire-format precision matches the API spec exactly (a 0.92
/// going through f32 round-trips to 0.9200000166893005 — wrong).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NoulAnswer {
    pub noul: f64,
}

/// `probabilities` is a full distribution (sums to 1) over the criteria keys.
/// `confidence` is required (per spec). `f64` for wire-format precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub probabilities: HashMap<String, f64>,
    pub confidence: f64,
}

/// `legend` is level-index (string) → level description.
/// `probabilities` is keyed by the same index strings. `confidence` required.
/// `f64` for wire-format precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScoreAnswer {
    pub score: f64,
    pub legend: HashMap<String, String>,
    pub probabilities: HashMap<String, f64>,
    pub confidence: f64,
}