//! Question types — Noul, Choice, Score.
//!
//! Spec: <https://docs.typesafe.ai/api#question-types>
//! > A `Question` is one of three types, set by its `type` field. All three
//! > share `type` and `instructions`; each adds its own `criteria`.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Tagged union of the three question kinds. The `type` field is the discriminant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Boolean probability evaluation (yes/no), returning a single probability value.
    Noul(NoulQuestion),
    /// Categorical choice evaluation across discrete options.
    Choice(ChoiceQuestion),
    /// Ordinal rating evaluation rated along an ordered rubric of at least 2 levels.
    Score(ScoreQuestion),
}

/// `instructions` is `string | object | array` per the spec.
/// We keep it as a generic JSON value so the model can be prompted with the
/// exact shape the caller chose.
pub type Instructions = serde_json::Value;

// ---------- Noul ----------

/// Yes/no question. Returns the probability the answer is yes.
///
/// Spec: <https://docs.typesafe.ai/api#noul>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NoulQuestion {
    /// Instructions describing what judgment is requested.
    /// Can be a string, object, or array.
    pub instructions: Instructions,
    /// Optional descriptions of what a yes and a no mean.
    /// Reserved keys: `true`, `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

/// Reserved `true`/`false` keys. `r#true` / `r#false` because those are
/// keywords in Rust (and in the JSON spec).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NoulCriteria {
    /// Meaning of a true/yes decision.
    #[serde(rename = "true")]
    pub r#true: String,
    /// Meaning of a false/no decision.
    #[serde(rename = "false")]
    pub r#false: String,
}

// ---------- Choice ----------

/// Pick one option from a set. Spec: <https://docs.typesafe.ai/api#choice>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChoiceQuestion {
    /// Instructions describing what categorical decision is requested.
    pub instructions: Instructions,
    /// Map of option key → rubric description. **Values may be `null`**
    /// ("use null when an option needs no extra detail").
    pub criteria: HashMap<String, Option<String>>,
}

// ---------- Score ----------

/// Rate along an ordered rubric. Spec: <https://docs.typesafe.ai/api#score>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScoreQuestion {
    /// Instructions describing what ordinal score is requested.
    pub instructions: Instructions,
    /// Ordered array of level descriptions. Must contain at least 2.
    pub criteria: Vec<String>,
}
