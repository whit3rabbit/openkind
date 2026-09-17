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
    Noul(NoulQuestion),
    Choice(ChoiceQuestion),
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
    #[serde(rename = "true")]
    pub r#true: String,
    #[serde(rename = "false")]
    pub r#false: String,
}

// ---------- Choice ----------

/// Pick one option from a set. Spec: <https://docs.typesafe.ai/api#choice>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChoiceQuestion {
    pub instructions: Instructions,
    /// Map of option key → rubric description. **Values may be `null`**
    /// ("use null when an option needs no extra detail").
    pub criteria: HashMap<String, Option<String>>,
}

// ---------- Score ----------

/// Rate along an ordered rubric. Spec: <https://docs.typesafe.ai/api#score>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScoreQuestion {
    pub instructions: Instructions,
    /// Ordered array of level descriptions. Must contain at least 2.
    pub criteria: Vec<String>,
}
