//! Question types — Noul, Choice, Score.
//!
//! Spec: <https://docs.typesafe.ai/api#question-types>
//! > A `Question` is one of three types, set by its `type` field. All three
//! > share `type` and `instructions`; each adds its own `criteria`.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::de::{Error as _, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

/// Tagged union of the three question kinds. The `type` field is the discriminant.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Boolean probability evaluation (yes/no), returning a single probability value.
    Noul(NoulQuestion),
    /// Categorical choice evaluation across discrete options.
    Choice(ChoiceQuestion),
    /// Ordinal rating evaluation rated along an ordered rubric of at least 2 levels.
    Score(ScoreQuestion),
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "lowercase")]
enum QuestionField {
    Type,
    Instructions,
    Criteria,
    #[serde(other)]
    Other,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum QuestionTag {
    Noul,
    Choice,
    Score,
}

// Serde's internally-tagged derive buffers the entire object into `Content`
// before it can pick a variant, allocating twice per question on every
// request parse. The visitor below streams the same wire format directly:
// it reads `type` (emitted first by this engine and the reference SDKs) and
// deserializes the remaining fields straight into the selected variant.
// Fields that legitimately arrive before `type` are buffered as JSON values
// and converted once the tag is known, and every rejected shape (missing or
// unknown tag, missing or duplicate fields, wrong field types) still fails
// exactly like the derived implementation.
impl<'de> Deserialize<'de> for Question {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct QuestionVisitor;

        impl<'de> Visitor<'de> for QuestionVisitor {
            type Value = Question;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a tagged question object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut tag: Option<QuestionTag> = None;
                // `instructions` has the same shape in every variant, so it
                // streams even before the tag is known. `criteria` differs
                // per variant and is buffered as a JSON value if it arrives
                // before the tag.
                let mut instructions: Option<Instructions> = None;
                let mut early_criteria: Option<serde_json::Value> = None;
                let mut noul_criteria: Option<Option<NoulCriteria>> = None;
                let mut choice_criteria: Option<HashMap<String, Option<String>>> = None;
                let mut score_criteria: Option<Vec<String>> = None;

                while let Some(key) = map.next_key::<QuestionField>()? {
                    match key {
                        QuestionField::Type => {
                            if tag.is_some() {
                                return Err(A::Error::duplicate_field("type"));
                            }
                            tag = Some(map.next_value()?);
                        }
                        QuestionField::Instructions => {
                            if instructions.is_some() {
                                return Err(A::Error::duplicate_field("instructions"));
                            }
                            instructions = Some(map.next_value()?);
                        }
                        QuestionField::Criteria => {
                            let already_buffered = early_criteria.is_some()
                                || noul_criteria.is_some()
                                || choice_criteria.is_some()
                                || score_criteria.is_some();
                            if already_buffered {
                                return Err(A::Error::duplicate_field("criteria"));
                            }
                            match tag {
                                None => early_criteria = Some(map.next_value()?),
                                Some(QuestionTag::Noul) => {
                                    noul_criteria = Some(map.next_value::<Option<NoulCriteria>>()?)
                                }
                                Some(QuestionTag::Choice) => {
                                    choice_criteria = Some(map.next_value()?)
                                }
                                Some(QuestionTag::Score) => {
                                    score_criteria = Some(map.next_value()?)
                                }
                            }
                        }
                        QuestionField::Other => {
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                }

                let tag = tag.ok_or_else(|| A::Error::missing_field("type"))?;
                let instructions =
                    instructions.ok_or_else(|| A::Error::missing_field("instructions"))?;
                match tag {
                    QuestionTag::Noul => Ok(Question::Noul(NoulQuestion {
                        instructions,
                        criteria: match noul_criteria {
                            Some(criteria) => criteria,
                            // `null` criteria deserializes to `None`, so the
                            // buffered value converts through the Option.
                            None => convert_criteria::<Option<NoulCriteria>, _>(early_criteria)?
                                .flatten(),
                        },
                    })),
                    QuestionTag::Choice => Ok(Question::Choice(ChoiceQuestion {
                        instructions,
                        criteria: match choice_criteria {
                            Some(criteria) => criteria,
                            None => convert_criteria(early_criteria)?
                                .ok_or_else(|| A::Error::missing_field("criteria"))?,
                        },
                    })),
                    QuestionTag::Score => Ok(Question::Score(ScoreQuestion {
                        instructions,
                        criteria: match score_criteria {
                            Some(criteria) => criteria,
                            None => convert_criteria(early_criteria)?
                                .ok_or_else(|| A::Error::missing_field("criteria"))?,
                        },
                    })),
                }
            }
        }

        deserializer.deserialize_map(QuestionVisitor)
    }
}

/// Convert a `criteria` value that arrived before the `type` tag into the
/// selected variant's criteria type.
fn convert_criteria<T, E>(early: Option<serde_json::Value>) -> Result<Option<T>, E>
where
    T: serde::de::DeserializeOwned,
    E: serde::de::Error,
{
    early
        .map(serde_json::from_value)
        .transpose()
        .map_err(E::custom)
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
