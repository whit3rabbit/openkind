//! Answer types — one per Question type.
//!
//! Spec: <https://docs.typesafe.ai/api#answer-types>
//! > Every answer carries a `type` matching its question. Choice and Score
//! > answers also carry a `confidence` between 0 to 1, derived from the
//! > answer's probability distribution.

use std::borrow::Cow;
use std::collections::HashMap;

use schemars::JsonSchema;
use serde::de::{Error as _, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

/// Tagged union of answer models matching the evaluated question types.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// Boolean probability answer containing a single probability value.
    Noul(NoulAnswer),
    /// Categorical choice answer containing selected label, probability distribution, and confidence.
    Choice(ChoiceAnswer),
    /// Ordinal rating answer containing the evaluated score, rubric legend, probabilities, and confidence.
    Score(ScoreAnswer),
}

// Streaming deserialization for the same reason as `Question`: serde's
// internally-tagged derive buffers every answer into `Content` before
// picking a variant, which showed up as a measurable share of client-side
// response decoding. The visitor reads `type` and deserializes the rest of
// the fields straight into the selected variant; fields arriving before the
// tag are buffered as JSON values and converted once the tag is known.
impl<'de> Deserialize<'de> for Answer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct AnswerVisitor;

        impl<'de> Visitor<'de> for AnswerVisitor {
            type Value = Answer;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a tagged answer object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut tag: Option<Cow<'de, str>> = None;
                let mut noul: Option<f64> = None;
                let mut choice: Option<String> = None;
                let mut probabilities: Option<HashMap<String, f64>> = None;
                let mut confidence: Option<f64> = None;
                let mut score: Option<f64> = None;
                let mut legend: Option<HashMap<String, String>> = None;
                // Fields seen before the tag, keyed in arrival order.
                let mut early: Vec<(&'static str, serde_json::Value)> = Vec::new();

                while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                    let field: &'static str = match key.as_ref() {
                        "type" => {
                            if tag.is_some() {
                                return Err(A::Error::duplicate_field("type"));
                            }
                            let value = map.next_value::<Cow<'de, str>>()?;
                            tag = Some(match value.as_ref() {
                                "noul" | "choice" | "score" => value,
                                other => {
                                    return Err(A::Error::unknown_variant(
                                        other,
                                        &["noul", "choice", "score"],
                                    ))
                                }
                            });
                            continue;
                        }
                        "noul" => "noul",
                        "choice" => "choice",
                        "probabilities" => "probabilities",
                        "confidence" => "confidence",
                        "score" => "score",
                        "legend" => "legend",
                        _ => {
                            let _ = map.next_value::<IgnoredAny>()?;
                            continue;
                        }
                    };
                    let Some(answer_tag) = tag.as_deref() else {
                        early.push((field, map.next_value()?));
                        continue;
                    };
                    // Known fields from other variants are still unknown
                    // fields for this variant, so their values are ignored.
                    if !field_belongs_to_answer(answer_tag, field) {
                        let _ = map.next_value::<IgnoredAny>()?;
                        continue;
                    }
                    let already = |early: &[(&'static str, serde_json::Value)]| {
                        early.iter().any(|(name, _)| *name == field)
                    };
                    let duplicate = match field {
                        "noul" => noul.is_some(),
                        "choice" => choice.is_some(),
                        "probabilities" => probabilities.is_some(),
                        "confidence" => confidence.is_some(),
                        "score" => score.is_some(),
                        "legend" => legend.is_some(),
                        _ => false,
                    } || already(&early);
                    if duplicate {
                        return Err(A::Error::duplicate_field(field));
                    }
                    match field {
                        "noul" => noul = Some(map.next_value()?),
                        "choice" => choice = Some(map.next_value()?),
                        "probabilities" => probabilities = Some(map.next_value()?),
                        "confidence" => confidence = Some(map.next_value()?),
                        "score" => score = Some(map.next_value()?),
                        "legend" => legend = Some(map.next_value()?),
                        _ => unreachable!("field matched the list above"),
                    }
                }

                // Replay any fields that arrived before the tag.
                let tag = tag.ok_or_else(|| A::Error::missing_field("type"))?;
                let mut replayed_fields = Vec::new();
                for (field, value) in early {
                    if !field_belongs_to_answer(&tag, field) {
                        continue;
                    }
                    if replayed_fields.contains(&field) {
                        return Err(A::Error::duplicate_field(field));
                    }
                    replayed_fields.push(field);
                    let converted: serde_json::Value = value;
                    match field {
                        "noul" => {
                            noul =
                                Some(serde_json::from_value(converted).map_err(A::Error::custom)?)
                        }
                        "choice" => {
                            choice =
                                Some(serde_json::from_value(converted).map_err(A::Error::custom)?)
                        }
                        "probabilities" => {
                            probabilities =
                                Some(serde_json::from_value(converted).map_err(A::Error::custom)?)
                        }
                        "confidence" => {
                            confidence =
                                Some(serde_json::from_value(converted).map_err(A::Error::custom)?)
                        }
                        "score" => {
                            score =
                                Some(serde_json::from_value(converted).map_err(A::Error::custom)?)
                        }
                        "legend" => {
                            legend =
                                Some(serde_json::from_value(converted).map_err(A::Error::custom)?)
                        }
                        _ => unreachable!("field matched the list above"),
                    }
                }

                match tag.as_ref() {
                    "noul" => Ok(Answer::Noul(NoulAnswer {
                        noul: noul.ok_or_else(|| A::Error::missing_field("noul"))?,
                    })),
                    "choice" => Ok(Answer::Choice(ChoiceAnswer {
                        choice: choice.ok_or_else(|| A::Error::missing_field("choice"))?,
                        probabilities: probabilities
                            .ok_or_else(|| A::Error::missing_field("probabilities"))?,
                        confidence: confidence
                            .ok_or_else(|| A::Error::missing_field("confidence"))?,
                    })),
                    "score" => Ok(Answer::Score(ScoreAnswer {
                        score: score.ok_or_else(|| A::Error::missing_field("score"))?,
                        legend: legend.ok_or_else(|| A::Error::missing_field("legend"))?,
                        probabilities: probabilities
                            .ok_or_else(|| A::Error::missing_field("probabilities"))?,
                        confidence: confidence
                            .ok_or_else(|| A::Error::missing_field("confidence"))?,
                    })),
                    _ => unreachable!("tag validated against the variant list above"),
                }
            }
        }

        deserializer.deserialize_map(AnswerVisitor)
    }
}

fn field_belongs_to_answer(tag: &str, field: &str) -> bool {
    match tag {
        "noul" => field == "noul",
        "choice" => matches!(field, "choice" | "probabilities" | "confidence"),
        "score" => matches!(field, "score" | "legend" | "probabilities" | "confidence"),
        _ => false,
    }
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
