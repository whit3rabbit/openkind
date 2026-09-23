//! Ergonomic constructors for [`Question`] and [`State`] wire types.
//!
//! The [`crate`] re-exports the `openkind-core` structs directly; these
//! helpers just remove the enum-wrapping noise for the common
//! string-instructions cases, mirroring the Python SDK's `Noul(...)`,
//! `Choice(...)`, `Score(...)` sugar:
//!
//! ```
//! use openkind_client::{question, Client};
//!
//! # async fn demo(client: &Client) -> Result<(), openkind_client::Error> {
//! let answer = client
//!     .system_one(
//!         "I was charged twice. Please help.",
//!         [
//!             ("billing", question::noul("Is this about billing?")),
//!             ("tone", question::choice("What is the tone?", [("calm", None), ("angry", None)])),
//!             ("severity", question::score("How severe?", ["low", "medium", "high"])),
//!         ],
//!     )
//!     .await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

use openkind_core::{ChoiceQuestion, NoulCriteria, NoulQuestion, Question, ScoreQuestion, State};

/// The wire type for question instructions: a JSON string, object, or array.
pub type Instructions = serde_json::Value;

/// A boolean-probability (`noul`) question with plain-text instructions.
pub fn noul(instructions: impl Into<Instructions>) -> Question {
    Question::Noul(NoulQuestion {
        instructions: instructions.into(),
        criteria: None,
    })
}

/// A `noul` question with explicit meanings for the true and false outcomes.
pub fn noul_with(
    instructions: impl Into<Instructions>,
    is_true: impl Into<String>,
    is_false: impl Into<String>,
) -> Question {
    Question::Noul(NoulQuestion {
        instructions: instructions.into(),
        criteria: Some(NoulCriteria {
            r#true: is_true.into(),
            r#false: is_false.into(),
        }),
    })
}

/// A categorical (`choice`) question. `criteria` maps each option key to an
/// optional description (`None` when the key needs no detail).
pub fn choice<K: Into<String>>(
    instructions: impl Into<Instructions>,
    criteria: impl IntoIterator<Item = (K, Option<String>)>,
) -> Question {
    Question::Choice(ChoiceQuestion {
        instructions: instructions.into(),
        criteria: criteria
            .into_iter()
            .map(|(k, v)| (k.into(), v))
            .collect::<HashMap<_, _, _>>(),
    })
}

/// An ordinal (`score`) question over an ordered rubric of at least two levels.
pub fn score(
    instructions: impl Into<Instructions>,
    levels: impl IntoIterator<Item: Into<String>>,
) -> Question {
    Question::Score(ScoreQuestion {
        instructions: instructions.into(),
        criteria: levels.into_iter().map(Into::into).collect(),
    })
}

/// Plain-text [`State`] (the most common case).
pub fn text(s: impl Into<String>) -> State {
    State::Text(s.into())
}

/// Structured JSON-object [`State`] (chat logs, app records, diffs, ...).
pub fn structured_object(value: serde_json::Map<String, serde_json::Value>) -> State {
    State::Object(value)
}

/// JSON-array [`State`] (e.g. an ordered list of chat entries).
pub fn structured_array(values: Vec<serde_json::Value>) -> State {
    State::Array(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openkind_core::{validate_request, SystemRequest};

    #[test]
    fn helpers_produce_valid_wire_requests() {
        let request = SystemRequest {
            state: text("I was charged twice. Please help."),
            model: "mock".into(),
            questions: HashMap::from_iter([
                ("billing".to_string(), noul("Is this about billing?")),
                (
                    "tone".to_string(),
                    choice("What is the tone?", [("calm", None), ("angry", None)]),
                ),
                (
                    "severity".to_string(),
                    score("How severe?", ["low", "medium", "high"]),
                ),
                (
                    "truthy".to_string(),
                    noul_with("Did the user dispute?", "user disputes", "user accepts"),
                ),
            ]),
        };
        assert!(validate_request(&request).is_ok(), "must pass validation");
    }

    #[test]
    fn noul_serializes_without_confidence() {
        let q = noul("yes or no?");
        let wire = serde_json::to_value(q).unwrap();
        assert_eq!(wire["type"], "noul");
        assert_eq!(wire["instructions"], "yes or no?");
    }

    #[test]
    fn noul_with_uses_reserved_true_false_keys() {
        let q = noul_with("disputed?", "disputed", "accepted");
        let wire = serde_json::to_value(q).unwrap();
        assert_eq!(wire["criteria"]["true"], "disputed");
        assert_eq!(wire["criteria"]["false"], "accepted");
    }

    #[test]
    fn structured_states_serialize() {
        let mut map = serde_json::Map::new();
        map.insert("user".into(), serde_json::json!({"id": 7}));
        assert!(matches!(structured_object(map), State::Object(_)));
        assert!(matches!(
            structured_array(vec![serde_json::json!("hello")]),
            State::Array(_)
        ));
    }
}
