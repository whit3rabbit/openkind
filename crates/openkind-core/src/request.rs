//! Request body. Spec: <https://docs.typesafe.ai/api#request-body>

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::question::Question;
use crate::state::State;

/// Hash builder for wire maps keyed by caller-chosen ids.
///
/// These maps are rebuilt on every request/response deserialization, where
/// short string keys make SipHash a measurable share of parse time. foldhash
/// keeps the same `HashMap` semantics (and randomized-per-instance keys)
/// with a much cheaper hash. Wire bytes are unaffected: JSON object order
/// is not part of the contract.
pub type WireHashState = foldhash::fast::RandomState;

/// Wire API version. Bumped on any breaking schema change.
pub const API_VERSION: &str = "jev-compatible-0.1";

/// Evaluation request payload representing a Jev-compatible System 1 judgment query.
///
/// See <https://docs.typesafe.ai/api#request-body> for the canonical wire specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SystemRequest {
    /// Required. The content to evaluate.
    pub state: State,

    /// Required. `"jev-latest"` or a registered model alias.
    /// We keep this required even though Phase 0 doesn't dispatch — the
    /// engine layer in Phase 2 will use it to pick a backend.
    pub model: String,

    /// Required. Map of user-chosen id → typed question. Keys are NOT sent
    /// to the model and are NOT used in inference (per spec).
    #[serde(deserialize_with = "deserialize_questions")]
    pub questions: HashMap<String, Question, WireHashState>,
}

/// Deserialize the question map pre-sized for typical batches.
///
/// JSON maps cannot advertise their length, so serde's generic
/// deserialization starts at capacity zero and rehashes through the small
/// growth ladder on every request. Pre-allocating for a typical batch, upgrading in one step
/// once a batch proves large
/// removes those intermediate reallocations; oversized batches simply grow
/// as before, and the accepted wire format is unchanged.
fn deserialize_questions<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, Question, WireHashState>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct QuestionsVisitor;

    impl<'de> serde::de::Visitor<'de> for QuestionsVisitor {
        type Value = HashMap<String, Question, WireHashState>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a map of questions")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut questions = HashMap::with_capacity_and_hasher(8, Default::default());
            while let Some(id) = map.next_key::<String>()? {
                questions.insert(id, map.next_value()?);
                if questions.len() == 12 {
                    // Larger batches jump straight to the big table instead
                    // of climbing the small growth ladder.
                    questions.reserve(20);
                }
            }
            Ok(questions)
        }
    }

    deserializer.deserialize_map(QuestionsVisitor)
}
