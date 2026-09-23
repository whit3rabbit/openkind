//! `state` field — the content to evaluate.
//!
//! From the spec:
//! > `state`: string | object | array (required)
//! > The content to evaluate. A plain string for text, or structured data
//! > (object/array) for things like chat logs, records, or the current
//! > state of your application.

use schemars::JsonSchema;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

/// State to evaluate. Permissive JSON shape: any string, object, or array.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum State {
    /// Plain text.
    Text(String),
    /// Structured object (e.g. `{ "user": {...}, "logs": [...], "diff": "..." }`).
    Object(serde_json::Map<String, serde_json::Value>),
    /// Ordered list (e.g. chat log entries).
    Array(Vec<serde_json::Value>),
}

// `#[serde(untagged)]` deserialization buffers the whole value into serde's
// internal `Content` type before picking a variant, allocating on every
// request parse. The visitor below streams the same decision directly from
// the payload shape with identical accepted inputs: strings, objects, and
// arrays deserialize to the same variants, and any other shape is rejected.
impl<'de> Deserialize<'de> for State {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StateVisitor;

        impl<'de> Visitor<'de> for StateVisitor {
            type Value = State;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a string, object, or array")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(State::Text(value.to_owned()))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut object = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    object.insert(key, map.next_value()?);
                }
                Ok(State::Object(object))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut array = Vec::new();
                while let Some(item) = seq.next_element::<serde_json::Value>()? {
                    array.push(item);
                }
                Ok(State::Array(array))
            }
        }

        deserializer.deserialize_any(StateVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_deserializes_each_json_shape_to_matching_variant() {
        let text: State = serde_json::from_str("\"hello\"").unwrap();
        assert_eq!(text, State::Text("hello".into()));

        let object: State = serde_json::from_str(r#"{"user": {"id": 7}, "logs": []}"#).unwrap();
        match object {
            State::Object(map) => {
                assert_eq!(map.len(), 2);
                assert_eq!(map["user"]["id"], serde_json::json!(7));
                assert_eq!(map["logs"], serde_json::json!([]));
            }
            other => panic!("expected object variant, got {other:?}"),
        }

        let array: State = serde_json::from_str(r#"["b", "a", {"k": 1}]"#).unwrap();
        match array {
            State::Array(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], serde_json::json!("b"));
                assert_eq!(items[1], serde_json::json!("a"));
                assert_eq!(items[2]["k"], serde_json::json!(1));
            }
            other => panic!("expected array variant, got {other:?}"),
        }

        // Each variant serializes back to the same JSON shape it came from.
        assert_eq!(
            serde_json::to_value(State::Text("hello".into())).unwrap(),
            serde_json::json!("hello")
        );
        let object: State = serde_json::from_str(r#"{"a": [1, true, null]}"#).unwrap();
        assert_eq!(
            serde_json::to_value(object).unwrap(),
            serde_json::json!({"a": [1, true, null]})
        );
    }

    #[test]
    fn state_rejects_number_bool_and_null() {
        for raw in ["7", "3.5", "true", "false", "null"] {
            let err = match serde_json::from_str::<State>(raw) {
                Ok(state) => panic!("`{raw}` must not deserialize into a State, got {state:?}"),
                Err(err) => err,
            };
            assert!(
                err.to_string().contains("a string, object, or array"),
                "unexpected error for `{raw}`: {err}"
            );
        }
    }
}
