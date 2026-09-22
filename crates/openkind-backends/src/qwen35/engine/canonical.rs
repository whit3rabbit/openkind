//! Structured state canonicalization for deterministic tokenization.

use openkind_core::State;

/// Render wire state to the renderer's state text.
///
/// Structured state is canonicalized by [`canonical_object_text`] /
/// [`canonical_array_text`] so the rendered state — and therefore the model's
/// input tokens and every digest derived from them — depends only on the
/// semantic content of the state, never on key construction order or the
/// serializer's map implementation. This formalizes the output the renderer
/// has always produced and is part of the `state_first` renderer's ordering
/// semantics, not a new renderer contract.
pub(crate) fn state_text(state: &State) -> String {
    match state {
        State::Text(text) => text.clone(),
        State::Object(map) => canonical_object_text(map),
        State::Array(items) => canonical_array_text(items),
    }
}

/// Recursively rebuild one JSON value with every object's keys sorted.
///
/// Sorting is byte-lexicographic over the UTF-8 key bytes (`str` ordering).
/// Arrays keep their order — order is semantic in arrays — and scalars are
/// carried through unchanged, so number formatting stays the serializer's
/// stable shortest-round-trip form.
fn canonical_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => serde_json::Value::Object(canonical_object(map)),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonical_value).collect())
        }
        scalar => scalar.clone(),
    }
}

fn canonical_object(
    map: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut sorted = serde_json::Map::with_capacity(map.len());
    for key in keys {
        sorted.insert(key.to_owned(), canonical_value(&map[key]));
    }
    sorted
}

/// Canonical compact JSON text for one structured-state object.
fn canonical_object_text(map: &serde_json::Map<String, serde_json::Value>) -> String {
    serde_json::to_string(&serde_json::Value::Object(canonical_object(map)))
        .expect("compact JSON serialization of a Value is infallible")
}

/// Canonical compact JSON text for one structured-state array.
fn canonical_array_text(items: &[serde_json::Value]) -> String {
    let canonical: Vec<serde_json::Value> = items.iter().map(canonical_value).collect();
    serde_json::to_string(&canonical).expect("compact JSON serialization of a Value is infallible")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_state_text_passes_text_through_verbatim() {
        assert_eq!(state_text(&State::Text("raw".into())), "raw");
        assert_eq!(state_text(&State::Text(String::new())), "");
    }

    #[test]
    fn structured_state_text_is_canonical_with_sorted_keys() {
        let object = |value: serde_json::Value| State::Object(value.as_object().cloned().unwrap());
        let array = |value: serde_json::Value| State::Array(value.as_array().cloned().unwrap());

        // Keys sorted at every nesting level; empty containers and null are
        // preserved.
        assert_eq!(
            state_text(&object(serde_json::json!({"b": 1, "a": {"y": 2, "x": 3}}))),
            r#"{"a":{"x":3,"y":2},"b":1}"#
        );
        assert_eq!(state_text(&object(serde_json::json!({}))), "{}");
        assert_eq!(state_text(&array(serde_json::json!([]))), "[]");
        assert_eq!(
            state_text(&object(serde_json::json!({"n": null}))),
            r#"{"n":null}"#
        );

        // Array order is semantic and never sorted.
        assert_eq!(
            state_text(&array(serde_json::json!([3, 1, {"b": 1, "a": 2}]))),
            r#"[3,1,{"a":2,"b":1}]"#
        );

        // Byte-lexicographic key ordering: multi-byte UTF-8 keys sort after
        // ASCII keys regardless of code-point or collation expectations.
        assert_eq!(
            state_text(&object(
                serde_json::json!({"unicode_é": "café", "unicode_z": "zürich"})
            )),
            "{\"unicode_z\":\"zürich\",\"unicode_é\":\"café\"}"
        );

        // Number forms are carried through unchanged: integers stay integers
        // (i64 boundaries) and floats keep the serializer's stable
        // shortest-round-trip form, including -0.0 and denormal/extreme
        // magnitudes.
        assert_eq!(
            state_text(&object(serde_json::json!({"i": 9223372036854775807i64}))),
            r#"{"i":9223372036854775807}"#
        );
        assert_eq!(
            state_text(&object(serde_json::json!({"z": -0.0}))),
            r#"{"z":-0.0}"#
        );
        assert_eq!(
            state_text(&object(
                serde_json::json!({"t": 1e-308, "h": 1.7976931348623157e308})
            )),
            r#"{"h":1.7976931348623157e+308,"t":1e-308}"#
        );
    }

    #[test]
    fn structured_state_canonicalization_formalizes_existing_bytes() {
        // The canonical form is byte-identical to what compact serialization
        // of the same value already produced (serde_json's default map sorts
        // keys), so this pins formalization rather than a renderer change.
        let complex = serde_json::json!({
            "z": [1, {"q": null, "b": [[], {}]}, 2.5],
            "a": {"deep": {"beta": -0.0, "alpha": "α"}},
            "m": "café"
        });
        let direct = serde_json::to_string(&complex).expect("direct serialization");
        let canonical = state_text(&State::Object(complex.as_object().cloned().unwrap()));
        assert_eq!(canonical, direct);

        // Key construction order cannot change the canonical text.
        let mut first = serde_json::Map::new();
        first.insert("alpha".into(), serde_json::json!(1));
        first.insert("beta".into(), serde_json::json!(2));
        let mut second = serde_json::Map::new();
        second.insert("beta".into(), serde_json::json!(2));
        second.insert("alpha".into(), serde_json::json!(1));
        assert_eq!(
            state_text(&State::Object(first)),
            state_text(&State::Object(second))
        );
    }
}
