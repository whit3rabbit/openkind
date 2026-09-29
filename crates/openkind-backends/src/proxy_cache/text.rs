//! Canonical text serialization for proxy-cache rows and task identity.
//!
//! A request is embedded by its `state` alone: string states pass through,
//! object and array states are serialized as canonical JSON (sorted keys,
//! compact separators). Instructions and criteria never enter the embedding —
//! they define task identity instead.

use openkind_core::State;

/// The store keeps and the encoder sees at most this many characters.
pub const MAX_TEXT_CHARS: usize = 32_768;

/// Serialize a JSON value with sorted keys and compact separators.
pub fn canonical_json(value: &serde_json::Value) -> String {
    fn write(value: &serde_json::Value, out: &mut String) {
        match value {
            serde_json::Value::Null => out.push_str("null"),
            serde_json::Value::Bool(true) => out.push_str("true"),
            serde_json::Value::Bool(false) => out.push_str("false"),
            serde_json::Value::Number(number) => {
                out.push_str(&number.to_string());
            }
            serde_json::Value::String(text) => {
                push_json_string(text, out);
            }
            serde_json::Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            serde_json::Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (index, key) in keys.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    push_json_string(key, out);
                    out.push(':');
                    write(&map[*key], out);
                }
                out.push('}');
            }
        }
    }

    let mut out = String::new();
    write(value, &mut out);
    out
}

/// JSON string literal with minimal escaping (`"` and `\` plus control
/// characters), matching `serde_json::to_string` byte-for-byte for the
/// characters task fingerprints care about while never allocating through
/// the serde writer stack.
fn push_json_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

/// The embedding text for a request: the `state` alone, truncated to
/// [`MAX_TEXT_CHARS`].
pub fn state_text(state: &State) -> String {
    let full = match state {
        State::Text(text) => text.clone(),
        State::Object(map) => canonical_json(&serde_json::Value::Object(map.clone())),
        State::Array(items) => canonical_json(&serde_json::Value::Array(items.clone())),
    };
    truncate_chars(&full, MAX_TEXT_CHARS)
}

/// Truncate to at most `max_chars` `char`s (not bytes).
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    text.chars().take(max_chars).collect()
}

/// Salted HMAC-free text hash for stores that keep no text: the first 16 hex
/// characters of SHA-256 over `salt || text`.
pub fn text_hash(text: &str, salt: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(salt);
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(16);
    for byte in &digest[..8] {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_keys_and_compacts() {
        assert_eq!(
            canonical_json(&json!({"b": 1, "a": [2, {"z": true, "y": null}]})),
            r#"{"a":[2,{"y":null,"z":true}],"b":1}"#
        );
    }

    #[test]
    fn state_text_passes_strings_and_canonicalizes_objects() {
        assert_eq!(state_text(&State::Text("hello".into())), "hello");
        let map = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            r#"{"b":1,"a":"x"}"#,
        )
        .unwrap();
        assert_eq!(state_text(&State::Object(map)), r#"{"a":"x","b":1}"#);
    }

    #[test]
    fn truncate_counts_characters_not_bytes() {
        assert_eq!(truncate_chars("héllo", 3), "hél");
        assert_eq!(truncate_chars("abc", 5), "abc");
    }

    #[test]
    fn text_hash_is_stable_and_salted() {
        assert_eq!(text_hash("abc", b"s"), text_hash("abc", b"s"));
        assert_ne!(text_hash("abc", b"s"), text_hash("abc", b"t"));
        assert_eq!(text_hash("abc", b"").len(), 16);
    }
}
