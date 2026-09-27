//! Renderer for the kev family: the reference `kev.model.encode` layout and
//! the `kev.api` JSON text rendering.
//!
//! One question row is the state document tokens followed by the branch —
//! `<|fim_middle|>`, the rendered instruction, then per option
//! `<|box_start|>` + option tokens + `<|box_end|>`, then `<|fim_suffix|>`.
//! Positions continue across state and branch (one causal row). Caller text
//! never produces delimiter tokens: the reference's rewrite turns
//! `<|name|>` into `<¦name¦>` before tokenizing.

use tokenizers::Tokenizer;

use crate::families::support::FamilyError;

use super::model::QuestionRow;

/// The five Qwen delimiter tokens the reference reuses, resolved from the
/// pinned tokenizer at load.
pub(crate) struct SpecialIds {
    pub state: u32,
    pub question: u32,
    pub option_start: u32,
    pub option_end: u32,
    pub decide: u32,
}

/// Offline renderer for the pinned kev profile.
pub struct KevRenderer {
    tokenizer: Tokenizer,
    special: SpecialIds,
}

/// Rewrite `<|name|>` to `<¦name¦>` so caller text can never produce a
/// delimiter token (the fast tokenizer ignores split-special settings).
fn escape_specials(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'<' && bytes.get(index + 1) == Some(&b'|') {
            let mut cursor = index + 2;
            let mut name_end = None;
            while cursor < bytes.len() {
                let byte = bytes[cursor];
                if byte.is_ascii_alphanumeric() || byte == b'_' {
                    cursor += 1;
                } else if byte == b'|' && bytes.get(cursor + 1) == Some(&b'>') {
                    name_end = Some(cursor);
                    break;
                } else {
                    break;
                }
            }
            if let Some(name_end) = name_end {
                out.push_str("<¦");
                out.push_str(&text[index + 2..name_end]);
                out.push_str("¦>");
                index = name_end + 2;
                continue;
            }
        }
        let ch = text[index..].chars().next().expect("non-empty suffix");
        out.push(ch);
        index += ch.len_utf8();
    }
    out
}

/// Render one JSON value to the text the model sees, mirroring the
/// reference `kev.api.render`: scalars via their string form (`true`/
/// `false` like Python's `str`), list items as `- ` bullets, object keys as
/// `key:` lines with nested values indented two spaces.
pub fn render_json(value: &serde_json::Value) -> String {
    fn render(value: &serde_json::Value, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        match value {
            serde_json::Value::Null => String::new(),
            serde_json::Value::Bool(true) => "True".to_owned(),
            serde_json::Value::Bool(false) => "False".to_owned(),
            serde_json::Value::Number(number) => number.to_string(),
            serde_json::Value::String(text) => text.clone(),
            serde_json::Value::Array(items) => items
                .iter()
                .map(|item| {
                    format!(
                        "{pad}- {}",
                        render(item, indent + 1).trim_start().to_owned()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
            serde_json::Value::Object(map) => map
                .iter()
                .map(|(key, value)| match value {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        format!("{pad}{key}:\n{}", render(value, indent + 1))
                    }
                    _ => format!("{pad}{key}: {}", render(value, indent)),
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
    render(value, 0)
}

/// One offered option's marker text: the key alone, or `key: description`
/// when the caller supplied one.
pub fn option_text(key: &str, description: Option<&str>) -> String {
    match description {
        Some(text) if !text.is_empty() => format!("{key}: {text}"),
        _ => key.to_owned(),
    }
}

impl KevRenderer {
    /// Load the digest-verified fast tokenizer and resolve the delimiter
    /// ids from it (never hardcoded).
    pub fn load(tokenizer_path: &std::path::Path) -> Result<Self, FamilyError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let resolve = |token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| FamilyError::ContractMismatch {
                    field: "tokenizer.delimiters",
                    expected: format!("{token} present"),
                    actual: "missing".to_owned(),
                })
        };
        let special = SpecialIds {
            state: resolve("<|fim_prefix|>")?,
            question: resolve("<|fim_middle|>")?,
            option_start: resolve("<|box_start|>")?,
            option_end: resolve("<|box_end|>")?,
            decide: resolve("<|fim_suffix|>")?,
        };
        Ok(Self { tokenizer, special })
    }

    fn user_tokens(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let encoding = self
            .tokenizer
            .encode(escape_specials(text).as_str(), false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode one question's branch onto the state prefix: one causal row
    /// of state tokens + branch tokens with contiguous positions, and the
    /// readout offsets of the decide and option-end positions.
    pub fn encode_row(
        &self,
        state_tokens: &[u32],
        instruction: &str,
        options: &[String],
    ) -> Result<QuestionRow, FamilyError> {
        if options.is_empty() {
            return Err(FamilyError::InvalidInput(
                "questions need at least one option".to_owned(),
            ));
        }
        let mut ids = Vec::with_capacity(state_tokens.len() + 64);
        ids.extend_from_slice(state_tokens);
        ids.push(self.special.question);
        let instruction_tokens = self.user_tokens(instruction)?;
        ids.extend(instruction_tokens.iter().copied());
        let mut option_ends = Vec::with_capacity(options.len());
        for option in options {
            let option_tokens = self.user_tokens(option)?;
            if option_tokens.is_empty() {
                return Err(FamilyError::InvalidInput(
                    "option text must tokenize to at least one token".to_owned(),
                ));
            }
            ids.push(self.special.option_start);
            ids.extend(option_tokens.iter().copied());
            ids.push(self.special.option_end);
            option_ends.push(ids.len() - 1);
        }
        ids.push(self.special.decide);
        if ids.len() > super::MAX_ROW_TOKENS {
            return Err(FamilyError::InvalidInput(format!(
                "encoded row length {} exceeds frozen maximum {}; truncation is forbidden",
                ids.len(),
                super::MAX_ROW_TOKENS
            )));
        }
        Ok(QuestionRow {
            decide: ids.len() - 1,
            opts: option_ends,
            ids,
        })
    }

    /// Tokenize the state document under the frozen state limit.
    pub fn state_tokens(&self, state: &str) -> Result<Vec<u32>, FamilyError> {
        let mut ids = Vec::with_capacity(state.len() / 3 + 1);
        ids.push(self.special.state);
        ids.extend(self.user_tokens(state)?);
        if ids.len() > super::MAX_STATE_TOKENS {
            return Err(FamilyError::InvalidInput(format!(
                "state length {} tokens exceeds frozen maximum {}; truncation is forbidden",
                ids.len(),
                super::MAX_STATE_TOKENS
            )));
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_rewriting_neutralizes_delimiters() {
        assert_eq!(escape_specials("plain text"), "plain text");
        assert_eq!(escape_specials("<|fim_prefix|>"), "<¦fim_prefix¦>");
        assert_eq!(escape_specials("a <|box_end|> b"), "a <¦box_end¦> b");
        // Incomplete or invalid shapes pass through untouched.
        assert_eq!(escape_specials("<|no close"), "<|no close");
        assert_eq!(escape_specials("<|bad char!|>"), "<|bad char!|>");
    }

    #[test]
    fn json_rendering_matches_the_reference_shapes() {
        assert_eq!(render_json(&serde_json::json!("hi")), "hi");
        assert_eq!(render_json(&serde_json::json!(true)), "True");
        assert_eq!(
            render_json(&serde_json::json!({"a": 1, "b": ["x", "y"]})),
            "a: 1\nb:\n  - x\n  - y"
        );
    }

    #[test]
    fn option_text_uses_description_when_supplied() {
        assert_eq!(option_text("gateway", None), "gateway");
        assert_eq!(
            option_text("gateway", Some("the routing product")),
            "gateway: the routing product"
        );
    }
}
