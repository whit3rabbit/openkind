//! Prompt renderer and label-token contract for the pinned `decider`
//! profiles.
//!
//! The bytes replicate the reference `decider` package exactly (plain
//! state-first layout, `_NoShuffle` serving path): a `Context:\n<state>`
//! block truncated keep-first to the state cap, then per row
//! `\n\nQuestion: <text>\nOptions:` with `(A) text` lines and the
//! `\nAnswer: (` slot. Up to ten options render as one tokenized string;
//! wider sets build the option lines from token ids so every label stays a
//! single token (`A..Z` then the first single-token two-letter uppercase
//! strings, 255 entries). State objects and arrays serialize as Python
//! `json.dumps(..., ensure_ascii=False)` with `_index` annotations on arrays
//! of at least [`DeciderProfile::annotate_min`] elements.

use tokenizers::Tokenizer;

use crate::families::decoder_logit_qwen35::renderer::python_json;
use crate::families::support::FamilyError;

use super::{DeciderProfile, NARROW_OPTIONS};

/// Width of the label head: `A..Z` plus the first single-token two-letter
/// uppercase strings (`AA`, `AB`, ...) up to 255 entries.
pub(crate) const MAX_LABELS: usize = 255;

/// One rendered scoring row: prompt ids whose final token is the answer slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedRow {
    prompt_ids: Vec<u32>,
}

impl RenderedRow {
    /// Prompt token ids ending at the answer slot.
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }
}

/// Offline renderer for the pinned `decider` profiles.
pub struct DeciderRenderer {
    tokenizer: Tokenizer,
    /// Token ids of the [`MAX_LABELS`] label strings, in reference
    /// `label_table` order (the first [`NARROW_OPTIONS`] are `A..J`).
    label_ids: Vec<u32>,
    max_state_tokens: usize,
    max_row_tokens: usize,
    annotate_min: usize,
}

impl DeciderRenderer {
    /// Load a digest-verified tokenizer and resolve the label-token contract.
    ///
    /// The first [`NARROW_OPTIONS`] labels must be the single tokens `A..J`
    /// (the reference asserts the same) and every kept label must tokenize to
    /// exactly one distinct token.
    pub fn load(
        tokenizer_path: &std::path::Path,
        profile: &DeciderProfile,
    ) -> Result<Self, FamilyError> {
        let bytes = std::fs::read(tokenizer_path).map_err(|source| FamilyError::Io {
            path: tokenizer_path.to_path_buf(),
            source,
        })?;
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let mut names: Vec<String> = (b'A'..=b'Z')
            .map(|letter| (letter as char).to_string())
            .collect();
        for first in b'A'..=b'Z' {
            for second in b'A'..=b'Z' {
                names.push(format!("{}{}", first as char, second as char));
            }
        }
        let mut label_ids = Vec::with_capacity(MAX_LABELS);
        for name in names {
            if label_ids.len() == MAX_LABELS {
                break;
            }
            let encoding = tokenizer
                .encode(name.as_str(), false)
                .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
            let ids = encoding.get_ids();
            if ids.len() == 1 {
                label_ids.push(ids[0]);
            }
        }
        if label_ids.len() != MAX_LABELS {
            return Err(FamilyError::ContractMismatch {
                field: "label_token_contract",
                expected: format!("{MAX_LABELS} single-token uppercase labels"),
                actual: format!("only {} single-token labels", label_ids.len()),
            });
        }
        if label_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != label_ids.len()
        {
            return Err(FamilyError::ContractMismatch {
                field: "label_token_contract",
                expected: "distinct label tokens".to_owned(),
                actual: "duplicate label tokens".to_owned(),
            });
        }
        for (index, letter) in (b'A'..b'A' + NARROW_OPTIONS as u8).enumerate() {
            let text = (letter as char).to_string();
            let encoding = tokenizer
                .encode(text.as_str(), false)
                .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
            if encoding.get_ids() != [label_ids[index]] {
                return Err(FamilyError::ContractMismatch {
                    field: "label_token_contract",
                    expected: format!("label {text} is the {}th table entry", index + 1),
                    actual: "mismatch".to_owned(),
                });
            }
        }
        Ok(Self {
            tokenizer,
            label_ids,
            max_state_tokens: profile.max_state_tokens,
            max_row_tokens: profile.max_row_tokens,
            annotate_min: profile.annotate_min,
        })
    }

    /// Tokenize and truncate the request state once per request (keep-first,
    /// as the reference `build` does for `Context:\n<state>`).
    pub fn state_ids(&self, state: &serde_json::Value) -> Result<Vec<u32>, FamilyError> {
        let text = match state {
            serde_json::Value::String(text) => text.clone(),
            value => python_json(&annotate_indices(value, self.annotate_min)),
        };
        let encoding = self
            .tokenizer
            .encode(format!("Context:\n{text}").as_str(), false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(self.max_state_tokens);
        Ok(ids)
    }

    /// Render one scoring row: the state context plus one question block.
    ///
    /// `options` are the option texts in render order; the row ends at the
    /// `Answer: (` slot.
    pub fn render_row(
        &self,
        state_ids: &[u32],
        question: &str,
        options: &[String],
    ) -> Result<RenderedRow, FamilyError> {
        if question.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "question instruction must contain non-whitespace text".to_owned(),
            ));
        }
        if options.len() < 2 {
            return Err(FamilyError::InvalidInput(format!(
                "a row needs at least 2 options, found {}",
                options.len()
            )));
        }
        if options.len() > MAX_LABELS {
            return Err(FamilyError::InvalidInput(format!(
                "a row scores at most {MAX_LABELS} options, found {}",
                options.len()
            )));
        }
        if options.iter().any(|option| option.trim().is_empty()) {
            return Err(FamilyError::InvalidInput(
                "option texts must contain non-whitespace text".to_owned(),
            ));
        }
        let mut ids = state_ids.to_vec();
        let head = format!("\n\nQuestion: {question}\nOptions:");
        if options.len() <= NARROW_OPTIONS {
            // The reference keeps the narrow rendering tokenized as one
            // string, unchanged since v1.
            let block = head
                + options
                    .iter()
                    .enumerate()
                    .map(|(index, option)| format!("\n({}) {option}", (b'A' + index as u8) as char))
                    .collect::<String>()
                    .as_str()
                + "\nAnswer: (";
            let encoding = self
                .tokenizer
                .encode(block.as_str(), false)
                .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
            ids.extend_from_slice(encoding.get_ids());
        } else {
            // Wide rendering builds the option lines from ids so every label
            // stays one token: "\n(" + <label> + ") text".
            let head_ids = self
                .tokenizer
                .encode(head.as_str(), false)
                .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
            let open_ids = self
                .tokenizer
                .encode("\n(", false)
                .map_err(|error| FamilyError::Tokenizer(error.to_string()))?
                .get_ids()
                .to_vec();
            ids.extend_from_slice(head_ids.get_ids());
            for (index, option) in options.iter().enumerate() {
                let text_ids = self
                    .tokenizer
                    .encode(format!(") {option}").as_str(), false)
                    .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
                ids.extend_from_slice(&open_ids);
                ids.push(self.label_ids[index]);
                ids.extend_from_slice(text_ids.get_ids());
            }
            let tail_ids = self
                .tokenizer
                .encode("\nAnswer: (", false)
                .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
            ids.extend_from_slice(tail_ids.get_ids());
        }
        if ids.len() > self.max_row_tokens {
            return Err(FamilyError::InvalidInput(format!(
                "rendered row length {} exceeds frozen maximum {}; truncation is forbidden",
                ids.len(),
                self.max_row_tokens
            )));
        }
        Ok(RenderedRow { prompt_ids: ids })
    }

    /// Label token ids for one row's options (the first `count` table
    /// entries; the first [`NARROW_OPTIONS`] are `A..J`).
    pub fn label_ids_for(&self, count: usize) -> &[u32] {
        &self.label_ids[..count]
    }

    /// Token count of one rendered row for usage accounting.
    pub fn row_token_count(&self, row: &RenderedRow) -> u32 {
        u32::try_from(row.prompt_ids().len()).unwrap_or(u32::MAX)
    }
}

/// Rewrite long arrays with their element positions, as the reference
/// `annotate_indices` does: a path such as `records[47].text` becomes a
/// lookup instead of a counting exercise.
fn annotate_indices(value: &serde_json::Value, min_len: usize) -> serde_json::Value {
    match value {
        serde_json::Value::Array(items) => {
            if items.len() >= min_len {
                serde_json::Value::Array(
                    items
                        .iter()
                        .enumerate()
                        .map(|(index, item)| match annotate_indices(item, min_len) {
                            serde_json::Value::Object(map) => {
                                let mut annotated = serde_json::Map::new();
                                annotated.insert("_index".to_owned(), serde_json::json!(index));
                                annotated.extend(map);
                                serde_json::Value::Object(annotated)
                            }
                            annotated => serde_json::json!({"_index": index, "value": annotated}),
                        })
                        .collect(),
                )
            } else {
                serde_json::Value::Array(
                    items
                        .iter()
                        .map(|item| annotate_indices(item, min_len))
                        .collect(),
                )
            }
        }
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .map(|(key, item)| (key.clone(), annotate_indices(item, min_len)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn annotate_indices_writes_positions_into_long_arrays_only() {
        let annotated = annotate_indices(&json!({"records": [{"text": "a"}, {"text": "b"}]}), 8);
        assert_eq!(
            annotated["records"][0],
            json!({"text": "a"}),
            "short arrays stay untouched"
        );
        let long: Vec<serde_json::Value> = (0..10).map(|index| json!({"text": index})).collect();
        let annotated = annotate_indices(&json!({"records": long}), 8);
        assert_eq!(
            annotated["records"][3],
            json!({"_index": 3, "text": 3}),
            "long dict arrays gain _index"
        );
        let scalars: Vec<serde_json::Value> = (0..9).map(serde_json::Value::from).collect();
        let annotated = annotate_indices(&json!(scalars), 8);
        assert_eq!(annotated[4], json!({"_index": 4, "value": 4}));
        // Nested arrays annotate recursively.
        let inner: Vec<serde_json::Value> = (0..8).map(|index| json!({"c": index})).collect();
        let nested = annotate_indices(&json!({"a": [{"b": inner}]}), 8);
        assert_eq!(nested["a"][0]["b"][7], json!({"_index": 7, "c": 7}));
    }
}
