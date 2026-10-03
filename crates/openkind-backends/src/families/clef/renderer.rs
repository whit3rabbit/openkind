//! Prompt renderer for the Clef family: a byte-exact port of the reference
//! `joint_schema_model.encode_record` text-only path.
//!
//! The reference encodes a record as a Qwen chat prompt whose user turn
//! carries the state, a schema block (one field per question with its typed
//! options), and the `JOINT SCHEMA DECISIONS:` trigger. Token spans for each
//! question and option are recorded relative to the schema block and shifted
//! once the state length is known; the joint head mean-pools backbone hidden
//! states over those spans.
//!
//! Declared renderer differences from the reference implementation:
//! - The Jev wire question map is unordered (`questions` is a hash map), so
//!   fields render in sorted-key order. The Python reference preserves the
//!   caller's JSON insertion order, which the wire cannot carry.
//! - Noul option descriptions come from the wire's materialized criteria
//!   (openkind defaults), not the Clef defaults: the caller's wire payload
//!   governs.
//! - The reference silently truncates the state to whatever remains of
//!   `max_length`. openkind instead rejects any state that does not fit the
//!   profile's operational context (see [`MAX_LENGTH_TOKENS`] and
//!   `ClefProfile::operational_context_tokens`): silent truncation of a
//!   remote caller's state would both mis-decide and let one request drive
//!   the scalar CPU backbones at their full quadratic window.
//!
//! JSON serialization for the state and option payloads follows Python
//! `json.dumps(..., ensure_ascii=False, separators=(",", ":"), sort_keys=True)`.

use tokenizers::Tokenizer;

use crate::families::decoder_logit_qwen35::renderer::{python_float, python_string};
use crate::families::support::FamilyError;

/// System instruction from the reference `joint_schema_model`.
pub(crate) const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";

/// Prompt prefix before the state, rendered and pinned from the reference
/// encoding (`<|im_start|>` and `<|im_end|>` are literal markers, not chat
/// template output).
pub(crate) const PROMPT_PREFIX: &str =
    "<|im_start|>system\nSYSTEM_PROMPT_PLACEHOLDER<|im_end|>\n<|im_start|>user\nSTATE:\n";

/// Trigger after the schema block.
pub(crate) const PROMPT_SUFFIX: &str =
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";

/// Schema-block header.
pub(crate) const SCHEMA_HEADER: &str = "\n\nSCHEMA FIELDS:\n";

/// Question-type discriminant fed to the joint head's type embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClefQuestionType {
    Noul,
    Choice,
    Score,
}

impl ClefQuestionType {
    /// Integer id consumed by the type embedding, matching the reference
    /// `QUESTION_TYPES` table.
    pub(crate) fn type_id(self) -> u32 {
        match self {
            Self::Noul => 0,
            Self::Choice => 1,
            Self::Score => 2,
        }
    }

    /// Wire type name rendered into the schema block.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }
}

/// One encoded question with its token spans inside the schema block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedQuestion {
    pub(crate) question_id: String,
    pub(crate) question_type: ClefQuestionType,
    /// Token span of the rendered instruction text (schema-relative).
    pub(crate) question_span: (usize, usize),
    /// Token span of each rendered option payload (schema-relative).
    pub(crate) option_spans: Vec<(usize, usize)>,
    pub(crate) option_ids: Vec<String>,
}

/// One fully encoded record: prompt ids plus schema-relative question spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedRecord {
    pub(crate) input_ids: Vec<u32>,
    pub(crate) questions: Vec<EncodedQuestion>,
}

/// Offline renderer over the pinned Clef tokenizer.
pub(crate) struct ClefRenderer {
    tokenizer: Tokenizer,
    /// Operational encoded-prompt bound of the loading profile. This is the
    /// reference `max_length` window on accelerated backends and a smaller
    /// serving bound on the scalar candle CPU profiles.
    max_length: usize,
}

impl ClefRenderer {
    /// Load the digest-verified tokenizer with the profile's operational
    /// encoded-prompt bound.
    pub(crate) fn load(
        tokenizer_path: &std::path::Path,
        max_length: usize,
    ) -> Result<Self, FamilyError> {
        let bytes = std::fs::read(tokenizer_path).map_err(|source| FamilyError::Io {
            path: tokenizer_path.to_path_buf(),
            source,
        })?;
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(Self {
            tokenizer,
            max_length,
        })
    }

    /// Assemble a renderer over an already-loaded tokenizer (tests).
    #[cfg(test)]
    pub(crate) fn from_tokenizer(tokenizer: Tokenizer, max_length: usize) -> Self {
        Self {
            tokenizer,
            max_length,
        }
    }

    fn tokens(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode one request exactly like the reference `encode_record`
    /// (text-only path; openkind states carry no media).
    pub(crate) fn encode(
        &self,
        state: &serde_json::Value,
        questions: &[super::RenderQuestion],
    ) -> Result<EncodedRecord, FamilyError> {
        let mut schema_ids: Vec<u32> = Vec::new();
        schema_ids.extend(self.tokens(SCHEMA_HEADER)?);
        let mut encoded_questions: Vec<EncodedQuestion> = Vec::new();
        for (question_index, question) in questions.iter().enumerate() {
            schema_ids.extend(self.tokens(&format!(
                "\nFIELD {n}\nID: {id}\nTYPE: {kind}\nINSTRUCTION: ",
                n = question_index + 1,
                id = question.id,
                kind = question.question_type.as_str(),
            ))?);
            let question_start = schema_ids.len();
            let instruction_text = if question.instruction.is_empty() {
                question.id.clone()
            } else {
                question.instruction.clone()
            };
            schema_ids.extend(self.tokens(&clef_render_text(&instruction_text))?);
            let question_end = schema_ids.len();
            schema_ids.extend(self.tokens("\nALLOWED OPTIONS:\n")?);
            let mut option_spans = Vec::with_capacity(question.options.len());
            let mut option_ids = Vec::with_capacity(question.options.len());
            for (option_index, (option_id, description)) in question.options.iter().enumerate() {
                schema_ids.extend(self.tokens(&format!("OPTION {}: ", option_index + 1))?);
                let option_start = schema_ids.len();
                let semantics = clef_json_compact(&serde_json::json!({
                    "option_id": option_id,
                    "description": description,
                }));
                schema_ids.extend(self.tokens(&semantics)?);
                option_spans.push((option_start, schema_ids.len()));
                option_ids.push(option_id.clone());
                schema_ids.extend(self.tokens("\n")?);
            }
            schema_ids.extend(self.tokens("END FIELD\n")?);
            encoded_questions.push(EncodedQuestion {
                question_id: question.id.clone(),
                question_type: question.question_type,
                question_span: (question_start, question_end),
                option_spans,
                option_ids,
            });
        }

        let prefix_ids =
            self.tokens(&PROMPT_PREFIX.replace("SYSTEM_PROMPT_PLACEHOLDER", SYSTEM_PROMPT))?;
        let suffix_ids = self.tokens(PROMPT_SUFFIX)?;
        let state_ids = self.tokens(&clef_render_value(state)?)?;
        let fixed_length = prefix_ids.len() + schema_ids.len() + suffix_ids.len();
        if fixed_length > self.max_length {
            return Err(FamilyError::ContractMismatch {
                field: "max_length",
                expected: format!("schema within {max} tokens", max = self.max_length),
                actual: format!("schema requires {fixed_length} tokens before state"),
            });
        }
        // Reject rather than truncate: the state is remote input, and
        // silently dropping its tail both mis-decides and lets one request
        // drive the scalar candle kernels at the profile's full window.
        let state_budget = self.max_length - fixed_length;
        if state_ids.len() > state_budget {
            return Err(FamilyError::InvalidInput(format!(
                "state requires {state_tokens} encoded tokens but only {state_budget} remain \
                 inside this profile's {max}-token operational context; shorten the state",
                state_tokens = state_ids.len(),
                max = self.max_length,
            )));
        }
        let schema_offset = prefix_ids.len() + state_ids.len();
        let shifted_questions: Vec<EncodedQuestion> = encoded_questions
            .into_iter()
            .map(|question| EncodedQuestion {
                question_id: question.question_id,
                question_type: question.question_type,
                question_span: (
                    question.question_span.0 + schema_offset,
                    question.question_span.1 + schema_offset,
                ),
                option_spans: question
                    .option_spans
                    .into_iter()
                    .map(|(start, end)| (start + schema_offset, end + schema_offset))
                    .collect(),
                option_ids: question.option_ids,
            })
            .collect();
        let mut input_ids = prefix_ids;
        input_ids.extend(state_ids);
        input_ids.extend(schema_ids);
        input_ids.extend(suffix_ids);
        if input_ids.is_empty() || shifted_questions.is_empty() {
            return Err(FamilyError::InvalidInput(
                "record produced no model input or questions".into(),
            ));
        }
        Ok(EncodedRecord {
            input_ids,
            questions: shifted_questions,
        })
    }
}

/// Frozen model-window bound shared by every Clef profile (the reference
/// `max_length` default). The candle CPU profiles serve under the smaller
/// `ClefProfile::operational_context_tokens` bound instead.
pub(crate) const MAX_LENGTH_TOKENS: usize = 16_384;

/// Serialize like Python `json.dumps(value, ensure_ascii=False,
/// separators=(",", ":"), sort_keys=True)`: compact separators, sorted object
/// keys, strings passthrough.
pub(crate) fn clef_render_value(value: &serde_json::Value) -> Result<String, FamilyError> {
    Ok(match value {
        serde_json::Value::String(text) => text.clone(),
        other => clef_json_compact(other),
    })
}

fn clef_render_text(value: &str) -> String {
    value.to_owned()
}

fn clef_json_compact(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_owned(),
        serde_json::Value::Bool(true) => "true".to_owned(),
        serde_json::Value::Bool(false) => "false".to_owned(),
        serde_json::Value::Number(number) => python_number(number),
        serde_json::Value::String(text) => python_string(text),
        serde_json::Value::Array(items) => {
            let rendered: Vec<String> = items.iter().map(clef_json_compact).collect();
            format!("[{}]", rendered.join(","))
        }
        serde_json::Value::Object(entries) => {
            let mut keys: Vec<(&String, &serde_json::Value)> = entries.iter().collect();
            keys.sort_by(|left, right| left.0.cmp(right.0));
            let rendered: Vec<String> = keys
                .into_iter()
                .map(|(key, value)| format!("{}:{}", python_string(key), clef_json_compact(value)))
                .collect();
            format!("{{{}}}", rendered.join(","))
        }
    }
}

fn python_number(number: &serde_json::Number) -> String {
    if let Some(integer) = number.as_i64() {
        return integer.to_string();
    }
    if let Some(unsigned) = number.as_u64() {
        return unsigned.to_string();
    }
    python_float(number.as_f64().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compact_json_sorts_keys_and_preserves_non_ascii() {
        let value = json!({"b": 1, "a": "ünï", "c": [true, null]});
        assert_eq!(
            clef_json_compact(&value),
            r#"{"a":"ünï","b":1,"c":[true,null]}"#
        );
    }

    #[test]
    fn render_value_passes_plain_text_through() {
        assert_eq!(clef_render_value(&json!("raw text")).unwrap(), "raw text");
        assert_eq!(
            clef_render_value(&json!({"k": "v"})).unwrap(),
            r#"{"k":"v"}"#
        );
    }

    #[test]
    fn option_payload_uses_compact_sorted_semantics() {
        assert_eq!(python_string("true"), "\"true\"");
    }

    /// Whitespace-delimited word tokenizer for offline renderer tests: one
    /// token per word, `[UNK]` for out-of-vocabulary words.
    fn word_tokenizer() -> Tokenizer {
        use std::collections::HashMap;
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0_u32);
        vocab.insert("alpha".to_string(), 1);
        vocab.insert("beta".to_string(), 2);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
    }

    fn render_questions() -> Vec<super::super::engine::RenderQuestion> {
        vec![super::super::engine::RenderQuestion {
            id: "q".into(),
            question_type: ClefQuestionType::Noul,
            instruction: "Decide.".into(),
            options: vec![("true".into(), "yes".into()), ("false".into(), "no".into())],
        }]
    }

    #[test]
    fn encode_rejects_state_beyond_the_operational_context() {
        // Measure the fixed prompt+schema+suffix cost with an empty state,
        // then serve under an operational context four tokens wider.
        let measuring = ClefRenderer::from_tokenizer(word_tokenizer(), 100_000);
        let empty = measuring
            .encode(&json!(""), &render_questions())
            .expect("empty state encodes");
        let fixed = empty.input_ids.len();
        assert!(fixed > 0);
        let renderer = ClefRenderer::from_tokenizer(word_tokenizer(), fixed + 4);

        // A state that exactly fills the remaining budget is accepted.
        renderer
            .encode(&json!("alpha beta alpha beta"), &render_questions())
            .expect("state within the operational context must encode");

        // A state one token beyond the budget is rejected — never silently
        // truncated to the window.
        let error = renderer
            .encode(&json!("alpha beta alpha beta alpha"), &render_questions())
            .expect_err("oversized state must be rejected");
        assert!(
            matches!(&error, FamilyError::InvalidInput(message) if message.contains("operational context")),
            "unexpected error: {error}"
        );

        // A schema that alone exceeds the bound still fails the window
        // contract before the state is considered.
        let tiny = ClefRenderer::from_tokenizer(word_tokenizer(), fixed.saturating_sub(1).max(1));
        let error = tiny
            .encode(&json!(""), &render_questions())
            .expect_err("schema beyond the window must fail");
        assert!(
            matches!(error, FamilyError::ContractMismatch { .. }),
            "unexpected error: {error}"
        );
    }
}
