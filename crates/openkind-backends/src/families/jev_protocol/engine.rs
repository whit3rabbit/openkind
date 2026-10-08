//! Decision-engine adapter for the JEV model (Qwen3.5 backbone).
//!
//! Every question is one independent forward over its rendered prompt: the
//! backbone returns the final-position logits of the option tokens, the
//! checkpoint's bias and per-kind temperature turn them into probabilities,
//! and the host maps those onto the Jev wire answer. Nothing is generated and
//! nothing is retained between requests. The backbone sits behind
//! [`JevForward`] so the adapter is tested without MLX.

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{
    Answer, ModelInfo, Question, State, SystemRequest, SystemResponse, Usage, WireHashState,
};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::families::support::{
    derive_profile_id, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
    FamilyLimits,
};
use crate::families::wire::{
    answer_from_probabilities, unpack_question, QuestionPrimitive, UnpackedQuestion,
};

use super::{
    choice_option_text, pins, render_prompt, ChoiceLabels, DecisionConfig, DecisionKind, Protocol,
    MAX_CHOICE_OPTIONS,
};

/// Backend identifier reported for the MLX 8-bit profile.
pub const BACKEND_ID: &str = "jev-27b-vl-mlx-8bit";

/// Longest rendered prompt the engine accepts. Attention scores are
/// `[heads, tokens, tokens]` in FP32 per full-attention layer, so the bound
/// keeps one request's transient memory in the low single-digit gigabytes.
pub const MAX_PROMPT_TOKENS: usize = 8_192;

/// Final-position logits of selected vocabulary rows for one prompt.
pub trait JevForward: Send + Sync {
    /// Run the backbone over `input_ids` and return, for each of `token_ids`,
    /// the logit at the last position.
    fn option_logits(&self, input_ids: &[u32], token_ids: &[u32]) -> Result<Vec<f64>, FamilyError>;
}

/// Blocking JEV evaluator over one loaded backbone.
pub struct JevEngine {
    forward: Arc<dyn JevForward>,
    tokenizer: Tokenizer,
    config: DecisionConfig,
    labels: ChoiceLabels,
    metadata: ModelInfo,
}

impl JevEngine {
    /// Wrap a backbone, tokenizer, and validated config as a bounded engine.
    pub fn bounded(
        forward: Arc<dyn JevForward>,
        tokenizer: Tokenizer,
        config: DecisionConfig,
        limits: FamilyLimits,
    ) -> Result<BoundedFamilyEngine, FamilyError> {
        let labels = ChoiceLabels::derive(|text| {
            tokenizer
                .encode(text, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| FamilyError::Tokenizer(error.to_string()))
        })?;
        Self::from_parts(forward, tokenizer, config, labels, limits)
    }

    /// Like [`Self::bounded`] with caller-derived choice labels.
    pub(crate) fn from_parts(
        forward: Arc<dyn JevForward>,
        tokenizer: Tokenizer,
        config: DecisionConfig,
        labels: ChoiceLabels,
        limits: FamilyLimits,
    ) -> Result<BoundedFamilyEngine, FamilyError> {
        let evaluator = Self::evaluator(forward, tokenizer, config, labels)?;
        Ok(BoundedFamilyEngine::new(Arc::new(evaluator), limits))
    }

    /// The unbounded evaluator behind [`Self::from_parts`].
    pub(crate) fn evaluator(
        forward: Arc<dyn JevForward>,
        tokenizer: Tokenizer,
        config: DecisionConfig,
        labels: ChoiceLabels,
    ) -> Result<Self, FamilyError> {
        if labels.names().len() < MAX_CHOICE_OPTIONS
            || labels.token_ids().len() != labels.names().len()
        {
            return Err(FamilyError::ContractMismatch {
                field: "choice_labels",
                expected: MAX_CHOICE_OPTIONS.to_string(),
                actual: labels.names().len().to_string(),
            });
        }
        let metadata = ModelInfo {
            name: format!(
                "jev:{}",
                derive_profile_id("jev", pins::REPOSITORY, pins::REVISION)
            ),
            description: format!(
                "{} (Qwen3.5 hybrid) affine 8-bit MLX conversion {}; options are read from \
                 final-position logits in one pass. Source {}@{}",
                pins::SOURCE_REPOSITORY
                    .rsplit('/')
                    .next()
                    .unwrap_or(pins::SOURCE_REPOSITORY),
                pins::REVISION,
                pins::SOURCE_REPOSITORY,
                pins::SOURCE_REVISION,
            ),
            release_date: "2026-10-08".to_owned(),
        };
        Ok(Self {
            forward,
            tokenizer,
            config,
            labels,
            metadata,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Answer one question, returning its answer and prompt token count.
    fn answer_question(
        &self,
        id: &str,
        question: &Question,
        state: &str,
        control: &FamilyControl,
    ) -> Result<(Answer, usize), FamilyError> {
        control.check()?;
        let unpacked = unpack_question(id, question)?;
        let kind = DecisionKind::from(unpacked.primitive);
        let options = prompt_options(question, &unpacked)?;
        let instruction = instruction_text(question, id);
        let prompt = render_prompt(
            Protocol::Jev,
            kind,
            state,
            &instruction,
            &options,
            &self.labels,
        )?;
        let input_ids = self.encode(&prompt)?;
        if input_ids.is_empty() || input_ids.len() > MAX_PROMPT_TOKENS {
            return Err(FamilyError::InvalidInput(format!(
                "question `{id}` renders {} tokens, outside 1..={MAX_PROMPT_TOKENS}",
                input_ids.len()
            )));
        }
        let token_ids = self
            .config
            .jev_token_ids(kind, options.len(), self.labels.token_ids())?;
        control.check()?;
        let gathered = self.forward.option_logits(&input_ids, &token_ids)?;
        if gathered.len() != token_ids.len() {
            return Err(FamilyError::InvalidInput(format!(
                "backbone returned {} logits for {} options",
                gathered.len(),
                token_ids.len()
            )));
        }
        let probabilities = self.config.jev_probabilities(kind, &gathered)?;
        Ok((
            answer_from_probabilities(&unpacked, &probabilities)?,
            input_ids.len(),
        ))
    }
}

impl FamilyEvaluator for JevEngine {
    fn backend_id(&self) -> &str {
        BACKEND_ID
    }

    fn model_metadata(&self) -> ModelInfo {
        self.metadata.clone()
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        control.check()?;
        let state = state_text(&request.state)?;
        // Questions evaluate in sorted-id order so usage and failures are
        // deterministic; each is independent of the others.
        let mut ordered: Vec<(&String, &Question)> = request.questions.iter().collect();
        ordered.sort_by(|left, right| left.0.cmp(right.0));
        let mut answers: HashMap<String, Answer, WireHashState> =
            HashMap::with_capacity_and_hasher(ordered.len(), Default::default());
        let mut input_tokens = 0_usize;
        for (id, question) in ordered {
            let (answer, tokens) = self.answer_question(id, question, &state, control)?;
            input_tokens = input_tokens.saturating_add(tokens);
            answers.insert(id.clone(), answer);
        }
        Ok(SystemResponse {
            model: request.model,
            answers,
            usage: Usage {
                input_tokens: u32::try_from(input_tokens).unwrap_or(u32::MAX),
                output_tokens: 0,
            },
        })
    }
}

/// Per-option prompt text in evaluation order.
///
/// Noul renders `false`, `true`; score renders the six bare levels (the rubric
/// text is response metadata only); choice renders `key: criterion` or the bare
/// key. Fails closed where the reference does: boolean questions take no
/// criteria and score questions need exactly six levels.
fn prompt_options(
    question: &Question,
    unpacked: &UnpackedQuestion,
) -> Result<Vec<String>, FamilyError> {
    match (question, unpacked.primitive) {
        (Question::Noul(noul), QuestionPrimitive::Noul) => {
            if noul.criteria.is_some() {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{}`: JEV boolean questions take no criteria",
                    unpacked.id
                )));
            }
            Ok(vec!["false".to_owned(), "true".to_owned()])
        }
        (_, QuestionPrimitive::Score) => {
            if unpacked.labels.len() != 6 {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{}`: JEV score questions need six levels, found {}",
                    unpacked.id,
                    unpacked.labels.len()
                )));
            }
            Ok(unpacked.labels.clone())
        }
        (_, QuestionPrimitive::Choice) => {
            if unpacked.labels.len() > MAX_CHOICE_OPTIONS {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{}`: JEV choice questions take at most {MAX_CHOICE_OPTIONS} options",
                    unpacked.id
                )));
            }
            Ok(unpacked
                .labels
                .iter()
                .zip(&unpacked.criteria)
                .map(|(label, criterion)| choice_option_text(label, criterion))
                .collect())
        }
        _ => Err(FamilyError::InvalidInput(format!(
            "question `{}` does not match its declared kind",
            unpacked.id
        ))),
    }
}

/// The instruction as the reference renders it: strings pass through, other
/// JSON is dumped Python-style, and an absent or empty instruction falls back
/// to the question id.
fn instruction_text(question: &Question, id: &str) -> String {
    let instructions = match question {
        Question::Noul(question) => &question.instructions,
        Question::Choice(question) => &question.instructions,
        Question::Score(question) => &question.instructions,
    };
    match instructions {
        Value::Null => id.to_owned(),
        Value::String(text) if text.is_empty() => id.to_owned(),
        Value::String(text) => text.clone(),
        other => python_json(other),
    }
}

/// The state as the reference renders it. Text passes through; structured
/// states use Python `json.dumps(..., ensure_ascii=False)` formatting. A
/// wire array is a JSON value here, not the reference's list of text and image
/// parts, so it is dumped whole rather than concatenated.
pub(super) fn state_text(state: &State) -> Result<String, FamilyError> {
    Ok(match state {
        State::Text(text) => text.clone(),
        State::Object(map) => python_json(&Value::Object(map.clone())),
        State::Array(items) => python_json(&Value::Array(items.clone())),
    })
}

/// Python `json.dumps(value, ensure_ascii=False)` formatting: `", "` and
/// `": "` separators, non-ASCII text kept literal. Object keys follow the
/// map's order (sorted for `serde_json`'s default map).
pub(super) fn python_json(value: &Value) -> String {
    let mut out = String::new();
    write_json(value, &mut out);
    out
}

fn write_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_string(key, out);
                out.push_str(": ");
                write_json(item, out);
            }
            out.push('}');
        }
    }
}

fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}
