//! Renderer for the strands-decider family: the reference
//! `strands_decider.prompting` layout and JSON state serialization.
//!
//! One question row is the `<state>` document followed by the question
//! block — `<question type="...">`, the reference header, the rendered
//! instructions, then per option one line `"{n}. {name} — {description}"`
//! (options numbered from 1, descriptions collapsed to one line) — closing
//! with `</options></question><answer>`. The pointer readout positions are
//! resolved from the tokenizer's offset mapping: each option's last token is
//! the one that has just read the whole option under causal attention, and
//! the pooled position is the final `<answer>` token.
//!
//! The window follows the reference `_fit` policy: the question and its
//! options claim up to [`super::MAX_QUESTION_TOKENS`] tokens first (a
//! longer question is truncated from the front, keeping the tail), and the
//! state receives the rest of the 4096-token window (keep-first).

use tokenizers::Tokenizer;

use crate::families::decoder_logit_qwen35::renderer::{python_json, python_string};
use crate::families::support::FamilyError;
use crate::families::wire::{QuestionPrimitive, UnpackedQuestion};

use super::{MAX_QUESTION_TOKENS, MAX_ROW_TOKENS, NOUL_DEFAULT_CRITERIA};

/// Reference header line per question primitive (`prompting.py`).
fn header(kind: QuestionPrimitive) -> &'static str {
    match kind {
        QuestionPrimitive::Noul => "Decide whether the statement is true of the state.",
        QuestionPrimitive::Choice => "Select exactly one option.",
        QuestionPrimitive::Score => {
            "Rate the state against the ordered levels below (lowest first)."
        }
    }
}

/// Reference wire type tag per question primitive.
fn kind_tag(kind: QuestionPrimitive) -> &'static str {
    match kind {
        QuestionPrimitive::Noul => "noul",
        QuestionPrimitive::Choice => "choice",
        QuestionPrimitive::Score => "score",
    }
}

/// Flatten a state or instruction payload to text, stably.
///
/// Strings are stripped; everything else is emitted as indented JSON so key
/// order and unicode stay deterministic (`prompting.py`:
/// `render_content`, `json.dumps(..., indent=2, ensure_ascii=False)`).
pub fn render_content(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.trim().to_owned(),
        other => python_json_indent(other),
    }
}

/// Serialize a JSON value exactly like Python
/// `json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False)`.
pub(crate) fn python_json_indent(value: &serde_json::Value) -> String {
    fn render(value: &serde_json::Value, level: usize) -> String {
        let pad = "  ".repeat(level);
        let inner_pad = "  ".repeat(level + 1);
        match value {
            serde_json::Value::Array(items) if items.is_empty() => "[]".to_owned(),
            serde_json::Value::Array(items) => format!(
                "[\n{}\n{pad}]",
                items
                    .iter()
                    .map(|item| format!("{inner_pad}{}", render(item, level + 1)))
                    .collect::<Vec<_>>()
                    .join(",\n")
            ),
            serde_json::Value::Object(entries) if entries.is_empty() => "{}".to_owned(),
            serde_json::Value::Object(entries) => format!(
                "{{\n{}\n{pad}}}",
                entries
                    .iter()
                    .map(|(key, value)| format!(
                        "{inner_pad}{}: {}",
                        python_string(key),
                        render(value, level + 1)
                    ))
                    .collect::<Vec<_>>()
                    .join(",\n")
            ),
            other => python_json(other),
        }
    }
    render(value, 0)
}

/// Collapse a description to one line (`" ".join(desc.split())`).
fn collapse_description(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One rendered question: the option `(name, description)` pairs in
/// evaluation order plus the primitive tag the prompt shows.
pub struct QuestionLayout {
    /// Wire primitive the question arrived as.
    pub primitive: QuestionPrimitive,
    /// `(name, description)` pairs in evaluation order, exactly as they
    /// render into the option block.
    pub pairs: Vec<(String, String)>,
}

/// Build the question layout the reference `render_question` produces.
///
/// `Noul` scores the `false`/`true` pair with the caller's criteria when the
/// wire supplies them and the reference defaults otherwise; `Choice` scores
/// the offered keys (our wire map is a hash map, so the sorted label order
/// of [`UnpackedQuestion`] is a declared determinism difference); `Score`
/// scores the ordered levels with the 0-based index as the name.
pub fn question_layout(
    unpacked: &UnpackedQuestion,
    explicit_noul_criteria: bool,
) -> QuestionLayout {
    let pairs = match unpacked.primitive {
        QuestionPrimitive::Noul => {
            let mut descriptions: [Option<&str>; 2] = [None, None];
            if explicit_noul_criteria {
                descriptions = [
                    Some(unpacked.criteria[0].as_str()),
                    Some(unpacked.criteria[1].as_str()),
                ];
            }
            ["false", "true"]
                .iter()
                .enumerate()
                .map(|(index, label)| {
                    let description = descriptions[index]
                        .unwrap_or(NOUL_DEFAULT_CRITERIA[index])
                        .to_owned();
                    ((*label).to_owned(), description)
                })
                .collect()
        }
        QuestionPrimitive::Choice => unpacked
            .labels
            .iter()
            .enumerate()
            .map(|(index, label)| {
                let description = collapse_description(&unpacked.criteria[index]);
                (label.clone(), description)
            })
            .collect(),
        QuestionPrimitive::Score => unpacked
            .labels
            .iter()
            .enumerate()
            .map(|(index, label)| {
                (
                    label.clone(),
                    collapse_description(&unpacked.criteria[index]),
                )
            })
            .collect(),
    };
    QuestionLayout {
        primitive: unpacked.primitive,
        pairs,
    }
}

/// One encoded question row plus its readout offsets (row-relative): the
/// final `<answer>` position and one last-token position per option.
#[derive(Debug, Clone)]
pub struct RenderedRow {
    /// Full row token ids: state tokens then question tokens.
    pub ids: Vec<u32>,
    /// Row-relative position of the final `<answer>` token (the pooled
    /// position).
    pub answer: usize,
    /// Row-relative last-token position per option, in evaluation order.
    pub opts: Vec<usize>,
}

/// Offline renderer for the pinned strands-decider profile.
pub struct StrandsRenderer {
    tokenizer: Tokenizer,
}

impl StrandsRenderer {
    /// Load the digest-verified fast tokenizer.
    pub fn load(tokenizer_path: &std::path::Path) -> Result<Self, FamilyError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(Self { tokenizer })
    }

    /// Encode the state document and one question into a single causal row,
    /// applying the reference question-first window policy.
    pub fn encode_row(
        &self,
        state: &serde_json::Value,
        layout: &QuestionLayout,
        instructions: &serde_json::Value,
    ) -> Result<RenderedRow, FamilyError> {
        let state_text = format!("<state>\n{}\n</state>\n", render_content(state));
        let (question_text, spans) = render_question_text(layout, &render_content(instructions));

        // The question claims the window first; a question longer than its
        // reserve is truncated from the front so the options and the
        // trailing `<answer>` marker survive.
        let question_encoding = self
            .tokenizer
            .encode(question_text.as_str(), false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let question_ids = question_encoding.get_ids().to_vec();
        let question_offsets = question_encoding.get_offsets();
        let reserve = question_ids.len().min(MAX_QUESTION_TOKENS);
        let cut = question_ids.len() - reserve;
        let question_ids = &question_ids[cut..];
        let question_offsets = &question_offsets[cut..];

        // Option read positions from the offset mapping: the last surviving
        // token inside each option's byte span. Zero-width offsets (added
        // tokens) never carry an option boundary.
        let mut opts = Vec::with_capacity(spans.len());
        for &(start, end) in &spans {
            let mut last = None;
            for (index, &(lo, hi)) in question_offsets.iter().enumerate() {
                if hi <= lo {
                    continue;
                }
                if lo >= start && hi <= end {
                    last = Some(index);
                }
            }
            let last = last.ok_or_else(|| {
                FamilyError::InvalidInput(format!(
                    "option span [{start},{end}) has no tokens left; the prompt was \
                     truncated through its option list"
                ))
            })?;
            opts.push(last);
        }

        // The state receives the rest of the window, keep-first.
        let state_budget = MAX_ROW_TOKENS.saturating_sub(reserve).max(1);
        let state_encoding = self
            .tokenizer
            .encode(state_text.as_str(), true)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let mut state_ids = state_encoding.get_ids().to_vec();
        if state_ids.len() > state_budget {
            state_ids.truncate(state_budget);
        }

        let state_len = state_ids.len();
        let mut ids = Vec::with_capacity(state_len + question_ids.len());
        ids.extend_from_slice(&state_ids);
        ids.extend_from_slice(question_ids);
        if ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "encoded question row is empty".to_owned(),
            ));
        }
        Ok(RenderedRow {
            answer: ids.len() - 1,
            opts: opts.into_iter().map(|index| state_len + index).collect(),
            ids,
        })
    }
}

/// Render the question block and each option line's byte span within it
/// (`prompting.py`: `render_question` + `_option_block`).
fn render_question_text(
    layout: &QuestionLayout,
    instructions: &str,
) -> (String, Vec<(usize, usize)>) {
    let prefix = format!(
        "<question type=\"{}\">\n{}\n{}\n<options>\n",
        kind_tag(layout.primitive),
        header(layout.primitive),
        instructions
    );
    let base = prefix.len();
    let mut block = String::new();
    let mut spans = Vec::with_capacity(layout.pairs.len());
    for (index, (name, description)) in layout.pairs.iter().enumerate() {
        if index > 0 {
            block.push('\n');
        }
        let start = base + block.len();
        block.push_str(&format!("{}. {}", index + 1, name));
        let description = collapse_description(description);
        if !description.is_empty() {
            block.push_str(" \u{2014} ");
            block.push_str(&description);
        }
        spans.push((start, base + block.len()));
    }
    let text = format!("{prefix}{block}\n</options>\n</question>\n<answer>");
    (text, spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_rendering_strips_strings_and_indents_structures() {
        assert_eq!(render_content(&serde_json::json!("  hi  ")), "hi");
        assert_eq!(render_content(&serde_json::json!({})), "{}");
        assert_eq!(render_content(&serde_json::json!([])), "[]");
        assert_eq!(
            render_content(&serde_json::json!({"a": 1, "b": ["x", "y"]})),
            "{\n  \"a\": 1,\n  \"b\": [\n    \"x\",\n    \"y\"\n  ]\n}"
        );
    }

    #[test]
    fn question_text_numbers_options_and_records_byte_spans() {
        let layout = QuestionLayout {
            primitive: QuestionPrimitive::Choice,
            pairs: vec![
                ("alpha".to_owned(), String::new()),
                ("beta".to_owned(), "the second option".to_owned()),
            ],
        };
        let (text, spans) = render_question_text(&layout, "Pick one.");
        assert_eq!(
            text,
            "<question type=\"choice\">\nSelect exactly one option.\nPick one.\n<options>\n\
             1. alpha\n2. beta \u{2014} the second option\n</options>\n</question>\n<answer>"
        );
        assert_eq!(spans.len(), 2);
        let option_lines: Vec<String> = spans
            .iter()
            .map(|&(start, end)| text[start..end].to_owned())
            .collect();
        assert_eq!(
            option_lines,
            vec!["1. alpha", "2. beta \u{2014} the second option"]
        );
    }

    #[test]
    fn descriptions_collapse_to_one_line() {
        assert_eq!(collapse_description("  a\n b\t c "), "a b c");
        assert_eq!(collapse_description("   "), "");
    }
}
