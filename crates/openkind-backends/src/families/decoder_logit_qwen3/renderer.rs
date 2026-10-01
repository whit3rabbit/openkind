//! Prompt renderer and letter-token contract for the raw `decoder-logit-qwen3`
//! controls.
//!
//! The decision payload reuses the `jevk5` letter-pass bytes (system
//! instruction, Python-`json.dumps`-shaped payload) so the controls are
//! methodology-comparable with the fine-tuned `decoder-logit-qwen35`
//! profiles; the chat-template wrapper is the pinned checkpoint's own
//! template with thinking off, which differs per profile family: the
//! thinking-capable 0.6B/1.7B templates close the assistant turn with the
//! empty `<think>` block, the non-thinking-only 2507 template ends at the
//! bare generation prompt. Because the wire `criteria` map is a hash map,
//! options render in the profile's deterministic sorted label order.

use tokenizers::Tokenizer;

use crate::families::decoder_logit_qwen35::renderer::python_json;
use crate::families::support::FamilyError;

use super::{AssistantTail, MAX_OPTIONS_PER_PASS, MAX_SEQUENCE_TOKENS, MIN_CANDIDATES};

pub(crate) const SYSTEM_INSTRUCTION: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

/// The template head shared by every pinned Qwen3 checkpoint (system + user
/// turns).
const CHAT_TEMPLATE_HEAD: &str =
    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n";

/// Default Noul criterion text used by the reference protocol when the
/// caller supplies no explicit criteria.
pub(crate) const NOUL_DEFAULT_DESCRIPTIONS: [&str; 2] =
    ["The proposition is false.", "The proposition is true."];

/// One rendered forward pass: prompt ids plus the letter id per option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPass {
    prompt_ids: Vec<u32>,
    letter_ids: Vec<u32>,
}

impl RenderedPass {
    /// Prompt token ids ending at the assistant answer slot.
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    /// One letter-token id per option in this pass, in option order.
    pub fn letter_ids(&self) -> &[u32] {
        &self.letter_ids
    }
}

/// Offline renderer for one pinned control profile.
pub struct Qwen3ControlRenderer {
    tokenizer: Tokenizer,
    letter_token_ids: Vec<u32>,
    assistant_tail: AssistantTail,
}

impl Qwen3ControlRenderer {
    /// Load a digest-verified tokenizer and resolve the letter-token contract.
    ///
    /// Every option letter must tokenize to exactly one token; a tokenizer
    /// that splits a letter fails the load instead of corrupting the readout.
    pub fn load(
        tokenizer_path: &std::path::Path,
        assistant_tail: AssistantTail,
    ) -> Result<Self, FamilyError> {
        let bytes = std::fs::read(tokenizer_path).map_err(|source| FamilyError::Io {
            path: tokenizer_path.to_path_buf(),
            source,
        })?;
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let letter_token_ids = super::LETTERS
            .iter()
            .map(|letter| {
                let text = (*letter as char).to_string();
                let encoding = tokenizer
                    .encode(text.as_str(), false)
                    .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
                let ids = encoding.get_ids();
                if ids.len() != 1 {
                    return Err(FamilyError::ContractMismatch {
                        field: "letter_token_contract",
                        expected: format!("letter {text} tokenizes to exactly one token"),
                        actual: format!("letter {text} tokenizes to {} tokens", ids.len()),
                    });
                }
                Ok(ids[0])
            })
            .collect::<Result<Vec<u32>, FamilyError>>()?;
        if letter_token_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != letter_token_ids.len()
        {
            return Err(FamilyError::ContractMismatch {
                field: "letter_token_contract",
                expected: "distinct letter tokens".to_owned(),
                actual: "duplicate letter tokens".to_owned(),
            });
        }
        Ok(Self {
            tokenizer,
            letter_token_ids,
            assistant_tail,
        })
    }

    /// Render one pass into a prompt and its per-option letter ids.
    ///
    /// `state` is the wire state as a JSON value (`evidence`), `criterion`
    /// the question instruction text, and `options` the `(label,
    /// description)` pairs scored by this pass in order.
    pub fn render_pass(
        &self,
        state: &serde_json::Value,
        criterion: &str,
        options: &[(String, String)],
    ) -> Result<RenderedPass, FamilyError> {
        if criterion.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "question instruction must contain non-whitespace text".to_owned(),
            ));
        }
        if options.len() < MIN_CANDIDATES {
            return Err(FamilyError::InvalidInput(format!(
                "a pass needs at least {MIN_CANDIDATES} options, found {}",
                options.len()
            )));
        }
        if options.len() > MAX_OPTIONS_PER_PASS {
            return Err(FamilyError::InvalidInput(format!(
                "a pass scores at most {MAX_OPTIONS_PER_PASS} options, found {}",
                options.len()
            )));
        }
        if options
            .iter()
            .any(|(_, description)| description.trim().is_empty())
        {
            return Err(FamilyError::InvalidInput(
                "option descriptions must contain non-whitespace text".to_owned(),
            ));
        }
        // The jevk5 letter-pass payload: every option maps to
        // `f"{label}: {description}"` and each option object serializes
        // `letter` before `description`; the bytes are built by hand so the
        // sorted wire map cannot reorder them.
        let payload_options: String = options
            .iter()
            .enumerate()
            .map(|(index, (label, description))| {
                let rendered_description = python_json(&serde_json::Value::String(format!(
                    "{label}: {description}"
                )));
                format!(
                    "{{\"letter\": \"{}\", \"description\": {}}}",
                    super::LETTERS[index] as char,
                    rendered_description
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let payload = format!(
            "{{\"evidence\": {}, \"criterion\": {}, \"options\": [{}]}}",
            python_json(state),
            python_json(&serde_json::Value::String(criterion.to_owned())),
            payload_options,
        );
        let prompt = CHAT_TEMPLATE_HEAD
            .replace("{system}", SYSTEM_INSTRUCTION)
            .replace("{user}", &payload)
            + self.assistant_tail.as_str();
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let prompt_ids = encoding.get_ids().to_vec();
        if prompt_ids.len() > MAX_SEQUENCE_TOKENS {
            return Err(FamilyError::InvalidInput(format!(
                "rendered prompt length {} exceeds frozen maximum {}; truncation is forbidden",
                prompt_ids.len(),
                MAX_SEQUENCE_TOKENS
            )));
        }
        let letter_ids = options
            .iter()
            .enumerate()
            .map(|(index, _)| self.letter_token_ids[index])
            .collect();
        Ok(RenderedPass {
            prompt_ids,
            letter_ids,
        })
    }

    /// Token count of one rendered pass for usage accounting.
    pub fn pass_token_count(&self, pass: &RenderedPass) -> u32 {
        u32::try_from(pass.prompt_ids().len()).unwrap_or(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn assistant_tails_end_the_prompt() {
        // The tail is appended after the head's user turn, matching the
        // reference templates' generation prompts.
        assert!(AssistantTail::EmptyThinkBlock
            .as_str()
            .ends_with("</think>\n\n"));
        assert!(AssistantTail::BareGenerationPrompt.as_str().ends_with('\n'));
        assert!(CHAT_TEMPLATE_HEAD.contains("{system}"));
        assert!(CHAT_TEMPLATE_HEAD.contains("{user}"));
    }

    #[test]
    fn payload_uses_the_letter_pass_shape() {
        // Reuses the shared python_json: this smoke check only pins that the
        // renderer module stays wired to it.
        assert_eq!(python_json(&json!("x")), "\"x\"");
    }
}
