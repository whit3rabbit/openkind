//! Prompt renderer and letter-token contract for the pinned JevK5 profile.
//!
//! The bytes replicate the reference `jevk5` runtime exactly: the pinned
//! system instruction, the decision payload as a Python-`json.dumps`-shaped
//! object (`ensure_ascii=False`, `", "`/`": "` separators), and Qwen3.5's
//! chat template rendered with thinking off. Because the wire `criteria` map
//! is a hash map, options are rendered in the profile's deterministic sorted
//! label order — a declared renderer difference from the reference runtime,
//! which preserves the caller's insertion order.

use tokenizers::Tokenizer;

use crate::families::python_json::python_json;
use crate::families::support::FamilyError;

use super::MAX_OPTIONS_PER_PASS;

pub(crate) const SYSTEM_INSTRUCTION: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

/// Qwen3.5 chat template with thinking off, rendered once and pinned so the
/// native path is token-identical to the reference `transformers` rendering.
const CHAT_TEMPLATE: &str = "<|im_start|>system\n{system}<|im_end|>\n\
<|im_start|>user\n{user}<|im_end|>\n\
<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Default Noul criterion text used by the reference runtime when the caller
/// supplies no explicit criteria.
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

/// Offline renderer for the pinned letter-pass profiles of this family.
pub struct Jevk5Renderer {
    tokenizer: Tokenizer,
    letter_token_ids: Vec<u32>,
    /// Frozen maximum rendered prompt length per pass of the loaded profile.
    max_sequence_tokens: usize,
}

impl Jevk5Renderer {
    /// Load a digest-verified tokenizer and resolve the letter-token contract.
    ///
    /// Every option letter must tokenize to exactly one token; a tokenizer
    /// that splits a letter fails the load instead of corrupting the readout.
    pub fn load(
        tokenizer_path: &std::path::Path,
        max_sequence_tokens: usize,
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
            max_sequence_tokens,
        })
    }

    /// The serialized `evidence` field of the payload. It depends on the
    /// request state alone, so a request renders it once and reuses it for
    /// every question and knockout pass through
    /// [`Self::render_pass_with_evidence`].
    pub fn render_evidence(state: &serde_json::Value) -> String {
        python_json(state)
    }

    /// Render one knockout pass into a prompt and its per-option letter ids.
    ///
    /// `evidence` is the wire state serialized by [`Self::render_evidence`],
    /// `criterion` the question instruction text, and `options` the `(label,
    /// description)` pairs scored by this pass in order.
    pub fn render_pass_with_evidence(
        &self,
        evidence: &str,
        criterion: &str,
        options: &[(String, String)],
    ) -> Result<RenderedPass, FamilyError> {
        if criterion.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "question instruction must contain non-whitespace text".to_owned(),
            ));
        }
        if options.len() < super::MIN_CANDIDATES {
            return Err(FamilyError::InvalidInput(format!(
                "a pass needs at least {} options, found {}",
                super::MIN_CANDIDATES,
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
        // The reference maps every option to `f"{label}: {description}"`
        // (jevk5 `decision_options`) and serializes each option object with
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
            evidence,
            python_json(&serde_json::Value::String(criterion.to_owned())),
            payload_options,
        );
        let prompt = CHAT_TEMPLATE
            .replace("{system}", SYSTEM_INSTRUCTION)
            .replace("{user}", &payload);
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let prompt_ids = encoding.get_ids().to_vec();
        if prompt_ids.len() > self.max_sequence_tokens {
            return Err(FamilyError::InvalidInput(format!(
                "rendered prompt length {} exceeds frozen maximum {}; truncation is forbidden",
                prompt_ids.len(),
                self.max_sequence_tokens
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
