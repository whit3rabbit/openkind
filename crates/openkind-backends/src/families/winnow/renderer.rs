//! Routing-prompt renderer for the winnow family.
//!
//! The prompt is the exact frozen format the LoRA adapter was trained on
//! (chat-wrapped prompt/completion pairs): the completion tokens are `" A"`
//! and `" B"`, so the letter contract resolves those token ids rather than
//! the bare letters.

use tokenizers::Tokenizer;

use crate::families::support::FamilyError;

const ROUTE_INSTRUCTION: &str = "Route this request to the matching sibling engine.";
const ROUTE_LABELS: &str = "Options:\nA. english\nB. multilingual\n\nAnswer with the single letter of the best option only.";
const ROUTE_COMPLETIONS: &[&str] = &[" A", " B"];

/// Offline routing-prompt renderer for the pinned winnow profile.
pub struct WinnowRenderer {
    tokenizer: Tokenizer,
    letter_token_ids: Vec<u32>,
}

impl WinnowRenderer {
    /// Load a digest-verified tokenizer and resolve the completion-token
    /// contract (`" A"`, `" B"` must each be single tokens).
    pub fn load(tokenizer_path: &std::path::Path) -> Result<Self, FamilyError> {
        let bytes = std::fs::read(tokenizer_path).map_err(|source| FamilyError::Io {
            path: tokenizer_path.to_path_buf(),
            source,
        })?;
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let letter_token_ids = ROUTE_COMPLETIONS
            .iter()
            .map(|text| {
                let encoding = tokenizer
                    .encode(*text, false)
                    .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
                let ids = encoding.get_ids();
                if ids.len() != 1 {
                    return Err(FamilyError::ContractMismatch {
                        field: "routing_token_contract",
                        expected: format!("completion {text:?} tokenizes to exactly one token"),
                        actual: format!("completion {text:?} tokenizes to {} tokens", ids.len()),
                    });
                }
                Ok(ids[0])
            })
            .collect::<Result<Vec<u32>, FamilyError>>()?;
        Ok(Self {
            tokenizer,
            letter_token_ids,
        })
    }

    /// Render one state document into the frozen routing prompt.
    ///
    /// The prompt ends at the assistant answer slot, exactly as the adapter
    /// was trained. No truncation is performed.
    pub fn render(&self, state: &str) -> Result<Vec<u32>, FamilyError> {
        if state.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "state must contain non-whitespace text".to_owned(),
            ));
        }
        let prompt = format!(
            "<|im_start|>user\n{ROUTE_INSTRUCTION}\n\nText: {state}\n\n{ROUTE_LABELS}<|im_end|>\n<|im_start|>assistant\n"
        );
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        if ids.len() > super::MAX_SEQUENCE_TOKENS {
            return Err(FamilyError::InvalidInput(format!(
                "rendered prompt length {} exceeds frozen maximum {}; truncation is forbidden",
                ids.len(),
                super::MAX_SEQUENCE_TOKENS
            )));
        }
        Ok(ids)
    }

    /// One completion-token id per routing label, in label order.
    pub fn letter_ids(&self) -> &[u32] {
        &self.letter_token_ids
    }
}
