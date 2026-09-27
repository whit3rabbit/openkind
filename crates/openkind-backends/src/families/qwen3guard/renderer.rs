//! Guard-prompt renderer for the qwen3guard family.
//!
//! The pinned checkpoint moderates the user-role text through its chat
//! template. The reference feeds the tokenized turn up to and including the
//! closing `<|im_end|>` — the prompt is `<|im_start|>user\n{state}<|im_end|>`
//! and the query-side Stream head scores that final `<|im_end|>` position.

use tokenizers::Tokenizer;

use crate::families::support::FamilyError;

/// Offline renderer for the pinned guard profile.
pub struct Qwen3GuardRenderer {
    tokenizer: Tokenizer,
}

impl Qwen3GuardRenderer {
    /// Load a digest-verified tokenizer.
    pub fn load(tokenizer_path: &std::path::Path) -> Result<Self, FamilyError> {
        let bytes = std::fs::read(tokenizer_path).map_err(|source| FamilyError::Io {
            path: tokenizer_path.to_path_buf(),
            source,
        })?;
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(Self { tokenizer })
    }

    /// Render one state document into the user-moderation prompt.
    ///
    /// No truncation is performed: an over-long prompt fails closed.
    pub fn render(&self, state: &str) -> Result<Vec<u32>, FamilyError> {
        if state.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "state must contain non-whitespace text".to_owned(),
            ));
        }
        // The reference truncates the turn at the closing `<|im_end|>` and
        // scores that position; a trailing newline would shift the readout
        // slot by one token and destroy the trained head alignment.
        let prompt = format!("<|im_start|>user\n{state}<|im_end|>");
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
}
