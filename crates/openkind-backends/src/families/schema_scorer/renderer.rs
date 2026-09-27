//! Query–passage pair renderer for the schema-scorer family.

use tokenizers::Tokenizer;

use crate::families::support::FamilyError;

/// Offline pair renderer for the pinned schema-scorer profile.
pub struct SchemaScorerRenderer {
    tokenizer: Tokenizer,
}

impl SchemaScorerRenderer {
    /// Load a digest-verified `tokenizer.json`.
    pub fn load(tokenizer_path: &std::path::Path) -> Result<Self, FamilyError> {
        let bytes = std::fs::read(tokenizer_path).map_err(|source| FamilyError::Io {
            path: tokenizer_path.to_path_buf(),
            source,
        })?;
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(Self { tokenizer })
    }

    /// Encode one `(query, passage)` pair with the checkpoint's own
    /// post-processor (`[CLS] query [SEP] passage [SEP]`).
    ///
    /// No truncation is performed: an over-long pair fails closed.
    pub fn encode_pair(&self, query: &str, passage: &str) -> Result<Vec<u32>, FamilyError> {
        if query.trim().is_empty() || passage.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "query and passage must contain non-whitespace text".to_owned(),
            ));
        }
        let encoding = self
            .tokenizer
            .encode((query, passage), true)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        if ids.len() > super::MAX_SEQUENCE_TOKENS {
            return Err(FamilyError::InvalidInput(format!(
                "encoded pair length {} exceeds frozen maximum {}; truncation is forbidden",
                ids.len(),
                super::MAX_SEQUENCE_TOKENS
            )));
        }
        Ok(ids)
    }
}
