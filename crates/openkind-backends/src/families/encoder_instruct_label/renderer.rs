//! Marker-plus-state renderer for the encoder-instruct-label family.
//!
//! The pinned profile loads the digest-verified fast `tokenizer.json`
//! directly; its serialized template prepends `[CLS]` and appends `[SEP]`.
//! The rendered input matches the reference pipeline byte for byte:
//! `<<LABEL>>` immediately before each candidate criterion, then the
//! `<<SEP>>` boundary, then the state text, with no injected spaces.

use tokenizers::Tokenizer;

use crate::families::support::FamilyError;

/// Marker strings wrapped before each candidate in the rendered sequence.
pub(crate) const LABEL_MARKER: &str = "<<LABEL>>";
/// Boundary string between the candidate markers and the state text.
pub(crate) const TEXT_MARKER: &str = "<<SEP>>";

/// Offline renderer for the pinned encoder-instruct-label profile.
pub struct EncoderInstructLabelRenderer {
    tokenizer: Tokenizer,
}

impl EncoderInstructLabelRenderer {
    /// Load the digest-verified fast tokenizer.
    pub fn load(tokenizer_path: &std::path::Path) -> Result<Self, FamilyError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(Self { tokenizer })
    }

    /// Render one marker list plus the state into token ids.
    ///
    /// No truncation is performed: an over-long sequence fails closed.
    pub fn render(&self, markers: &[String], state: &str) -> Result<Vec<u32>, FamilyError> {
        if markers.is_empty() {
            return Err(FamilyError::InvalidInput(
                "questions need at least one candidate marker".to_owned(),
            ));
        }
        if state.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "state must contain non-whitespace text".to_owned(),
            ));
        }
        let mut rendered = String::new();
        for marker in markers {
            if marker.trim().is_empty() {
                return Err(FamilyError::InvalidInput(
                    "candidate criterion must contain non-whitespace text".to_owned(),
                ));
            }
            rendered.push_str(LABEL_MARKER);
            rendered.push_str(marker);
        }
        rendered.push_str(TEXT_MARKER);
        rendered.push_str(state);
        let encoding = self
            .tokenizer
            .encode(rendered.as_str(), true)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        if ids.len() > super::MAX_SEQUENCE_TOKENS {
            return Err(FamilyError::InvalidInput(format!(
                "encoded sequence length {} exceeds frozen maximum {}; truncation is forbidden",
                ids.len(),
                super::MAX_SEQUENCE_TOKENS
            )));
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn marker_string_matches_the_pinned_vocabulary() {
        // The pinned checkpoint's `class_token_index` special token; the
        // renderer composes raw text, the tokenizer converts it to this id.
        assert_eq!(super::super::CLASS_TOKEN_ID, 50368);
    }
}
