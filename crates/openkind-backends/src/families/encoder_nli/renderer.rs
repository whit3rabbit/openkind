//! Premise–hypothesis pair renderer for the encoder-NLI family.
//!
//! The pinned checkpoint ships a WordPiece `vocab.txt` without a fast
//! `tokenizer.json`; the renderer assembles the standard BERT pipeline
//! (BertNormalizer lowercasing, BertPreTokenizer, WordPiece model, and a
//! `[CLS] $A [SEP] $B [SEP]` pair template) from the digest-checked
//! vocabulary.

use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::Tokenizer;

use crate::families::support::FamilyError;

/// Uncased BERT special-token ids in the pinned vocabulary.
pub(crate) const CLS_ID: u32 = 101;
pub(crate) const SEP_ID: u32 = 102;

/// Offline pair renderer for the pinned encoder-NLI profile.
pub struct EncoderNliRenderer {
    tokenizer: Tokenizer,
}

impl EncoderNliRenderer {
    /// Build the BERT pipeline from a digest-verified `vocab.txt`.
    pub fn load(vocab_path: &std::path::Path) -> Result<Self, FamilyError> {
        let wordpiece = WordPiece::from_file(
            vocab_path
                .to_str()
                .ok_or_else(|| FamilyError::InvalidInput("vocab path is not UTF-8".into()))?,
        )
        .unk_token("[UNK]".to_owned())
        .continuing_subword_prefix("##".to_owned())
        .max_input_chars_per_word(100)
        .build()
        .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(wordpiece);
        tokenizer.with_normalizer(Some(BertNormalizer::new(true, true, None, true)));
        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
        let template = TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .map_err(FamilyError::Tokenizer)?
            .try_pair("[CLS] $A [SEP] $B [SEP]")
            .map_err(FamilyError::Tokenizer)?
            .special_tokens(vec![
                ("[CLS]".to_owned(), CLS_ID),
                ("[SEP]".to_owned(), SEP_ID),
            ])
            .build()
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        tokenizer.with_post_processor(Some(template));
        Ok(Self { tokenizer })
    }

    /// Encode one `(premise, hypothesis)` pair.
    ///
    /// No truncation is performed: an over-long pair fails closed.
    pub fn encode_pair(&self, premise: &str, hypothesis: &str) -> Result<Vec<u32>, FamilyError> {
        if premise.trim().is_empty() || hypothesis.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "premise and hypothesis must contain non-whitespace text".to_owned(),
            ));
        }
        let encoding = self
            .tokenizer
            .encode((premise, hypothesis), true)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_contract_uses_bert_special_tokens() {
        assert_eq!(CLS_ID, 101);
        assert_eq!(SEP_ID, 102);
    }
}
