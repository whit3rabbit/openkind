//! Sequence renderer for the von family: a faithful port of the reference's
//! `pack_sequence`, `_neutralise`, and `_fit_state`.
//!
//! Template: `[CLS] <question> <state> [SEP] ([MASK] + option)… [SEP]` —
//! the question merges into the prefix (`"{question} {state}"`, or the bare
//! state without a question). `[MASK]`/`[SEP]` literals in user text are
//! neutralised with a zero-width joiner so callers cannot forge structure.
//! Long states middle-truncate to 60% head / 40% tail joined by `" ... "`,
//! leaving room for the question and option reserve.

use tokenizers::Tokenizer;

use crate::families::support::FamilyError;

use super::VonSpecials;

/// One rendered question row: token ids plus marker positions.
#[derive(Debug, Clone)]
pub(crate) struct RenderedQuestion {
    /// The full token sequence including the `[CLS]`/`[SEP]` frame.
    pub ids: Vec<u32>,
    /// Position of each option's marker token, ascending.
    pub markers: Vec<usize>,
}

/// Offline renderer for the pinned von tokenizer.
pub struct VonRenderer {
    tokenizer: Tokenizer,
    specials: VonSpecials,
    mask_text: &'static str,
    sep_text: &'static str,
}

impl VonRenderer {
    /// Load the digest-verified fast tokenizer and fail closed unless the
    /// pinned special-token strings resolve to the pinned ids.
    ///
    /// The shipped `tokenizer.json` embeds training-era truncation (512) and
    /// padding settings; the reference runs through `transformers`, which
    /// leaves both off, so the loader strips them to match: truncation would
    /// silently drop the trailing option markers that sit at the end of the
    /// packed sequence.
    pub fn load(
        tokenizer_path: &std::path::Path,
        specials: &VonSpecials,
    ) -> Result<Self, FamilyError> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?
            .with_padding(None);
        let check = |token: &str, expected: u32| -> Result<(), FamilyError> {
            let actual =
                tokenizer
                    .token_to_id(token)
                    .ok_or_else(|| FamilyError::ContractMismatch {
                        field: "tokenizer.special",
                        expected: format!("{token} = {expected}"),
                        actual: "missing".to_owned(),
                    })?;
            if actual != expected {
                return Err(FamilyError::contract(
                    "tokenizer.special",
                    format!("{token} = {expected}"),
                    format!("{token} = {actual}"),
                ));
            }
            Ok(())
        };
        for (token, id) in [
            ("[CLS]", specials.cls),
            ("[SEP]", specials.sep),
            ("[PAD]", specials.pad),
            ("[MASK]", specials.mask),
        ] {
            check(token, id)?;
        }
        Ok(Self {
            tokenizer,
            specials: *specials,
            mask_text: "[MASK]",
            sep_text: "[SEP]",
        })
    }

    /// Assemble a renderer over an already-loaded tokenizer (tests).
    #[cfg(test)]
    pub(crate) fn from_tokenizer(tokenizer: Tokenizer, specials: VonSpecials) -> Self {
        Self {
            tokenizer,
            specials,
            mask_text: "[MASK]",
            sep_text: "[SEP]",
        }
    }

    /// Break special-token literals in user text with a zero-width joiner:
    /// the visible text is unchanged but the tokenizer no longer maps them
    /// to the special ids. Port of the reference `_neutralise`.
    pub(crate) fn neutralise(&self, text: &str) -> String {
        let mut out = text.to_owned();
        for special in [self.mask_text, self.sep_text] {
            let broken = format!("{}\u{200d}{}", &special[..1], &special[1..]);
            out = out.replace(special, &broken);
        }
        out
    }

    /// Pack state, question, and candidate options into the option-marker
    /// string. Port of the reference `pack_sequence` (digit splitting is
    /// pinned off for this checkpoint).
    pub(crate) fn pack(&self, state: &str, question: &str, options: &[String]) -> String {
        let mask = self.mask_text;
        let sep = self.sep_text;
        let state = self.neutralise(state);
        let question = self.neutralise(question);
        let options: Vec<String> = options
            .iter()
            .map(|option| self.neutralise(option))
            .collect();
        let prefix = if question.is_empty() {
            state.trim().to_owned()
        } else {
            format!("{question} {state}").trim().to_owned()
        };
        let opts_packed = options
            .iter()
            .map(|option| format!("{mask} {}", option.trim()))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{prefix} {sep} {opts_packed}")
    }

    /// Encode the packed string with the `[CLS] … [SEP]` frame the
    /// tokenizer's post-processor adds, and locate the option markers.
    pub(crate) fn encode_packed(&self, packed: &str) -> Result<RenderedQuestion, FamilyError> {
        let encoding = self
            .tokenizer
            .encode(packed, true)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let markers: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == self.specials.mask)
            .map(|(position, _)| position)
            .collect();
        Ok(RenderedQuestion { ids, markers })
    }

    /// Encode text without special tokens, counting state tokens exactly as
    /// the reference's truncation and temperature features do.
    pub(crate) fn encode_plain(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode raw ids back to text (the truncation joiner rebuild).
    pub(crate) fn decode(&self, ids: &[u32]) -> Result<String, FamilyError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))
    }

    /// Middle-truncate the state so question + option markers always fit:
    /// keeps 60% head, 40% tail, joined by `" ... "`. Port of the reference
    /// `_fit_state` (truncate policy; the refuse policy is not compiled in).
    pub(crate) fn fit_state(
        &self,
        profile: &super::VonProfile,
        state_text: &str,
        question: &str,
        options: &[String],
    ) -> Result<String, FamilyError> {
        let max_text_bytes = profile.max_sequence_tokens.saturating_mul(64);
        if question.len() > max_text_bytes {
            return Err(FamilyError::InvalidInput(format!(
                "question instructions length {} bytes exceeds maximum allowed text limit {} bytes",
                question.len(),
                max_text_bytes
            )));
        }
        let total_option_bytes: usize = options.iter().map(|opt| opt.len()).sum();
        if total_option_bytes > max_text_bytes {
            return Err(FamilyError::InvalidInput(format!(
                "question options total length {} bytes exceeds maximum allowed text limit {} bytes",
                total_option_bytes,
                max_text_bytes
            )));
        }

        let reserve_reserve = self.pack("", question, options);
        let reserve = self.encode_plain(&reserve_reserve)?.len() + 8;
        let window = profile.max_sequence_tokens;
        if reserve > window {
            return Err(FamilyError::InvalidInput(format!(
                "question and options reserve {reserve} tokens, exceeding maximum sequence window {window}"
            )));
        }
        let remaining = window.saturating_sub(reserve);
        if remaining == 0 {
            return Err(FamilyError::InvalidInput(format!(
                "question and options reserve {reserve} tokens, leaving no budget for state in maximum sequence window {window}"
            )));
        }
        let ids = self.encode_plain(state_text)?;
        if ids.len() <= remaining {
            return Ok(state_text.to_owned());
        }
        if remaining < 16 {
            return Err(FamilyError::InvalidInput(format!(
                "remaining state budget {remaining} tokens is insufficient for state truncation"
            )));
        }
        let limit = profile.max_state_tokens.min(remaining);
        let head = (limit as f64 * 0.6) as usize;
        let tail = limit - head - 2;
        let fitted = format!(
            "{} ... {}",
            self.decode(&ids[..head])?,
            self.decode(&ids[ids.len() - tail..])?
        );
        Ok(fitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::processors::template::TemplateProcessing;

    fn test_tokenizer(specials: &VonSpecials) -> Tokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0_u32);
        vocab.insert("[CLS]".to_string(), specials.cls);
        vocab.insert("[SEP]".to_string(), specials.sep);
        vocab.insert("[PAD]".to_string(), specials.pad);
        vocab.insert("[MASK]".to_string(), specials.mask);
        vocab.insert("word".to_string(), 1);
        vocab.insert("state".to_string(), 2);
        vocab.insert("opt".to_string(), 3);
        vocab.insert("...".to_string(), 4);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let template = TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .expect("template single")
            .special_tokens(vec![
                ("[CLS]".to_string(), specials.cls),
                ("[SEP]".to_string(), specials.sep),
            ])
            .build()
            .expect("template build");
        tokenizer.with_post_processor(Some(template));
        tokenizer
    }

    fn test_profile(max_sequence_tokens: usize) -> super::super::VonProfile {
        let mut profile = super::super::VON;
        profile.max_sequence_tokens = max_sequence_tokens;
        profile.max_state_tokens = max_sequence_tokens;
        profile
    }

    #[test]
    fn fit_state_rejects_oversized_fixed_reserve() {
        let specials = super::super::VON.specials;
        let renderer = VonRenderer::from_tokenizer(test_tokenizer(&specials), specials);
        let profile = test_profile(20);

        // Build question text that exceeds the 20-token window by itself
        let large_question = ["word"; 30].join(" ");
        let options = vec!["opt".to_string(), "opt".to_string()];
        let error = renderer
            .fit_state(&profile, "state", &large_question, &options)
            .expect_err("oversized reserve must be rejected");

        assert!(
            matches!(&error, FamilyError::InvalidInput(msg) if msg.contains("exceeding maximum sequence window 20")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn fit_state_rejects_huge_inputs_at_byte_boundary() {
        let specials = super::super::VON.specials;
        let renderer = VonRenderer::from_tokenizer(test_tokenizer(&specials), specials);
        let profile = test_profile(20);

        // 20 tokens * 64 = 1280 bytes limit
        let huge_question = "a".repeat(1281);
        let options = vec!["opt".to_string()];
        let error = renderer
            .fit_state(&profile, "state", &huge_question, &options)
            .expect_err("huge byte input must be rejected early");

        assert!(
            matches!(&error, FamilyError::InvalidInput(msg) if msg.contains("exceeds maximum allowed text limit")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn fit_state_rejects_when_reserve_leaves_insufficient_state_budget() {
        let specials = super::super::VON.specials;
        let renderer = VonRenderer::from_tokenizer(test_tokenizer(&specials), specials);
        let profile = test_profile(25);

        // Question and options consume around 15 tokens; reserve has +8 framing,
        // leaving < 16 remaining tokens for state truncation.
        let question = ["word"; 5].join(" ");
        let options = vec!["opt".to_string(), "opt".to_string()];
        let long_state = ["state"; 50].join(" ");

        let error = renderer
            .fit_state(&profile, &long_state, &question, &options)
            .expect_err("insufficient state budget must be rejected");

        assert!(
            matches!(&error, FamilyError::InvalidInput(msg) if msg.contains("insufficient for state truncation")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn exactly_at_window_render_and_one_token_over_limit() {
        let specials = super::super::VON.specials;
        let renderer = VonRenderer::from_tokenizer(test_tokenizer(&specials), specials);

        let question = "word";
        let options = vec!["opt".to_string(), "opt".to_string()];
        let state = "state state";

        let reserve = renderer
            .encode_plain(&renderer.pack("", question, &options))
            .unwrap()
            .len()
            + 8;
        let exact_tokens_window = reserve + 2;
        let exact_profile = test_profile(exact_tokens_window);
        let fitted = renderer
            .fit_state(&exact_profile, state, question, &options)
            .expect("exact state fits in window");
        assert_eq!(fitted, state);

        let packed = renderer.pack(&fitted, question, &options);
        let rendered = renderer.encode_packed(&packed).expect("encode");
        let rendered_tokens = rendered.ids.len();

        // 1. Exactly at window: profile max_sequence_tokens matches rendered length
        let window_profile = test_profile(rendered_tokens);
        assert_eq!(rendered.ids.len(), window_profile.max_sequence_tokens);
        assert!(rendered.ids.len() <= window_profile.max_sequence_tokens);

        // 2. One token over the limit: profile limit is rendered_tokens - 1
        let tight_profile = test_profile(rendered_tokens - 1);
        assert_eq!(rendered.ids.len(), tight_profile.max_sequence_tokens + 1);

        let check_result = if rendered.ids.len() > tight_profile.max_sequence_tokens {
            Err(FamilyError::InvalidInput(format!(
                "rendered {} tokens, exceeding maximum sequence window {}",
                rendered.ids.len(),
                tight_profile.max_sequence_tokens
            )))
        } else {
            Ok(())
        };
        assert!(
            matches!(&check_result, Err(FamilyError::InvalidInput(msg)) if msg.contains("exceeding maximum sequence window")),
            "unexpected result: {check_result:?}"
        );
    }
}
