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
    pub(crate) fn encode_packed(
        &self,
        packed: &str,
        max_sequence_tokens: usize,
    ) -> Result<RenderedQuestion, FamilyError> {
        let encoding = self
            .tokenizer
            .encode(packed, true)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        ensure_sequence_fits(ids.len(), max_sequence_tokens)?;
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
        let reserve_reserve = self.pack("", question, options);
        let reserve = self
            .encode_plain(&reserve_reserve)?
            .len()
            .checked_add(8)
            .ok_or_else(|| {
                FamilyError::InvalidInput("von question token reserve overflowed".to_owned())
            })?;
        let window = profile.max_sequence_tokens;
        let limit = state_token_limit(window, profile.max_state_tokens, reserve)?;
        let ids = self.encode_plain(state_text)?;
        if ids.len() <= limit {
            return Ok(state_text.to_owned());
        }
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

fn ensure_sequence_fits(actual: usize, maximum: usize) -> Result<(), FamilyError> {
    if actual > maximum {
        return Err(FamilyError::InvalidInput(format!(
            "von packed sequence has {actual} tokens, exceeding the {maximum}-token model window"
        )));
    }
    Ok(())
}

fn state_token_limit(
    window: usize,
    max_state_tokens: usize,
    reserve: usize,
) -> Result<usize, FamilyError> {
    let available = window
        .checked_sub(reserve)
        .filter(|available| *available >= 16)
        .ok_or_else(|| {
            FamilyError::InvalidInput(format!(
                "von question and options leave fewer than 16 state tokens in the {window}-token model window"
            ))
        })?;
    Ok(16.max(max_state_tokens.min(available)))
}

#[cfg(test)]
mod tests {
    use super::{ensure_sequence_fits, state_token_limit};

    #[test]
    fn rejects_a_sequence_over_the_model_window() {
        assert!(ensure_sequence_fits(8_193, 8_192).is_err());
        assert!(ensure_sequence_fits(8_192, 8_192).is_ok());
    }

    #[test]
    fn rejects_question_reserve_that_crowds_out_the_state() {
        assert!(state_token_limit(8_192, 4_096, 8_177).is_err());
        assert_eq!(state_token_limit(8_192, 4_096, 8_176).unwrap(), 16);
    }
}
