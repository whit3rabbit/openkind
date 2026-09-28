//! Sequence renderer for the laya family: a faithful port of the
//! reference's `build_sequence`.
//!
//! Template: `[CLS] <type> question: <instructions> [SEP] ([MASK] +
//! option)… [SEP] <state> [SEP]`. Option spans cap at 48 text tokens; when
//! the spans overflow `head_max_len` each span (marker included) is cut to
//! `max(4, (head_max_len − 16) / options)` and the instruction keeps
//! `max(8, budget)` tokens. The state keeps its first `room` tokens
//! (newest `room` tokens for array states), and literal mask-token text in
//! user input is replaced with a space. Heads that would drop a marker fail
//! closed instead of truncating.

use tokenizers::Tokenizer;

use crate::families::support::FamilyError;

use super::LayaSpecials;

/// Reference cap on one option span's text tokens (the marker is extra).
const OPTION_SPAN_TOKENS: usize = 48;
/// Minimum remaining head budget before per-option cuts kick in.
const OPTION_BUDGET_FLOOR: usize = 16;

/// One rendered question row: token ids plus marker positions.
#[derive(Debug, Clone)]
pub(crate) struct RenderedQuestion {
    /// The full token sequence, at most `max_len` long.
    pub ids: Vec<u32>,
    /// Position of each option's marker token, ascending.
    pub markers: Vec<usize>,
}

/// Offline renderer for one pinned laya tokenizer.
pub struct LayaRenderer {
    tokenizer: Tokenizer,
}

impl LayaRenderer {
    /// Load the digest-verified fast tokenizer and fail closed unless the
    /// pinned special-token strings resolve to the pinned ids.
    pub fn load(
        tokenizer_path: &std::path::Path,
        specials: &LayaSpecials,
    ) -> Result<Self, FamilyError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
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
        let strings = [
            (specials.cls_token(), specials.cls),
            (specials.sep_token(), specials.sep),
            (specials.pad_token(), specials.pad),
            (specials.mask_token, specials.mask),
        ];
        for (token, id) in strings {
            check(token, id)?;
        }
        Ok(Self { tokenizer })
    }

    /// Encode text without special tokens, as every piece of a laya
    /// sequence is assembled from raw ids.
    pub(crate) fn encode_plain(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
}

impl LayaSpecials {
    fn cls_token(&self) -> &'static str {
        match self.cls {
            50_281 => "[CLS]",
            _ => "<bos>",
        }
    }

    fn sep_token(&self) -> &'static str {
        match self.sep {
            50_282 => "[SEP]",
            _ => "<eos>",
        }
    }

    fn pad_token(&self) -> &'static str {
        match self.pad {
            50_283 => "[PAD]",
            _ => "<pad>",
        }
    }
}

/// Static inputs of one `build_sequence` call: the loaded tokenizer pair and
/// the profile's frozen budgets and truncation side.
pub(crate) struct SequencePlan<'a> {
    /// Renderer holding the pinned tokenizer.
    pub renderer: &'a LayaRenderer,
    /// Pinned special-token ids and mask surface string.
    pub specials: &'a LayaSpecials,
    /// Frozen maximum sequence length (`max_len`).
    pub max_len: usize,
    /// Frozen head budget (`head_max_len`).
    pub head_max_len: usize,
    /// Keep the newest state tokens instead of the first (array states).
    pub truncate_left: bool,
}

/// Build one question row, mirroring the reference exactly.
pub(crate) fn build_sequence(
    plan: &SequencePlan<'_>,
    type_name: &str,
    instructions: &str,
    options: &[String],
    state_ids: &[u32],
) -> Result<RenderedQuestion, FamilyError> {
    let SequencePlan {
        renderer,
        specials,
        max_len,
        head_max_len,
        truncate_left,
    } = *plan;
    if options.is_empty() {
        return Err(FamilyError::InvalidInput(
            "questions need at least one option".to_owned(),
        ));
    }
    let sanitize = |text: &str| text.replace(specials.mask_token, " ");
    let head_text = format!("{type_name} question: {}", sanitize(instructions));
    let head_ids = renderer.encode_plain(&head_text)?;
    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(options.len());
    for option in options {
        let text = format!(" {}", sanitize(option));
        let mut span = renderer.encode_plain(&text)?;
        span.truncate(OPTION_SPAN_TOKENS);
        let mut span_with_marker = Vec::with_capacity(span.len() + 1);
        span_with_marker.push(specials.mask);
        span_with_marker.extend(span);
        opt_ids.push(span_with_marker);
    }
    let mut opt_budget =
        head_max_len as isize - opt_ids.iter().map(Vec::len).sum::<usize>() as isize;
    if opt_budget < OPTION_BUDGET_FLOOR as isize {
        let per = (head_max_len - OPTION_BUDGET_FLOOR) / opt_ids.len().max(1);
        let per = per.max(4);
        for span in &mut opt_ids {
            span.truncate(per);
        }
        opt_budget = head_max_len as isize - opt_ids.iter().map(Vec::len).sum::<usize>() as isize;
    }
    let head_keep = (opt_budget.max(8) as usize).min(head_ids.len());
    let mut ids = Vec::with_capacity(max_len);
    ids.push(specials.cls);
    ids.extend_from_slice(&head_ids[..head_keep]);
    ids.push(specials.sep);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for span in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(span);
    }
    ids.push(specials.sep);
    let room = max_len.saturating_sub(ids.len() + 1);
    let state_slice: &[u32] = if truncate_left {
        &state_ids[state_ids.len().saturating_sub(room)..]
    } else {
        &state_ids[..state_ids.len().min(room)]
    };
    ids.extend_from_slice(state_slice);
    ids.push(specials.sep);
    ids.truncate(max_len);
    let markers: Vec<usize> = markers.into_iter().filter(|&m| m < max_len).collect();
    if markers.len() != options.len() {
        return Err(FamilyError::InvalidInput(format!(
            "question options exceed head_max_len={head_max_len}"
        )));
    }
    Ok(RenderedQuestion { ids, markers })
}
