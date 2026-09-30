//! The pinned single-user Qwen chat template with an empty reasoning chain.

use std::path::Path;
use tokenizers::Tokenizer;

use super::MAX_TOKENS;
use crate::families::kev::renderer::escape_named_specials as escape_specials;
use crate::families::support::FamilyError;

/// One causal row and the suffix positions consumed by the pointer head.
#[derive(Debug, Clone)]
pub struct JeevesRow {
    /// Exact encoded prompt, empty chain, and repeated option suffix.
    pub ids: Vec<u32>,
    /// The second option block's option-end positions, in supplied order.
    pub option_positions: Vec<usize>,
    /// The final decision marker position.
    pub decide_position: usize,
}

/// Offline fast tokenizer plus the pinned renderer semantics.
pub struct JeevesRenderer {
    tokenizer: Tokenizer,
    option_end: u32,
    decide: u32,
    think_end: u32,
}

impl JeevesRenderer {
    /// Load a caller-verified tokenizer without fetching any assets.
    pub fn load(path: &Path) -> Result<Self, FamilyError> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let resolve = |token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| FamilyError::Tokenizer(format!("missing Jeeves marker {token}")))
        };
        let option_end = resolve("<|box_end|>")?;
        let decide = resolve("<|fim_suffix|>")?;
        let think_end = resolve("</think>")?;
        Ok(Self {
            tokenizer,
            option_end,
            decide,
            think_end,
        })
    }

    /// Mirror the pinned template's single-user, enable-thinking branch.
    /// The newline between `<think>` and `</think>` is the empty chain.
    pub fn text(state: &str, instruction: &str, options: &[String]) -> String {
        let (prompt, suffix) = Self::segments(state, instruction, options);
        format!("{prompt}\n{suffix}")
    }

    fn segments(state: &str, instruction: &str, options: &[String]) -> (String, String) {
        let block: String = options
            .iter()
            .map(|option| format!("<|box_start|>{}<|box_end|>\n", escape_specials(option)))
            .collect();
        let content = format!(
            "<|fim_prefix|>{}\n<|fim_middle|>{}\n{block}",
            escape_specials(state),
            escape_specials(instruction)
        );
        (
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                content.trim()
            ),
            format!("</think>\n\n{block}<|fim_suffix|>"),
        )
    }

    /// Encode the upstream segments separately, then select the suffix block.
    pub fn encode(
        &self,
        state: &str,
        instruction: &str,
        options: &[String],
    ) -> Result<JeevesRow, FamilyError> {
        if options.is_empty() || options.len() > 255 {
            return Err(FamilyError::InvalidInput(
                "Jeeves needs 1..255 offered options".into(),
            ));
        }
        let (prompt, suffix) = Self::segments(state, instruction, options);
        let mut ids = Vec::new();
        // Concatenating text before BPE merges the prompt newline with the
        // empty-chain newline and changes the reference's two token IDs.
        for segment in [&prompt, "\n", &suffix] {
            let encoded = self
                .tokenizer
                .encode(segment, false)
                .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
            ids.extend_from_slice(encoded.get_ids());
        }
        if ids.is_empty() || ids.len() > MAX_TOKENS || ids.last() != Some(&self.decide) {
            return Err(FamilyError::InvalidInput(format!(
                "invalid Jeeves row length {}; maximum {MAX_TOKENS}, no truncation",
                ids.len()
            )));
        }
        let start = ids
            .iter()
            .rposition(|id| *id == self.think_end)
            .ok_or_else(|| FamilyError::Tokenizer("missing suffix think-end marker".into()))?;
        let option_positions: Vec<_> = ids
            .iter()
            .enumerate()
            .skip(start)
            .filter_map(|(index, id)| (*id == self.option_end).then_some(index))
            .collect();
        if option_positions.len() != options.len() {
            return Err(FamilyError::Tokenizer(
                "suffix option count mismatch".into(),
            ));
        }
        Ok(JeevesRow {
            decide_position: ids.len() - 1,
            ids,
            option_positions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_chain_and_repeated_options_match_upstream_layout() {
        let text = JeevesRenderer::text("paid", "Was it paid?", &["no".into(), "yes".into()]);
        assert_eq!(text, "<|im_start|>user\n<|fim_prefix|>paid\n<|fim_middle|>Was it paid?\n<|box_start|>no<|box_end|>\n<|box_start|>yes<|box_end|><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|box_start|>no<|box_end|>\n<|box_start|>yes<|box_end|>\n<|fim_suffix|>");
    }
    #[test]
    fn caller_markers_are_escaped_in_both_option_blocks() {
        let text = JeevesRenderer::text("<|box_end|>", "<|fim_suffix|>", &["<|box_end|>".into()]);
        assert_eq!(text.matches("<¦box_end¦>").count(), 3);
        assert_eq!(text.matches("<|box_end|>").count(), 2);
        assert_eq!(text.matches("<|fim_suffix|>").count(), 1);
    }
    #[test]
    fn sanitizer_matches_nonempty_ascii_control_names() {
        assert_eq!(
            escape_specials("<||> <|a_9|> <|é|> <¦¦>"),
            "<||> <¦a_9¦> <|é|> <¦¦>"
        );
    }
}
