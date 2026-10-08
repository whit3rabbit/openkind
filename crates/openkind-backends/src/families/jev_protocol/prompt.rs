use crate::families::support::FamilyError;

use super::config::{DecisionKind, Protocol, MAX_CHOICE_OPTIONS};

/// Prefix the GEV prompt starts with; JEV has none.
pub const GEV_BOS: &str = "<bos>";

/// Letters labelling GEV choice options, at most 16 per pass.
pub const GEV_GROUP_LABELS: [&str; 16] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
];

/// Choice labels shown in the prompt and the vocabulary token each is scored at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChoiceLabels {
    names: Vec<String>,
    token_ids: Vec<u32>,
}

impl ChoiceLabels {
    /// JEV: derive up to 256 single-token labels (`A`..`Z`, then `AA`..`ZZ`).
    ///
    /// A label qualifies when it encodes to exactly one token and that token
    /// also appears in the encoding of `"x\n{label}) y"`, the form the
    /// prompt renders. `encode` must be the checkpoint tokenizer without
    /// special tokens.
    pub fn derive(
        mut encode: impl FnMut(&str) -> Result<Vec<u32>, FamilyError>,
    ) -> Result<Self, FamilyError> {
        let letters: Vec<char> = ('A'..='Z').collect();
        let candidates = letters.iter().map(|c| c.to_string()).chain(
            letters
                .iter()
                .flat_map(|a| letters.iter().map(move |b| format!("{a}{b}"))),
        );
        let (mut names, mut token_ids) = (Vec::new(), Vec::new());
        for label in candidates {
            let ids = encode(&label)?;
            let context = encode(&format!("x\n{label}) y"))?;
            if let [id] = ids[..] {
                if context.contains(&id) {
                    names.push(label);
                    token_ids.push(id);
                }
            }
            if names.len() == MAX_CHOICE_OPTIONS {
                break;
            }
        }
        Ok(Self { names, token_ids })
    }

    /// 256 distinct synthetic labels with unique token ids, for adapter tests.
    #[cfg(test)]
    pub(crate) fn synthetic() -> Self {
        let names: Vec<String> = (0..MAX_CHOICE_OPTIONS)
            .map(|index| format!("L{index}"))
            .collect();
        let token_ids = (0..MAX_CHOICE_OPTIONS as u32)
            .map(|index| 1_000 + index)
            .collect();
        Self { names, token_ids }
    }

    /// GEV: the fixed `A`..`P` labels, with no vocabulary tokens.
    pub fn gev() -> Self {
        Self {
            names: GEV_GROUP_LABELS
                .iter()
                .map(|label| (*label).to_owned())
                .collect(),
            token_ids: Vec::new(),
        }
    }

    /// Prompt label per option index.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Scored vocabulary token per option index; empty for GEV.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

/// `"{label}: {criterion}"`, or just the label when the criterion is absent
/// or repeats the label. The wire's `unpack_question` fills a missing
/// criterion with the label, so equality stands in for the reference's
/// `None`/empty check.
pub fn choice_option_text(label: &str, criterion: &str) -> String {
    if criterion.is_empty() || criterion == label {
        label.to_owned()
    } else {
        format!("{label}: {criterion}")
    }
}

/// Render one single-pass prompt.
///
/// `options` are the per-option texts: for choice, [`choice_option_text`]
/// values, which are prefixed with `labels` here; for noul and score, the
/// fixed `false`/`true` or `0`..`5` strings. Score criteria are never shown
/// to the model; they are response metadata only.
pub fn render_prompt(
    protocol: Protocol,
    kind: DecisionKind,
    state: &str,
    question: &str,
    options: &[String],
    labels: &ChoiceLabels,
) -> Result<String, FamilyError> {
    let lines: Vec<String> = if kind == DecisionKind::Choice {
        let names = labels.names();
        if options.len() > names.len() {
            return Err(FamilyError::InvalidInput(format!(
                "{} options exceed the {} available choice labels",
                options.len(),
                names.len()
            )));
        }
        options
            .iter()
            .zip(names)
            .map(|(option, name)| format!("{name}) {option}"))
            .collect()
    } else {
        options.to_vec()
    };
    let prefix = match protocol {
        Protocol::Jev => "",
        Protocol::Gev => GEV_BOS,
    };
    Ok(format!(
        "{prefix}[kind] {}\n[state] {state}\n[question] {question}\n[options]\n{}\n[decision]:",
        kind.as_str(),
        lines.join("\n")
    ))
}
