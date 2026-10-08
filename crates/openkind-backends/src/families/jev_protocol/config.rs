use std::collections::BTreeMap;
use std::ops::Range;

use serde::Deserialize;

use crate::families::support::{temperature_softmax, FamilyError};
use crate::families::wire::QuestionPrimitive;

/// Most options one choice question may carry, for both protocols.
pub const MAX_CHOICE_OPTIONS: usize = 256;

/// Slot widths pinned by both published checkpoints.
const NOUL_SLOTS: usize = 2;
const SCORE_SLOTS: usize = 6;
const CHOICE_SLOTS: usize = 16;

/// Which published model family a `decision_config` belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// Qwen3.5 backbone; options are read from vocabulary logits.
    Jev,
    /// Gemma 4 backbone; options are read from a separate softcapped head.
    Gev,
}

impl Protocol {
    /// Root `config.json` `model_type` for the protocol.
    pub fn model_type(self) -> &'static str {
        match self {
            Self::Jev => "jev",
            Self::Gev => "gev",
        }
    }
}

/// Question kind as the protocol names it. Booleans are `noul`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionKind {
    /// Boolean question, options `false` then `true`.
    Noul,
    /// Six-level ordinal question, levels `0` to `5`.
    Score,
    /// Categorical question.
    Choice,
}

impl DecisionKind {
    /// The kind string rendered into the prompt and used as a config key.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Score => "score",
            Self::Choice => "choice",
        }
    }

    fn slots(self) -> usize {
        match self {
            Self::Noul => NOUL_SLOTS,
            Self::Score => SCORE_SLOTS,
            Self::Choice => CHOICE_SLOTS,
        }
    }
}

impl From<QuestionPrimitive> for DecisionKind {
    fn from(primitive: QuestionPrimitive) -> Self {
        match primitive {
            QuestionPrimitive::Noul => Self::Noul,
            QuestionPrimitive::Score => Self::Score,
            QuestionPrimitive::Choice => Self::Choice,
        }
    }
}

/// The `decision_config` object in the checkpoint's root `config.json`.
///
/// JEV carries `verbalizer_ids` and `bias`; GEV carries `softcap`. A config
/// with the other protocol's fields is rejected rather than reinterpreted.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionConfig {
    ranges: BTreeMap<String, [usize; 2]>,
    #[serde(default)]
    verbalizer_ids: Vec<u32>,
    #[serde(default)]
    bias: Vec<f64>,
    #[serde(default)]
    softcap: Option<f64>,
    temperature_by_type: BTreeMap<String, f64>,
    lora_alpha: f64,
    lora_r: f64,
}

impl DecisionConfig {
    /// Parse and validate a `decision_config` JSON object for `protocol`.
    pub fn from_json(protocol: Protocol, json: &[u8]) -> Result<Self, FamilyError> {
        let config: Self = serde_json::from_slice(json).map_err(|error| {
            FamilyError::InvalidInput(format!("decision_config does not parse: {error}"))
        })?;
        config.validate(protocol)?;
        Ok(config)
    }

    fn validate(&self, protocol: Protocol) -> Result<(), FamilyError> {
        let invalid = |message: String| Err(FamilyError::InvalidInput(message));
        let mut width = 0;
        for kind in [
            DecisionKind::Noul,
            DecisionKind::Score,
            DecisionKind::Choice,
        ] {
            let Some(&[start, end]) = self.ranges.get(kind.as_str()) else {
                return invalid(format!("decision_config.ranges lacks `{}`", kind.as_str()));
            };
            if end < start || end - start != kind.slots() {
                return invalid(format!(
                    "decision_config.ranges.{} is [{start}, {end}], expected {} slots",
                    kind.as_str(),
                    kind.slots()
                ));
            }
            width = width.max(end);
            let temperature = self.temperature_by_type.get(kind.as_str()).copied();
            if !temperature.is_some_and(|t| t.is_finite() && t > 0.0) {
                return invalid(format!(
                    "decision_config.temperature_by_type.{} must be finite and positive",
                    kind.as_str()
                ));
            }
        }
        if self.ranges.len() != 3 {
            return invalid("decision_config.ranges has kinds beyond noul, score, choice".into());
        }
        if !(self.lora_alpha.is_finite() && self.lora_alpha > 0.0)
            || !(self.lora_r.is_finite() && self.lora_r > 0.0)
        {
            return invalid("decision_config lora_alpha and lora_r must be positive".into());
        }
        match protocol {
            Protocol::Jev => {
                if self.softcap.is_some() {
                    return invalid("a jev decision_config must not carry `softcap`".into());
                }
                if self.verbalizer_ids.len() != width || self.bias.len() != width {
                    return invalid(format!(
                        "jev verbalizer_ids ({}) and bias ({}) must each have {width} entries",
                        self.verbalizer_ids.len(),
                        self.bias.len()
                    ));
                }
                if self.bias.iter().any(|value| !value.is_finite()) {
                    return invalid("jev bias contains a non-finite value".into());
                }
            }
            Protocol::Gev => {
                if !self.verbalizer_ids.is_empty() || !self.bias.is_empty() {
                    return invalid(
                        "a gev decision_config must not carry verbalizer_ids or bias".into(),
                    );
                }
                if !self.softcap.is_some_and(|cap| cap.is_finite() && cap > 0.0) {
                    return invalid(
                        "a gev decision_config needs a positive finite `softcap`".into(),
                    );
                }
            }
        }
        Ok(())
    }

    /// LoRA merge scale, `lora_alpha / lora_r`, as the reference applies it.
    pub fn lora_scale(&self) -> f64 {
        self.lora_alpha / self.lora_r
    }

    /// Head output width (GEV `head` out-features): the largest range end.
    pub fn slot_width(&self) -> usize {
        self.ranges
            .values()
            .map(|range| range[1])
            .max()
            .unwrap_or(0)
    }

    /// GEV softcap, `None` for JEV.
    pub fn softcap(&self) -> Option<f64> {
        self.softcap
    }

    fn range(&self, kind: DecisionKind) -> [usize; 2] {
        self.ranges[kind.as_str()]
    }

    fn temperature(&self, kind: DecisionKind) -> f64 {
        self.temperature_by_type[kind.as_str()]
    }

    /// Reject option counts the pinned slots cannot represent.
    fn check_count(kind: DecisionKind, count: usize, choice_cap: usize) -> Result<(), FamilyError> {
        let ok = match kind {
            DecisionKind::Noul => count == NOUL_SLOTS,
            DecisionKind::Score => count == SCORE_SLOTS,
            DecisionKind::Choice => (1..=choice_cap).contains(&count),
        };
        if ok {
            Ok(())
        } else {
            Err(FamilyError::InvalidInput(format!(
                "{} question has {count} options, outside the protocol bounds",
                kind.as_str()
            )))
        }
    }

    /// JEV: vocabulary token ids whose logits are the option scores.
    ///
    /// Choice uses the derived label tokens (up to 256); the other kinds use
    /// the checkpoint's `verbalizer_ids` slice for the kind.
    pub fn jev_token_ids(
        &self,
        kind: DecisionKind,
        count: usize,
        choice_label_ids: &[u32],
    ) -> Result<Vec<u32>, FamilyError> {
        Self::check_count(kind, count, MAX_CHOICE_OPTIONS)?;
        if kind == DecisionKind::Choice {
            return choice_label_ids
                .get(..count)
                .map(<[u32]>::to_vec)
                .ok_or_else(|| {
                    FamilyError::InvalidInput(format!(
                        "only {} choice label tokens are available for {count} options",
                        choice_label_ids.len()
                    ))
                });
        }
        let [start, _] = self.range(kind);
        Ok(self.verbalizer_ids[start..start + count].to_vec())
    }

    /// JEV: option probabilities from the logits gathered at
    /// [`Self::jev_token_ids`]. Adds the slot bias (zero beyond the choice
    /// range), then applies the per-kind temperature softmax.
    pub fn jev_probabilities(
        &self,
        kind: DecisionKind,
        gathered: &[f64],
    ) -> Result<Vec<f64>, FamilyError> {
        Self::check_count(kind, gathered.len(), MAX_CHOICE_OPTIONS)?;
        let [start, end] = self.range(kind);
        let scores: Vec<f64> = gathered
            .iter()
            .enumerate()
            .map(|(index, logit)| {
                let slot = start + index;
                logit + if slot < end { self.bias[slot] } else { 0.0 }
            })
            .collect();
        temperature_softmax(&scores, self.temperature(kind))
    }

    /// GEV: the head-output window holding the first `count` option scores.
    pub fn gev_window(
        &self,
        kind: DecisionKind,
        count: usize,
    ) -> Result<Range<usize>, FamilyError> {
        Self::check_count(kind, count, CHOICE_SLOTS)?;
        let [start, _] = self.range(kind);
        Ok(start..start + count)
    }

    /// GEV: `softcap * tanh(logit / softcap)`, applied to every head output.
    pub fn gev_softcap(&self, logits: &[f64]) -> Result<Vec<f64>, FamilyError> {
        let cap = self
            .softcap
            .ok_or_else(|| FamilyError::InvalidInput("decision_config has no softcap".into()))?;
        Ok(logits
            .iter()
            .map(|logit| cap * (logit / cap).tanh())
            .collect())
    }

    /// GEV: option probabilities from softcapped head outputs in the window
    /// returned by [`Self::gev_window`]. No bias is applied.
    pub fn gev_probabilities(
        &self,
        kind: DecisionKind,
        window: &[f64],
    ) -> Result<Vec<f64>, FamilyError> {
        Self::check_count(kind, window.len(), CHOICE_SLOTS)?;
        temperature_softmax(window, self.temperature(kind))
    }
}
