//! Exact safetensors deserialization and feature evaluation for the score-summary head.

use std::collections::BTreeSet;

use safetensors::SafeTensors;

use super::math::{first_argmax, score_summary, stable_softmax};
use super::tensors::{array6, scalar, tensor_values, validate_positive_std, validate_tensor};
use super::types::{
    HeadEvaluation, PolicyAction, PrimitiveKind, FEATURE_WIDTH, SCORE_SUMMARY_WIDTH, TENSOR_SPECS,
};
use crate::qwen35::Qwen35Error;

/// Selected frozen scalar rank head plus score-summary rejection head.
#[derive(Debug, Clone)]
pub struct ScoreSummaryHead {
    rank_mean: Vec<f64>,
    rank_std: Vec<f64>,
    rank_weight: Vec<f64>,
    rank_bias: f64,
    reject_mean: [f64; SCORE_SUMMARY_WIDTH],
    reject_std: [f64; SCORE_SUMMARY_WIDTH],
    reject_weight: [f64; SCORE_SUMMARY_WIDTH],
    reject_bias: f64,
    temperature: f64,
    policy_threshold: f64,
}

impl ScoreSummaryHead {
    /// Load and validate the exact fitted safetensors readout.
    pub fn from_safetensors(
        bytes: &[u8],
        temperature: f64,
        policy_threshold: f64,
    ) -> Result<Self, Qwen35Error> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(Qwen35Error::InvalidInput(
                "temperature must be finite and greater than zero".into(),
            ));
        }
        if !policy_threshold.is_finite() || !(0.0..=1.0).contains(&policy_threshold) {
            return Err(Qwen35Error::InvalidInput(
                "policy threshold must be finite and lie in [0, 1]".into(),
            ));
        }

        let tensors = SafeTensors::deserialize(bytes)?;
        let expected: BTreeSet<String> = TENSOR_SPECS
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect();
        let actual: BTreeSet<String> = tensors
            .names()
            .into_iter()
            .map(|name| name.to_string())
            .collect();
        if expected != actual {
            return Err(Qwen35Error::TensorSetMismatch {
                expected: expected.into_iter().collect(),
                actual: actual.into_iter().collect(),
            });
        }

        for (name, shape) in TENSOR_SPECS {
            validate_tensor(&tensors, name, shape)?;
        }

        let rank_mean = tensor_values(&tensors, "rank.mean")?;
        let rank_std = tensor_values(&tensors, "rank.std")?;
        validate_positive_std("rank.std", &rank_std)?;
        let rank_weight = tensor_values(&tensors, "rank.linear.weight")?;
        let rank_bias = scalar(&tensors, "rank.linear.bias")?;

        // `rank.none` is an intentionally unused training parameter in the
        // exported score-summary graph, but its presence and finiteness remain
        // part of the locked tensor inventory.
        let _unused_rank_none = scalar(&tensors, "rank.none")?;

        let reject_mean = array6(tensor_values(&tensors, "reject.mean")?, "reject.mean")?;
        let reject_std = array6(tensor_values(&tensors, "reject.std")?, "reject.std")?;
        validate_positive_std("reject.std", &reject_std)?;
        let reject_weight = array6(
            tensor_values(&tensors, "reject.net.weight")?,
            "reject.net.weight",
        )?;
        let reject_bias = scalar(&tensors, "reject.net.bias")?;

        Ok(Self {
            rank_mean,
            rank_std,
            rank_weight,
            rank_bias,
            reject_mean,
            reject_std,
            reject_weight,
            reject_bias,
            temperature,
            policy_threshold,
        })
    }

    /// Evaluate candidate-conditioned hidden features through the selected graph.
    pub fn evaluate(
        &self,
        primitive: PrimitiveKind,
        candidate_features: &[Vec<f32>],
    ) -> Result<HeadEvaluation, Qwen35Error> {
        validate_cardinality(primitive, candidate_features.len())?;

        let mut candidate_logits = Vec::with_capacity(candidate_features.len());
        for (index, feature) in candidate_features.iter().enumerate() {
            if feature.len() != FEATURE_WIDTH {
                return Err(Qwen35Error::InvalidInput(format!(
                    "candidate {index} has feature width {}, expected {FEATURE_WIDTH}",
                    feature.len()
                )));
            }
            if let Some((offset, value)) = feature
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(Qwen35Error::InvalidInput(format!(
                    "candidate {index} feature {offset} is not finite: {value}"
                )));
            }

            let logit = feature
                .iter()
                .enumerate()
                .fold(self.rank_bias, |acc, (offset, value)| {
                    let normalized =
                        (f64::from(*value) - self.rank_mean[offset]) / self.rank_std[offset];
                    acc + normalized * self.rank_weight[offset]
                });
            if !logit.is_finite() {
                return Err(Qwen35Error::Numerical(format!(
                    "candidate {index} logit is not finite"
                )));
            }
            candidate_logits.push(logit);
        }

        let none_logit = if primitive == PrimitiveKind::Choice {
            let summary = score_summary(&candidate_logits)?;
            let none = summary
                .iter()
                .enumerate()
                .fold(self.reject_bias, |acc, (offset, value)| {
                    let normalized = (value - self.reject_mean[offset]) / self.reject_std[offset];
                    acc + normalized * self.reject_weight[offset]
                });
            if !none.is_finite() {
                return Err(Qwen35Error::Numerical(
                    "semantic-none logit is not finite".into(),
                ));
            }
            Some(none)
        } else {
            None
        };

        let mut full_logits = candidate_logits.clone();
        if let Some(none) = none_logit {
            full_logits.push(none);
        }
        let full_probabilities = stable_softmax(&full_logits, self.temperature)?;
        let selected = first_argmax(&full_probabilities);
        let top_probability = full_probabilities[selected];
        let selected_candidate_index = (selected < candidate_features.len()).then_some(selected);
        let policy_action = match selected_candidate_index {
            Some(candidate_index) if top_probability >= self.policy_threshold => {
                PolicyAction::Accept { candidate_index }
            }
            _ => PolicyAction::Review,
        };
        let candidate_probabilities = full_probabilities[..candidate_features.len()].to_vec();
        let none_probability = none_logit.map(|_| full_probabilities[candidate_features.len()]);

        Ok(HeadEvaluation::new(
            candidate_logits,
            none_logit,
            candidate_probabilities,
            none_probability,
            selected_candidate_index,
            top_probability,
            policy_action,
        ))
    }
}

fn validate_cardinality(primitive: PrimitiveKind, candidates: usize) -> Result<(), Qwen35Error> {
    let valid = match primitive {
        PrimitiveKind::Choice => candidates >= 1,
        PrimitiveKind::Noul => candidates == 2,
        PrimitiveKind::Score => candidates >= 2,
    };
    if valid {
        Ok(())
    } else {
        Err(Qwen35Error::InvalidInput(format!(
            "{primitive:?} received {candidates} candidate features"
        )))
    }
}

#[cfg(test)]
impl ScoreSummaryHead {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn test_instance(
        rank_mean: Vec<f64>,
        rank_std: Vec<f64>,
        rank_weight: Vec<f64>,
        rank_bias: f64,
        reject_mean: [f64; SCORE_SUMMARY_WIDTH],
        reject_std: [f64; SCORE_SUMMARY_WIDTH],
        reject_weight: [f64; SCORE_SUMMARY_WIDTH],
        reject_bias: f64,
        temperature: f64,
        policy_threshold: f64,
    ) -> Self {
        Self {
            rank_mean,
            rank_std,
            rank_weight,
            rank_bias,
            reject_mean,
            reject_std,
            reject_weight,
            reject_bias,
            temperature,
            policy_threshold,
        }
    }

    pub(crate) fn set_reject_bias(&mut self, val: f64) {
        self.reject_bias = val;
    }

    pub(crate) fn set_policy_threshold(&mut self, val: f64) {
        self.policy_threshold = val;
    }
}
