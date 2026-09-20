//! Unit tests for score summary head, statistics, and safetensors validation.

use std::borrow::Cow;

use safetensors::tensor::{serialize, Dtype, View};

use super::evaluation::ScoreSummaryHead;
use super::math::{score_summary, stable_softmax};
use super::types::{PolicyAction, PrimitiveKind, FEATURE_WIDTH, SCORE_SUMMARY_WIDTH, TENSOR_SPECS};
use crate::qwen35::Qwen35Error;

struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn serialized_head(mut alter: impl FnMut(&str, &mut OwnedTensor)) -> Vec<u8> {
    let tensors = TENSOR_SPECS.iter().map(|(name, shape)| {
        let elements = shape.iter().product::<usize>().max(1);
        let fill = if name.ends_with(".std") { 1.0_f32 } else { 0.0 };
        let mut tensor = OwnedTensor {
            dtype: Dtype::F32,
            shape: shape.to_vec(),
            data: (0..elements).flat_map(|_| fill.to_le_bytes()).collect(),
        };
        alter(name, &mut tensor);
        ((*name).to_string(), tensor)
    });
    serialize(tensors, &None).unwrap()
}

fn test_head() -> ScoreSummaryHead {
    ScoreSummaryHead::test_instance(
        vec![0.0; FEATURE_WIDTH],
        vec![1.0; FEATURE_WIDTH],
        {
            let mut values = vec![0.0; FEATURE_WIDTH];
            values[0] = 1.0;
            values
        },
        0.0,
        [0.0; SCORE_SUMMARY_WIDTH],
        [1.0; SCORE_SUMMARY_WIDTH],
        [0.0; SCORE_SUMMARY_WIDTH],
        -100.0,
        1.0,
        0.5,
    )
}

fn feature(first: f32) -> Vec<f32> {
    let mut values = vec![0.0; FEATURE_WIDTH];
    values[0] = first;
    values
}

#[test]
fn score_summary_matches_population_statistics() {
    let summary = score_summary(&[1.0, 2.0, 4.0]).unwrap();
    assert_eq!(summary[0], 4.0);
    assert_eq!(summary[1], 2.0);
    assert!((summary[2] - 7.0 / 3.0).abs() < 1e-12);
    assert!((summary[3] - 1.247_219_128_924_647).abs() < 1e-12);
    let expected_lme = 4.0 + ((-3.0_f64).exp() + (-2.0_f64).exp() + 1.0).ln() - 3.0_f64.ln();
    assert!((summary[4] - expected_lme).abs() < 1e-12);
    assert_eq!(summary[5], 3.0_f64.ln());
}

#[test]
fn softmax_is_stable_for_large_logits() {
    let probabilities = stable_softmax(&[10_000.0, 9_999.0], 1.0).unwrap();
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    assert!(probabilities[0] > probabilities[1]);
}

#[test]
fn tie_breaking_selects_first_candidate() {
    let result = test_head()
        .evaluate(PrimitiveKind::Noul, &[feature(1.0), feature(1.0)])
        .unwrap();
    assert_eq!(result.selected_candidate_index(), Some(0));
    assert_eq!(
        result.policy_action(),
        PolicyAction::Accept { candidate_index: 0 }
    );
}

#[test]
fn choice_adds_none_while_other_primitives_do_not() {
    let head = test_head();
    let choice = head
        .evaluate(PrimitiveKind::Choice, &[feature(2.0)])
        .unwrap();
    assert!(choice.none_logit().is_some());
    assert!(choice.none_probability().is_some());
    assert_eq!(choice.full_probabilities().len(), 2);

    let noul = head
        .evaluate(PrimitiveKind::Noul, &[feature(2.0), feature(1.0)])
        .unwrap();
    assert_eq!(noul.none_logit(), None);
    assert_eq!(noul.none_probability(), None);

    let score = head
        .evaluate(PrimitiveKind::Score, &[feature(2.0), feature(1.0)])
        .unwrap();
    assert_eq!(score.none_logit(), None);
    assert_eq!(score.none_probability(), None);
}

#[test]
fn semantic_none_and_low_confidence_route_to_review() {
    let mut none_head = test_head();
    none_head.set_reject_bias(100.0);
    let none = none_head
        .evaluate(PrimitiveKind::Choice, &[feature(1.0)])
        .unwrap();
    assert_eq!(none.selected_candidate_index(), None);
    assert_eq!(none.policy_action(), PolicyAction::Review);

    let mut threshold_head = test_head();
    threshold_head.set_policy_threshold(0.9);
    let low_confidence = threshold_head
        .evaluate(PrimitiveKind::Noul, &[feature(0.0), feature(0.0)])
        .unwrap();
    assert_eq!(low_confidence.policy_action(), PolicyAction::Review);
}

#[test]
fn primitive_cardinality_and_features_are_validated() {
    let head = test_head();
    assert!(head.evaluate(PrimitiveKind::Choice, &[]).is_err());
    assert!(head.evaluate(PrimitiveKind::Noul, &[feature(0.0)]).is_err());
    assert!(head
        .evaluate(PrimitiveKind::Score, &[feature(0.0)])
        .is_err());
    assert!(head
        .evaluate(PrimitiveKind::Choice, &[vec![0.0; 3]])
        .is_err());
    let mut non_finite = feature(0.0);
    non_finite[2] = f32::NAN;
    assert!(head.evaluate(PrimitiveKind::Choice, &[non_finite]).is_err());
}

#[test]
fn tensor_dtype_shape_finiteness_and_std_are_validated() {
    let valid = serialized_head(|_, _| {});
    assert!(ScoreSummaryHead::from_safetensors(&valid, 1.0, 0.98).is_ok());

    let missing = serialize(Vec::<(String, OwnedTensor)>::new(), &None).unwrap();
    assert!(matches!(
        ScoreSummaryHead::from_safetensors(&missing, 1.0, 0.98),
        Err(Qwen35Error::TensorSetMismatch { .. })
    ));

    let wrong_dtype = serialized_head(|name, tensor| {
        if name == "rank.mean" {
            tensor.dtype = Dtype::F64;
            tensor.data = vec![0; FEATURE_WIDTH * 8];
        }
    });
    assert!(matches!(
        ScoreSummaryHead::from_safetensors(&wrong_dtype, 1.0, 0.98),
        Err(Qwen35Error::InvalidTensor { name, .. }) if name == "rank.mean"
    ));

    let wrong_shape = serialized_head(|name, tensor| {
        if name == "rank.mean" {
            tensor.shape = vec![1, FEATURE_WIDTH];
        }
    });
    assert!(matches!(
        ScoreSummaryHead::from_safetensors(&wrong_shape, 1.0, 0.98),
        Err(Qwen35Error::InvalidTensor { name, .. }) if name == "rank.mean"
    ));

    let non_finite = serialized_head(|name, tensor| {
        if name == "rank.linear.bias" {
            tensor.data = f32::NAN.to_le_bytes().to_vec();
        }
    });
    assert!(matches!(
        ScoreSummaryHead::from_safetensors(&non_finite, 1.0, 0.98),
        Err(Qwen35Error::InvalidTensor { name, .. }) if name == "rank.linear.bias"
    ));

    let zero_std = serialized_head(|name, tensor| {
        if name == "reject.std" {
            tensor.data = vec![0; SCORE_SUMMARY_WIDTH * 4];
        }
    });
    assert!(matches!(
        ScoreSummaryHead::from_safetensors(&zero_std, 1.0, 0.98),
        Err(Qwen35Error::InvalidTensor { name, .. }) if name == "reject.std"
    ));
}
