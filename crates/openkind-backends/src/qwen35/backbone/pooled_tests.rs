//! Candidate pooling tests use content-bearing synthetic hybrid states.

use std::cell::RefCell;

use openkind_runtime::BatchForwardMode;

use super::batched::{run_batched_nested, run_batched_nested_pooled};
use super::model::BackboneState;
use super::nested::{BatchContinuation, NestedQuestion, SequentialNestedExecutor};
use super::test_support::{SyntheticExecutor, ROOT_IDS};
use crate::qwen35::Qwen35Error;

#[derive(Default)]
struct RecordingExecutor {
    batch_sizes: RefCell<Vec<usize>>,
}

impl SequentialNestedExecutor for RecordingExecutor {
    type State = BackboneState;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        SyntheticExecutor.prefill(input_ids)
    }

    fn continue_from(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        SyntheticExecutor.continue_from(state, suffix_ids)
    }

    fn continue_batch_from(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        self.batch_sizes.borrow_mut().push(states.len());
        let position = states[0].position();
        let mode = if states.len() >= 2 && states.iter().all(|state| state.position() == position) {
            BatchForwardMode::Vectorized
        } else {
            BatchForwardMode::PerLane
        };
        let lanes = states
            .iter()
            .zip(suffix_ids)
            .map(|(&state, &suffix)| SyntheticExecutor.continue_from(state, suffix))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BatchContinuation::new(lanes, mode))
    }
}

fn assert_same_results(
    baseline: &super::batched::BatchedNestedRun<BackboneState>,
    pooled: &super::batched::BatchedNestedRun<BackboneState>,
) {
    assert_eq!(pooled.root_feature(), baseline.root_feature());
    assert_eq!(pooled.questions().len(), baseline.questions().len());
    for (left, right) in pooled.questions().iter().zip(baseline.questions()) {
        assert_eq!(left.question_feature(), right.question_feature());
        assert_eq!(
            left.question_state().strict_fingerprint(),
            right.question_state().strict_fingerprint()
        );
        assert_eq!(left.candidate_batch_bytes(), right.candidate_batch_bytes());
        assert_eq!(left.candidates().len(), right.candidates().len());
        for (candidate, expected) in left.candidates().iter().zip(right.candidates()) {
            assert_eq!(candidate.feature(), expected.feature());
            assert_eq!(
                candidate.state().strict_fingerprint(),
                expected.state().strict_fingerprint()
            );
        }
    }
}

#[test]
fn pooled_candidates_from_equal_position_questions_keep_order_and_isolation() {
    let plans = [
        NestedQuestion {
            question_ids: &[20, 21],
            candidate_suffix_ids: &[&[30], &[31, 32]],
        },
        NestedQuestion {
            question_ids: &[40, 41],
            candidate_suffix_ids: &[&[50, 51, 52], &[53]],
        },
    ];
    let baseline_executor = RecordingExecutor::default();
    let baseline = run_batched_nested(&baseline_executor, ROOT_IDS, &plans).unwrap();
    let pooled_executor = RecordingExecutor::default();
    let pooled = run_batched_nested_pooled(&pooled_executor, ROOT_IDS, &plans, 8).unwrap();

    assert_same_results(&baseline, &pooled);
    assert_eq!(*baseline_executor.batch_sizes.borrow(), [2, 2, 2]);
    assert_eq!(*pooled_executor.batch_sizes.borrow(), [2, 4]);
    assert_eq!(pooled.batch_forward_mode(), BatchForwardMode::Vectorized);
}

#[test]
fn pooled_candidates_use_per_lane_fallback_for_singleton_position_groups() {
    let plans = [
        NestedQuestion {
            question_ids: &[20, 21],
            candidate_suffix_ids: &[&[30], &[31, 32]],
        },
        NestedQuestion {
            question_ids: &[40],
            candidate_suffix_ids: &[&[50, 51, 52]],
        },
    ];
    let baseline = run_batched_nested(&RecordingExecutor::default(), ROOT_IDS, &plans).unwrap();
    let executor = RecordingExecutor::default();
    let pooled = run_batched_nested_pooled(&executor, ROOT_IDS, &plans, 8).unwrap();

    assert_same_results(&baseline, &pooled);
    assert_eq!(*executor.batch_sizes.borrow(), [2, 1, 2]);
    assert_eq!(pooled.batch_forward_mode(), BatchForwardMode::PerLane);
}

#[test]
fn pooled_candidates_reject_invalid_limits_and_empty_suffixes() {
    let plans = [NestedQuestion {
        question_ids: &[20],
        candidate_suffix_ids: &[&[30]],
    }];
    assert!(matches!(
        run_batched_nested_pooled(&SyntheticExecutor, ROOT_IDS, &plans, 1),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("2..=8")
    ));
    let invalid = [NestedQuestion {
        question_ids: &[20],
        candidate_suffix_ids: &[&[]],
    }];
    assert!(matches!(
        run_batched_nested_pooled(&SyntheticExecutor, ROOT_IDS, &invalid, 8),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("empty candidate suffix")
    ));
}
