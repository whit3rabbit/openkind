//! The flat diagnostic must preserve nested candidate features and ordering.

use std::cell::RefCell;

use openkind_runtime::BatchForwardMode;

use super::batched::run_flat_batched_candidates;
use super::model::BackboneState;
use super::nested::{
    run_sequential_nested, BatchContinuation, NestedQuestion, SequentialNestedExecutor,
};
use super::test_support::{SyntheticExecutor, ROOT_IDS};
use crate::qwen35::Qwen35Error;

#[derive(Default)]
struct RecordingExecutor {
    batch_sizes: RefCell<Vec<usize>>,
}

impl SequentialNestedExecutor for RecordingExecutor {
    type State = BackboneState;

    fn prefill(&self, ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        SyntheticExecutor.prefill(ids)
    }

    fn continue_from(
        &self,
        state: &Self::State,
        ids: &[u32],
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        let (_, next) = SyntheticExecutor.continue_from(state, ids)?;
        // Model output at the last token depends on its absolute position,
        // not on where the continuation was split into calls.
        Ok((
            vec![*ids.last().unwrap() as f32, next.position as f32],
            next,
        ))
    }

    fn continue_batch_from(
        &self,
        states: &[&Self::State],
        suffixes: &[&[u32]],
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        self.batch_sizes.borrow_mut().push(states.len());
        let lanes = states
            .iter()
            .zip(suffixes)
            .map(|(&state, &ids)| self.continue_from(state, ids))
            .collect::<Result<Vec<_>, _>>()?;
        let mode = if states.len() >= 2 {
            BatchForwardMode::Vectorized
        } else {
            BatchForwardMode::PerLane
        };
        Ok(BatchContinuation::new(lanes, mode))
    }
}

#[test]
fn flat_lanes_match_nested_features_across_mixed_lengths_and_chunks() {
    let plans = [
        NestedQuestion {
            question_ids: &[20, 21],
            candidate_suffix_ids: &[&[30], &[31, 32], &[33]],
        },
        NestedQuestion {
            question_ids: &[40],
            candidate_suffix_ids: &[&[50, 51], &[52], &[53, 54, 55]],
        },
    ];
    let expected = run_sequential_nested(&RecordingExecutor::default(), ROOT_IDS, &plans).unwrap();
    let executor = RecordingExecutor::default();
    let flat = run_flat_batched_candidates(&executor, ROOT_IDS, &plans, 4).unwrap();

    assert_eq!(*executor.batch_sizes.borrow(), [4, 2]);
    assert_eq!(flat.batch_forward_mode(), BatchForwardMode::Vectorized);
    assert!(flat.peak_batch_bytes() > 0);
    for (q, question) in expected.questions().iter().enumerate() {
        for (k, candidate) in question.candidates().iter().enumerate() {
            assert_eq!(flat.candidate_features()[q][k], candidate.feature());
        }
    }
}

#[test]
fn flat_singleton_uses_per_lane_fallback_and_keeps_order() {
    let candidate_ids = (30..39).map(|id| vec![id]).collect::<Vec<_>>();
    let candidates = candidate_ids.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let plans = [NestedQuestion {
        question_ids: &[20],
        candidate_suffix_ids: &candidates,
    }];
    let executor = RecordingExecutor::default();
    let flat = run_flat_batched_candidates(&executor, ROOT_IDS, &plans, 8).unwrap();
    assert_eq!(*executor.batch_sizes.borrow(), [8, 1]);
    assert_eq!(flat.batch_forward_mode(), BatchForwardMode::PerLane);
    assert_eq!(flat.candidate_features()[0].len(), 9);
}

#[test]
fn flat_rejects_invalid_limits_and_empty_suffixes() {
    let valid = [NestedQuestion {
        question_ids: &[20],
        candidate_suffix_ids: &[&[30]],
    }];
    assert!(matches!(
        run_flat_batched_candidates(&SyntheticExecutor, ROOT_IDS, &valid, 1),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("2..=8")
    ));
    let invalid = [NestedQuestion {
        question_ids: &[20],
        candidate_suffix_ids: &[&[]],
    }];
    assert!(matches!(
        run_flat_batched_candidates(&SyntheticExecutor, ROOT_IDS, &invalid, 8),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("empty suffix")
    ));
}
