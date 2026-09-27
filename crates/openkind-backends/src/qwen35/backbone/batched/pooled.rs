//! Opt-in candidate pooling across questions at the same continuation position.
//!
//! This is a diagnostic execution shape. The ordinary batched runner and the
//! automatic scheduler remain unchanged while model-backed timing is open.

use std::collections::BTreeMap;

use crate::qwen35::Qwen35Error;
use openkind_runtime::branch::BranchableState;
use openkind_runtime::BatchForwardMode;

use super::{
    run_batched_questions, BatchedCandidateResult, BatchedNestedRun, BatchedQuestionResult,
    BatchedQuestions,
};
use crate::qwen35::backbone::nested::{NestedQuestion, SequentialNestedExecutor};

/// Prefill once, vectorize question lanes, then pool candidate lanes from
/// different questions when their continuation positions match.
///
/// Candidate batches are limited to `max_lanes` (2–8). A remaining single
/// lane uses the executor's per-lane fallback. Results are restored to the
/// original question and candidate order before readout.
///
/// # Errors
/// Returns an error for invalid plans, state drift, branch mutation, or an
/// executor failure.
pub fn run_batched_nested_pooled<E: SequentialNestedExecutor>(
    executor: &E,
    root_ids: &[u32],
    plans: &[NestedQuestion<'_>],
    max_lanes: usize,
) -> Result<BatchedNestedRun<E::State>, Qwen35Error> {
    if !(2..=8).contains(&max_lanes) {
        return Err(Qwen35Error::InvalidInput(
            "candidate pooling requires 2..=8 lanes".into(),
        ));
    }
    if root_ids.is_empty() {
        return Err(Qwen35Error::InvalidInput(
            "batched nested execution requires a non-empty root".into(),
        ));
    }
    for (question_index, plan) in plans.iter().enumerate() {
        if plan.candidate_suffix_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {question_index} has no candidate suffixes"
            )));
        }
        if plan
            .candidate_suffix_ids
            .iter()
            .any(|suffix| suffix.is_empty())
        {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {question_index} has an empty candidate suffix"
            )));
        }
    }

    let (root_feature, root_state) = executor.prefill(root_ids)?;
    let root_fingerprint = root_state.scheduling_fingerprint();
    let BatchedQuestions {
        question_features,
        question_states,
        batch_bytes: question_batch_bytes,
        batch_forward_mode: question_forward_mode,
    } = run_batched_questions(executor, &root_state, plans)?;

    let source_fingerprints = question_states
        .iter()
        .map(BranchableState::scheduling_fingerprint)
        .collect::<Vec<_>>();
    let mut candidate_batch_bytes = vec![0_usize; plans.len()];
    let mut results: Vec<Vec<Option<BatchedCandidateResult<E::State>>>> = plans
        .iter()
        .map(|plan| (0..plan.candidate_suffix_ids.len()).map(|_| None).collect())
        .collect();
    let mut by_position: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (question_index, plan) in plans.iter().enumerate() {
        for candidate_index in 0..plan.candidate_suffix_ids.len() {
            by_position
                .entry(question_states[question_index].position())
                .or_default()
                .push((question_index, candidate_index));
        }
    }

    let mut batch_forward_mode = question_forward_mode;
    for lanes in by_position.values_mut() {
        // Longest-first packing bounds right-padding work without changing
        // the candidate order seen by the readout.
        lanes.sort_by(|&(aq, ak), &(bq, bk)| {
            plans[bq].candidate_suffix_ids[bk]
                .len()
                .cmp(&plans[aq].candidate_suffix_ids[ak].len())
                .then((aq, ak).cmp(&(bq, bk)))
        });
        let mut offset = 0;
        while offset < lanes.len() {
            let remaining = lanes.len() - offset;
            let width = if remaining == max_lanes + 1 {
                max_lanes - 1
            } else {
                remaining.min(max_lanes)
            };
            let group = &lanes[offset..offset + width];
            let forks = group
                .iter()
                .map(|&(q, _)| question_states[q].fork_one())
                .collect::<Result<Vec<_>, _>>()?;
            let fork_fingerprints = forks
                .iter()
                .map(BranchableState::scheduling_fingerprint)
                .collect::<Vec<_>>();
            let state_refs = forks.iter().collect::<Vec<_>>();
            let suffixes = group
                .iter()
                .map(|&(q, k)| plans[q].candidate_suffix_ids[k])
                .collect::<Vec<_>>();
            for (&(q, _), state) in group.iter().zip(&forks) {
                candidate_batch_bytes[q] =
                    candidate_batch_bytes[q].saturating_add(state.tensor_storage_bytes());
            }
            let continuation = executor.continue_batch_from(&state_refs, &suffixes)?;
            if continuation.batch_forward_mode() == BatchForwardMode::PerLane {
                batch_forward_mode = BatchForwardMode::PerLane;
            }
            let advanced = continuation.into_lanes();
            if advanced.len() != group.len() {
                return Err(Qwen35Error::InvalidInput(format!(
                    "pooled candidate continuation returned {} lanes, expected {}",
                    advanced.len(),
                    group.len()
                )));
            }
            for (index, (&(q, k), (feature, state))) in group.iter().zip(advanced).enumerate() {
                let expected_position = question_states[q].position() + suffixes[index].len();
                if state.position() != expected_position {
                    return Err(Qwen35Error::InvalidInput(format!(
                        "candidate {q}:{k} advanced to position {}, expected {expected_position}",
                        state.position()
                    )));
                }
                results[q][k] = Some(BatchedCandidateResult { feature, state });
            }
            for (index, fork) in forks.iter().enumerate() {
                if fork.scheduling_fingerprint() != fork_fingerprints[index] {
                    return Err(Qwen35Error::InvalidInput(format!(
                        "pooled fork {index} changed while advancing candidates"
                    )));
                }
            }
            for (index, question_state) in question_states.iter().enumerate() {
                if question_state.scheduling_fingerprint() != source_fingerprints[index] {
                    return Err(Qwen35Error::InvalidInput(format!(
                        "question {index} changed while advancing pooled candidates"
                    )));
                }
            }
            offset += width;
        }
    }

    if root_state.scheduling_fingerprint() != root_fingerprint {
        return Err(Qwen35Error::InvalidInput(
            "shared root changed while executing pooled candidates".into(),
        ));
    }
    let questions = question_features
        .into_iter()
        .zip(question_states)
        .zip(results)
        .zip(candidate_batch_bytes)
        .enumerate()
        .map(
            |(q, (((question_feature, question_state), slots), candidate_batch_bytes))| {
                let candidates = slots
                    .into_iter()
                    .enumerate()
                    .map(|(k, slot)| {
                        slot.ok_or_else(|| {
                            Qwen35Error::InvalidInput(format!(
                                "pooled candidate {q}:{k} produced no result"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(BatchedQuestionResult {
                    question_feature,
                    question_state,
                    candidate_batch_bytes,
                    candidates,
                })
            },
        )
        .collect::<Result<Vec<_>, Qwen35Error>>()?;

    Ok(BatchedNestedRun {
        root_feature,
        root_state,
        question_batch_bytes,
        batch_forward_mode,
        questions,
    })
}
