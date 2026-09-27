//! Phase 3.6/3.7 breadth-first batched execution over branchable state.
//!
//! Phase 3.6 fans one immutable prefilled root into a `Q` lane
//! [`BranchableState::fork_batch`] and evaluates the question suffixes
//! breadth-first, so every question state exists before any candidate work
//! starts. Phase 3.7 fans each question state into a `K` lane batch and
//! evaluates the candidate suffixes breadth-first the same way. Every lane
//! owns its complete continuation tensors, so advancing one lane can never
//! mutate its siblings, its source state, or the shared root, and the runners
//! verify that isolation structurally after every stage and fail closed.
//!
//! Like the sequential baseline, the orchestration is expressed against
//! [`SequentialNestedExecutor`] and validated offline on synthetic states;
//! the `qwen35_batched_parity` example establishes the checkpoint-gated
//! parity of the batched path against the sequential baseline. The executor
//! reports whether lane advancement used per-lane calls or a true vectorized
//! forward; state fan-out alone never implies compute batching.

use crate::qwen35::Qwen35Error;
use openkind_runtime::branch::{BranchBatch, BranchableState};
use openkind_runtime::BatchForwardMode;

use super::model::{BackboneState, Qwen35Backbone};
use super::nested::{NestedQuestion, SequentialNestedExecutor};

mod flat;
mod pooled;
mod types;

pub use flat::{run_flat_batched_candidates, FlatBatchRun};
pub use pooled::run_batched_nested_pooled;
pub use types::*;

/// Fan one prefilled root into `plans.len()` isolated lanes and evaluate the
/// question suffixes breadth-first (Phase 3.6).
///
/// The source root and every fan-out lane stay unchanged while other lanes
/// advance; violations fail closed.
///
/// # Errors
/// Returns [`Qwen35Error`] for empty plans, executor failures, position
/// drift, and branch immutability violations.
pub fn run_batched_questions<E: SequentialNestedExecutor>(
    executor: &E,
    root: &E::State,
    plans: &[NestedQuestion<'_>],
) -> Result<BatchedQuestions<E::State>, Qwen35Error> {
    if plans.is_empty() {
        return Err(Qwen35Error::InvalidInput(
            "batched question execution requires at least one question".into(),
        ));
    }
    let root_fingerprint = root.scheduling_fingerprint();
    let root_position = root.position();

    let batch = root.fork_batch(plans.len())?;
    if batch.lanes() != plans.len() {
        return Err(Qwen35Error::InvalidInput(format!(
            "fan-out produced {} lanes, expected {}",
            batch.lanes(),
            plans.len()
        )));
    }
    let batch_bytes = batch.tensor_storage_bytes();
    let mut lane_fingerprints = Vec::with_capacity(plans.len());
    for lane_index in 0..plans.len() {
        lane_fingerprints.push(batch.select(lane_index)?.scheduling_fingerprint());
    }

    let mut suffixes = Vec::with_capacity(plans.len());
    for (index, plan) in plans.iter().enumerate() {
        if plan.question_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {index} has an empty token suffix"
            )));
        }
        suffixes.push(plan.question_ids);
    }
    let lanes = (0..plans.len())
        .map(|index| batch.select(index))
        .collect::<Result<Vec<_>, _>>()?;
    let lane_refs = lanes.iter().collect::<Vec<_>>();
    let continuation = executor.continue_batch_from(&lane_refs, &suffixes)?;
    let batch_forward_mode = continuation.batch_forward_mode();
    let advanced_lanes = continuation.into_lanes();
    if advanced_lanes.len() != plans.len() {
        return Err(Qwen35Error::InvalidInput(format!(
            "question continuation returned {} lanes, expected {}",
            advanced_lanes.len(),
            plans.len()
        )));
    }

    let mut question_features = Vec::with_capacity(plans.len());
    let mut question_states = Vec::with_capacity(plans.len());
    for (index, (feature, state)) in advanced_lanes.into_iter().enumerate() {
        let plan = &plans[index];
        let expected_position = root_position + plan.question_ids.len();
        if state.position() != expected_position {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {index} advanced to position {}, expected {expected_position}",
                state.position()
            )));
        }
        question_features.push(feature);
        question_states.push(state);
    }
    if root.scheduling_fingerprint() != root_fingerprint {
        return Err(Qwen35Error::InvalidInput(
            "shared root changed while advancing question lanes".into(),
        ));
    }
    for (lane_index, expected_fingerprint) in lane_fingerprints.iter().enumerate() {
        if batch.select(lane_index)?.scheduling_fingerprint() != *expected_fingerprint {
            return Err(Qwen35Error::InvalidInput(format!(
                "fan-out lane {lane_index} changed while advancing question lanes"
            )));
        }
    }

    Ok(BatchedQuestions {
        question_features,
        question_states,
        batch_bytes,
        batch_forward_mode,
    })
}

/// Fan one question state into `suffixes.len()` isolated lanes and evaluate
/// the candidate suffixes breadth-first (Phase 3.7).
///
/// The source question state and every fan-out lane stay unchanged while
/// other lanes advance; violations fail closed.
///
/// # Errors
/// Returns [`Qwen35Error`] for empty suffix lists, executor failures,
/// position drift, and branch immutability violations.
pub fn run_batched_candidates<E: SequentialNestedExecutor>(
    executor: &E,
    question_state: &E::State,
    suffixes: &[&[u32]],
) -> Result<BatchedCandidates<E::State>, Qwen35Error> {
    if suffixes.is_empty() {
        return Err(Qwen35Error::InvalidInput(
            "batched candidate execution requires at least one candidate suffix".into(),
        ));
    }
    let source_fingerprint = question_state.scheduling_fingerprint();

    let batch = question_state.fork_batch(suffixes.len())?;
    if batch.lanes() != suffixes.len() {
        return Err(Qwen35Error::InvalidInput(format!(
            "fan-out produced {} lanes, expected {}",
            batch.lanes(),
            suffixes.len()
        )));
    }
    let batch_bytes = batch.tensor_storage_bytes();
    let mut lane_fingerprints = Vec::with_capacity(suffixes.len());
    for lane_index in 0..suffixes.len() {
        lane_fingerprints.push(batch.select(lane_index)?.scheduling_fingerprint());
    }

    for (index, suffix_ids) in suffixes.iter().enumerate() {
        if suffix_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(format!(
                "candidate {index} has an empty suffix"
            )));
        }
    }
    let lanes = (0..suffixes.len())
        .map(|index| batch.select(index))
        .collect::<Result<Vec<_>, _>>()?;
    let lane_refs = lanes.iter().collect::<Vec<_>>();
    let continuation = executor.continue_batch_from(&lane_refs, suffixes)?;
    let batch_forward_mode = continuation.batch_forward_mode();
    let advanced_lanes = continuation.into_lanes();
    if advanced_lanes.len() != suffixes.len() {
        return Err(Qwen35Error::InvalidInput(format!(
            "candidate continuation returned {} lanes, expected {}",
            advanced_lanes.len(),
            suffixes.len()
        )));
    }

    let mut candidate_features = Vec::with_capacity(suffixes.len());
    let mut candidate_states = Vec::with_capacity(suffixes.len());
    for (index, (feature, state)) in advanced_lanes.into_iter().enumerate() {
        let suffix_ids = suffixes[index];
        let expected_position = question_state.position() + suffix_ids.len();
        if state.position() != expected_position {
            return Err(Qwen35Error::InvalidInput(format!(
                "candidate {index} advanced to position {}, expected {expected_position}",
                state.position()
            )));
        }
        candidate_features.push(feature);
        candidate_states.push(state);
    }
    if question_state.scheduling_fingerprint() != source_fingerprint {
        return Err(Qwen35Error::InvalidInput(
            "question state changed while advancing candidate lanes".into(),
        ));
    }
    for (lane_index, expected_fingerprint) in lane_fingerprints.iter().enumerate() {
        if batch.select(lane_index)?.scheduling_fingerprint() != *expected_fingerprint {
            return Err(Qwen35Error::InvalidInput(format!(
                "fan-out lane {lane_index} changed while advancing candidate lanes"
            )));
        }
    }

    Ok(BatchedCandidates {
        candidate_features,
        candidate_states,
        batch_bytes,
        batch_forward_mode,
    })
}

/// Execute the Phase 3.6/3.7 breadth-first graph: prefill the shared state
/// once, fan it into `Q` question lanes, then fan each question state into
/// `K` candidate lanes.
///
/// # Errors
/// Returns [`Qwen35Error`] for empty plans, executor failures, position
/// drift, and branch immutability violations.
pub fn run_batched_nested<E: SequentialNestedExecutor>(
    executor: &E,
    root_ids: &[u32],
    plans: &[NestedQuestion<'_>],
) -> Result<BatchedNestedRun<E::State>, Qwen35Error> {
    if root_ids.is_empty() {
        return Err(Qwen35Error::InvalidInput(
            "batched nested execution requires a non-empty root".into(),
        ));
    }
    let (root_feature, root_state) = executor.prefill(root_ids)?;
    let root_fingerprint = root_state.scheduling_fingerprint();
    let questions = run_batched_questions(executor, &root_state, plans)?;

    let BatchedQuestions {
        question_features,
        question_states,
        batch_bytes: question_batch_bytes,
        batch_forward_mode: question_forward_mode,
    } = questions;

    let mut batch_forward_mode = question_forward_mode;
    let mut results = Vec::with_capacity(question_states.len());
    for ((question_state, question_feature), plan) in question_states
        .into_iter()
        .zip(question_features)
        .zip(plans)
    {
        let candidates =
            run_batched_candidates(executor, &question_state, plan.candidate_suffix_ids)?;
        let BatchedCandidates {
            candidate_features,
            candidate_states,
            batch_bytes: candidate_batch_bytes,
            batch_forward_mode: candidate_forward_mode,
        } = candidates;
        if candidate_forward_mode == BatchForwardMode::PerLane {
            batch_forward_mode = BatchForwardMode::PerLane;
        }
        results.push(BatchedQuestionResult {
            question_feature,
            question_state,
            candidate_batch_bytes,
            candidates: candidate_features
                .into_iter()
                .zip(candidate_states)
                .map(|(feature, state)| BatchedCandidateResult { feature, state })
                .collect(),
        });
    }

    if root_state.scheduling_fingerprint() != root_fingerprint {
        return Err(Qwen35Error::InvalidInput(
            "shared root changed while executing the batched nested run".into(),
        ));
    }

    Ok(BatchedNestedRun {
        root_feature,
        root_state,
        question_batch_bytes,
        batch_forward_mode,
        questions: results,
    })
}

impl Qwen35Backbone {
    /// Breadth-first batched execution against the pinned Qwen3.5 backbone
    /// (Phase 3.6 question fan-out plus Phase 3.7 candidate fan-outs).
    ///
    /// # Errors
    /// Returns [`Qwen35Error`] for empty plans, identity mismatches, and
    /// backbone execution failures.
    pub fn evaluate_batched_nested(
        &self,
        root_ids: &[u32],
        plans: &[NestedQuestion<'_>],
    ) -> Result<BatchedNestedRun<BackboneState>, Qwen35Error> {
        run_batched_nested(self, root_ids, plans)
    }
}
