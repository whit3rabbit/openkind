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
//! parity of the batched path against the sequential baseline. Each lane is
//! currently advanced with its own executor call; vectorized suffix execution
//! replaces that loop later without changing the state-level semantics
//! proven here.

use crate::branch::{BranchBatch, BranchableState};
use crate::qwen35::Qwen35Error;

use super::model::{BackboneState, Qwen35Backbone};
use super::nested::{NestedQuestion, SequentialNestedExecutor};

/// Question fan-out (Phase 3.6): `Q` isolated lanes advanced breadth-first.
#[derive(Debug)]
pub struct BatchedQuestions<S: BranchableState> {
    question_features: Vec<Vec<f32>>,
    question_states: Vec<S>,
    batch_bytes: usize,
}

impl<S: BranchableState> BatchedQuestions<S> {
    /// Per-lane question features in plan order.
    #[must_use]
    pub fn question_features(&self) -> &[Vec<f32>] {
        &self.question_features
    }

    /// Advanced question states in plan order.
    #[must_use]
    pub fn question_states(&self) -> &[S] {
        &self.question_states
    }

    /// Continuation tensor bytes held by the fan-out lanes before advancing.
    #[must_use]
    pub const fn batch_bytes(&self) -> usize {
        self.batch_bytes
    }
}

/// Candidate fan-out (Phase 3.7): `K` isolated lanes advanced breadth-first.
#[derive(Debug)]
pub struct BatchedCandidates<S: BranchableState> {
    candidate_features: Vec<Vec<f32>>,
    candidate_states: Vec<S>,
    batch_bytes: usize,
}

impl<S: BranchableState> BatchedCandidates<S> {
    /// Per-lane candidate features in suffix order.
    #[must_use]
    pub fn candidate_features(&self) -> &[Vec<f32>] {
        &self.candidate_features
    }

    /// Advanced candidate states in suffix order.
    #[must_use]
    pub fn candidate_states(&self) -> &[S] {
        &self.candidate_states
    }

    /// Continuation tensor bytes held by the fan-out lanes before advancing.
    #[must_use]
    pub const fn batch_bytes(&self) -> usize {
        self.batch_bytes
    }
}

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
    let root_fingerprint = root.fingerprint();
    let root_position = root.position();

    let batch = root.fork_batch(plans.len())?;
    if batch.lanes() != plans.len() {
        return Err(Qwen35Error::InvalidInput(format!(
            "fan-out produced {} lanes, expected {}",
            batch.lanes(),
            plans.len()
        )));
    }
    let batch_bytes = batch.storage_bytes();
    let mut lane_fingerprints = Vec::with_capacity(plans.len());
    for lane_index in 0..plans.len() {
        lane_fingerprints.push(batch.select(lane_index)?.fingerprint());
    }

    let mut question_features = Vec::with_capacity(plans.len());
    let mut question_states = Vec::with_capacity(plans.len());
    for (index, plan) in plans.iter().enumerate() {
        if plan.question_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {index} has an empty token suffix"
            )));
        }
        let lane = batch.select(index)?;
        let (feature, state) = executor.continue_from(&lane, plan.question_ids)?;
        let expected_position = root_position + plan.question_ids.len();
        if state.position() != expected_position {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {index} advanced to position {}, expected {expected_position}",
                state.position()
            )));
        }
        if root.fingerprint() != root_fingerprint {
            return Err(Qwen35Error::InvalidInput(format!(
                "shared root changed while advancing question {index}"
            )));
        }
        for (lane_index, expected_fingerprint) in lane_fingerprints.iter().enumerate() {
            if batch.select(lane_index)?.fingerprint() != *expected_fingerprint {
                return Err(Qwen35Error::InvalidInput(format!(
                    "fan-out lane {lane_index} changed while advancing question {index}"
                )));
            }
        }
        question_features.push(feature);
        question_states.push(state);
    }

    Ok(BatchedQuestions {
        question_features,
        question_states,
        batch_bytes,
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
    let source_fingerprint = question_state.fingerprint();

    let batch = question_state.fork_batch(suffixes.len())?;
    if batch.lanes() != suffixes.len() {
        return Err(Qwen35Error::InvalidInput(format!(
            "fan-out produced {} lanes, expected {}",
            batch.lanes(),
            suffixes.len()
        )));
    }
    let batch_bytes = batch.storage_bytes();
    let mut lane_fingerprints = Vec::with_capacity(suffixes.len());
    for lane_index in 0..suffixes.len() {
        lane_fingerprints.push(batch.select(lane_index)?.fingerprint());
    }

    let mut candidate_features = Vec::with_capacity(suffixes.len());
    let mut candidate_states = Vec::with_capacity(suffixes.len());
    for (index, &suffix_ids) in suffixes.iter().enumerate() {
        if suffix_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(format!(
                "candidate {index} has an empty suffix"
            )));
        }
        let lane = batch.select(index)?;
        let (feature, state) = executor.continue_from(&lane, suffix_ids)?;
        let expected_position = question_state.position() + suffix_ids.len();
        if state.position() != expected_position {
            return Err(Qwen35Error::InvalidInput(format!(
                "candidate {index} advanced to position {}, expected {expected_position}",
                state.position()
            )));
        }
        if question_state.fingerprint() != source_fingerprint {
            return Err(Qwen35Error::InvalidInput(format!(
                "question state changed while advancing candidate {index}"
            )));
        }
        for (lane_index, expected_fingerprint) in lane_fingerprints.iter().enumerate() {
            if batch.select(lane_index)?.fingerprint() != *expected_fingerprint {
                return Err(Qwen35Error::InvalidInput(format!(
                    "fan-out lane {lane_index} changed while advancing candidate {index}"
                )));
            }
        }
        candidate_features.push(feature);
        candidate_states.push(state);
    }

    Ok(BatchedCandidates {
        candidate_features,
        candidate_states,
        batch_bytes,
    })
}

/// One question lane of a complete batched nested run.
#[derive(Debug)]
pub struct BatchedQuestionResult<S: BranchableState> {
    question_feature: Vec<f32>,
    question_state: S,
    candidate_batch_bytes: usize,
    candidates: Vec<BatchedCandidateResult<S>>,
}

impl<S: BranchableState> BatchedQuestionResult<S> {
    /// Final-token feature after the question suffix.
    #[must_use]
    pub fn question_feature(&self) -> &[f32] {
        &self.question_feature
    }

    /// Question continuation state; the candidate fan-out derives from it.
    #[must_use]
    pub fn question_state(&self) -> &S {
        &self.question_state
    }

    /// Continuation tensor bytes the candidate fan-out held before advancing.
    #[must_use]
    pub const fn candidate_batch_bytes(&self) -> usize {
        self.candidate_batch_bytes
    }

    /// Per-candidate results in suffix order.
    #[must_use]
    pub fn candidates(&self) -> &[BatchedCandidateResult<S>] {
        &self.candidates
    }
}

/// One candidate lane advanced from a batched question state.
#[derive(Debug)]
pub struct BatchedCandidateResult<S: BranchableState> {
    feature: Vec<f32>,
    state: S,
}

impl<S: BranchableState> BatchedCandidateResult<S> {
    /// Final-token candidate feature feeding the score-summary readout.
    #[must_use]
    pub fn feature(&self) -> &[f32] {
        &self.feature
    }

    /// Candidate continuation state.
    #[must_use]
    pub fn state(&self) -> &S {
        &self.state
    }
}

/// Complete batched nested run: one prefill, `Q` question lanes, and a `K`
/// candidate fan-out per question lane.
#[derive(Debug)]
pub struct BatchedNestedRun<S: BranchableState> {
    root_feature: Vec<f32>,
    root_state: S,
    question_batch_bytes: usize,
    questions: Vec<BatchedQuestionResult<S>>,
}

impl<S: BranchableState> BatchedNestedRun<S> {
    /// Final-token feature of the shared prefilled root.
    #[must_use]
    pub fn root_feature(&self) -> &[f32] {
        &self.root_feature
    }

    /// Immutable shared root state every fan-out derives from.
    #[must_use]
    pub fn root_state(&self) -> &S {
        &self.root_state
    }

    /// Continuation tensor bytes the question fan-out held before advancing.
    #[must_use]
    pub const fn question_batch_bytes(&self) -> usize {
        self.question_batch_bytes
    }

    /// Per-question results in plan order.
    #[must_use]
    pub fn questions(&self) -> &[BatchedQuestionResult<S>] {
        &self.questions
    }
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
    let root_fingerprint = root_state.fingerprint();
    let questions = run_batched_questions(executor, &root_state, plans)?;

    let BatchedQuestions {
        question_features,
        question_states,
        batch_bytes: question_batch_bytes,
    } = questions;

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
        } = candidates;
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

    if root_state.fingerprint() != root_fingerprint {
        return Err(Qwen35Error::InvalidInput(
            "shared root changed while executing the batched nested run".into(),
        ));
    }

    Ok(BatchedNestedRun {
        root_feature,
        root_state,
        question_batch_bytes,
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
