//! Phase 3.5 sequential nested execution over branchable continuation state.
//!
//! The nested graph prefills the shared state once, then evaluates every
//! question by forking the immutable root, advancing that fork with the
//! question tokens, forking the question state per candidate, and advancing
//! each candidate fork with its suffix. Every fork comes from
//! [`BranchableState::fork_one`], so a candidate branch can never mutate its
//! sibling branches, its question state, or the shared root, and the runner
//! verifies that immutability structurally after every stage and fails
//! closed.
//!
//! The orchestration is expressed against [`SequentialNestedExecutor`] so its
//! semantics stay testable offline; the pinned Qwen3.5 backbone provides the
//! checkpoint-gated implementation validated by the `qwen35_nested_parity`
//! example.

use crate::qwen35::{ExecutionControl, Qwen35Error};
use openkind_runtime::branch::BranchableState;
use openkind_runtime::BatchForwardMode;

use super::model::{BackboneState, Qwen35Backbone};

/// Model execution backend for the sequential nested graph.
///
/// Implementations return the final-token feature of the executed sequence
/// together with its complete continuation state, and must never mutate a
/// state passed to [`SequentialNestedExecutor::continue_from`].
pub trait SequentialNestedExecutor {
    /// Complete continuation state produced and consumed by this executor.
    type State: BranchableState;

    /// Evaluate a fresh sequence and return its final-token feature plus the
    /// complete continuation state.
    ///
    /// # Errors
    /// Fails when the sequence cannot be executed.
    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error>;

    /// Evaluate a fresh sequence with request-scoped lifecycle checks.
    /// Executors that cannot interrupt an active kernel still check before
    /// and after the forward so callers can bound cancellation latency.
    fn prefill_controlled(
        &self,
        input_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        control.check()?;
        let output = self.prefill(input_ids)?;
        control.check()?;
        Ok(output)
    }

    /// Continue `state` with `suffix_ids` without mutating it, returning the
    /// suffix's final-token feature plus the advanced continuation state.
    ///
    /// # Errors
    /// Fails when the state identity cannot be continued or the suffix cannot
    /// be executed.
    fn continue_from(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error>;

    /// Continue several independent states in one executor operation.
    ///
    /// The default preserves the per-lane implementation. A backend that
    /// implements a real multi-lane forward may override this method and
    /// return [`BatchForwardMode::Vectorized`]. The reported mode describes
    /// the physical forwards performed, not the requested state topology.
    ///
    /// # Errors
    /// Fails when the lane counts differ, a suffix is empty, or any lane
    /// cannot be continued.
    fn continue_batch_from(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        validate_batch_inputs(states.len(), suffix_ids)?;
        let mut lanes = Vec::with_capacity(states.len());
        for (&state, &suffix) in states.iter().zip(suffix_ids) {
            lanes.push(self.continue_from(state, suffix)?);
        }
        Ok(BatchContinuation::new(lanes, BatchForwardMode::PerLane))
    }

    /// Request-scoped variant of [`Self::continue_batch_from`].
    ///
    /// The default checks lifecycle state around each per-lane forward. A
    /// backend that overrides both batch methods may check once before and
    /// after a non-interruptible vectorized kernel.
    fn continue_batch_from_controlled(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
        control: &ExecutionControl,
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        validate_batch_inputs(states.len(), suffix_ids)?;
        let mut lanes = Vec::with_capacity(states.len());
        for (&state, &suffix) in states.iter().zip(suffix_ids) {
            lanes.push(self.continue_from_controlled(state, suffix, control)?);
        }
        Ok(BatchContinuation::new(lanes, BatchForwardMode::PerLane))
    }

    /// Continue a sequence with request-scoped lifecycle checks.
    fn continue_from_controlled(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        control.check()?;
        let output = self.continue_from(state, suffix_ids)?;
        control.check()?;
        Ok(output)
    }
}

/// Results from advancing a set of continuation lanes.
#[derive(Debug)]
pub struct BatchContinuation<S> {
    lanes: Vec<(Vec<f32>, S)>,
    batch_forward_mode: BatchForwardMode,
}

impl<S> BatchContinuation<S> {
    /// Build a batch result with its observed physical forward mode.
    #[must_use]
    pub fn new(lanes: Vec<(Vec<f32>, S)>, batch_forward_mode: BatchForwardMode) -> Self {
        Self {
            lanes,
            batch_forward_mode,
        }
    }

    /// Consume the result and return features and advanced states in lane order.
    #[must_use]
    pub fn into_lanes(self) -> Vec<(Vec<f32>, S)> {
        self.lanes
    }

    /// Consume the result and return its lanes and physical forward mode.
    #[must_use]
    pub fn into_parts(self) -> (Vec<(Vec<f32>, S)>, BatchForwardMode) {
        (self.lanes, self.batch_forward_mode)
    }

    /// Physical forward mode observed for this batch.
    #[must_use]
    pub const fn batch_forward_mode(&self) -> BatchForwardMode {
        self.batch_forward_mode
    }
}

fn validate_batch_inputs(states: usize, suffix_ids: &[&[u32]]) -> Result<(), Qwen35Error> {
    if states == 0 || states != suffix_ids.len() {
        return Err(Qwen35Error::InvalidInput(format!(
            "batch continuation has {states} states and {} suffixes",
            suffix_ids.len()
        )));
    }
    if suffix_ids.iter().any(|suffix| suffix.is_empty()) {
        return Err(Qwen35Error::InvalidInput(
            "batch continuation contains an empty suffix".into(),
        ));
    }
    Ok(())
}

/// Adapter that runs an existing strategy through one request's control token.
pub(crate) struct ControlledExecutor<'a, E> {
    executor: &'a E,
    control: &'a ExecutionControl,
}

impl<'a, E> ControlledExecutor<'a, E> {
    pub(crate) const fn new(executor: &'a E, control: &'a ExecutionControl) -> Self {
        Self { executor, control }
    }
}

impl<E: SequentialNestedExecutor> SequentialNestedExecutor for ControlledExecutor<'_, E> {
    type State = E::State;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        self.executor.prefill_controlled(input_ids, self.control)
    }

    fn continue_from(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        self.executor
            .continue_from_controlled(state, suffix_ids, self.control)
    }

    fn continue_batch_from(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        self.executor
            .continue_batch_from_controlled(states, suffix_ids, self.control)
    }

    fn prefill_controlled(
        &self,
        input_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        self.executor.prefill_controlled(input_ids, control)
    }

    fn continue_from_controlled(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        self.executor
            .continue_from_controlled(state, suffix_ids, control)
    }

    fn continue_batch_from_controlled(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
        control: &ExecutionControl,
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        self.executor
            .continue_batch_from_controlled(states, suffix_ids, control)
    }
}

/// One question plan inside a nested run.
#[derive(Debug, Clone, Copy)]
pub struct NestedQuestion<'a> {
    /// Question token suffix applied to a fork of the prefilled root.
    pub question_ids: &'a [u32],
    /// Per-candidate token suffixes, each applied to a fresh fork of the
    /// question state.
    pub candidate_suffix_ids: &'a [&'a [u32]],
}

/// Root of one completed sequential nested run.
#[derive(Debug)]
pub struct NestedRun<S: BranchableState> {
    root_feature: Vec<f32>,
    root_state: S,
    questions: Vec<NestedQuestionResult<S>>,
}

impl<S: BranchableState> NestedRun<S> {
    /// Final-token feature of the shared prefilled root.
    #[must_use]
    pub fn root_feature(&self) -> &[f32] {
        &self.root_feature
    }

    /// Immutable shared root state every question fork derives from.
    #[must_use]
    pub fn root_state(&self) -> &S {
        &self.root_state
    }

    /// Per-question results in plan order.
    #[must_use]
    pub fn questions(&self) -> &[NestedQuestionResult<S>] {
        &self.questions
    }
}

/// One question branch and its candidate branches.
#[derive(Debug)]
pub struct NestedQuestionResult<S: BranchableState> {
    question_feature: Vec<f32>,
    question_state: S,
    candidates: Vec<NestedCandidateResult<S>>,
}

impl<S: BranchableState> NestedQuestionResult<S> {
    /// Final-token feature after the question suffix.
    #[must_use]
    pub fn question_feature(&self) -> &[f32] {
        &self.question_feature
    }

    /// Question continuation state; candidate forks derive from it.
    #[must_use]
    pub fn question_state(&self) -> &S {
        &self.question_state
    }

    /// Per-candidate results in plan order.
    #[must_use]
    pub fn candidates(&self) -> &[NestedCandidateResult<S>] {
        &self.candidates
    }
}

/// One candidate branch advanced from a question state.
#[derive(Debug)]
pub struct NestedCandidateResult<S: BranchableState> {
    feature: Vec<f32>,
    state: S,
}

impl<S: BranchableState> NestedCandidateResult<S> {
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

/// Execute the sequential nested graph: prefill once, then
/// `fork question -> advance question -> fork candidate -> advance candidate`.
///
/// The runner fail-closes when the executor advances a state to an unexpected
/// position, when a question state changes while its candidates execute, or
/// when the shared root changes over the run.
///
/// # Errors
/// Returns [`Qwen35Error`] for empty plans, executor failures, and branch
/// immutability violations.
pub fn run_sequential_nested<E: SequentialNestedExecutor>(
    executor: &E,
    root_ids: &[u32],
    questions: &[NestedQuestion<'_>],
) -> Result<NestedRun<E::State>, Qwen35Error> {
    if root_ids.is_empty() {
        return Err(Qwen35Error::InvalidInput(
            "sequential nested execution requires a non-empty root".into(),
        ));
    }
    if questions.is_empty() {
        return Err(Qwen35Error::InvalidInput(
            "sequential nested execution requires at least one question".into(),
        ));
    }

    let (root_feature, root_state) = executor.prefill(root_ids)?;
    let root_fingerprint = root_state.scheduling_fingerprint();
    let root_position = root_state.position();

    let mut results = Vec::with_capacity(questions.len());
    for (question_index, question) in questions.iter().enumerate() {
        if question.question_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {question_index} has an empty token suffix"
            )));
        }
        if question.candidate_suffix_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {question_index} has no candidate suffixes"
            )));
        }

        let question_branch = root_state.fork_one()?;
        let (question_feature, question_state) =
            executor.continue_from(&question_branch, question.question_ids)?;
        let expected_position = root_position + question.question_ids.len();
        if question_state.position() != expected_position {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {question_index} advanced to position {}, expected {expected_position}",
                question_state.position()
            )));
        }
        let question_fingerprint = question_state.scheduling_fingerprint();

        let mut candidates = Vec::with_capacity(question.candidate_suffix_ids.len());
        for (candidate_index, &suffix_ids) in question.candidate_suffix_ids.iter().enumerate() {
            if suffix_ids.is_empty() {
                return Err(Qwen35Error::InvalidInput(format!(
                    "question {question_index} candidate {candidate_index} has an empty suffix"
                )));
            }
            let candidate_branch = question_state.fork_one()?;
            let (feature, state) = executor.continue_from(&candidate_branch, suffix_ids)?;
            let expected_position = question_state.position() + suffix_ids.len();
            if state.position() != expected_position {
                return Err(Qwen35Error::InvalidInput(format!(
                    "question {question_index} candidate {candidate_index} advanced to position {}, expected {expected_position}",
                    state.position()
                )));
            }
            if question_state.scheduling_fingerprint() != question_fingerprint {
                return Err(Qwen35Error::InvalidInput(format!(
                    "question {question_index} state changed while executing candidate \
                     {candidate_index}"
                )));
            }
            candidates.push(NestedCandidateResult { feature, state });
        }

        if root_state.scheduling_fingerprint() != root_fingerprint {
            return Err(Qwen35Error::InvalidInput(format!(
                "shared root changed while executing question {question_index}"
            )));
        }
        results.push(NestedQuestionResult {
            question_feature,
            question_state,
            candidates,
        });
    }

    Ok(NestedRun {
        root_feature,
        root_state,
        questions: results,
    })
}

impl<E: SequentialNestedExecutor> SequentialNestedExecutor for &E {
    type State = E::State;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        (*self).prefill(input_ids)
    }

    fn continue_from(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        (*self).continue_from(state, suffix_ids)
    }

    fn continue_batch_from(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        (*self).continue_batch_from(states, suffix_ids)
    }

    fn continue_batch_from_controlled(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
        control: &ExecutionControl,
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        (*self).continue_batch_from_controlled(states, suffix_ids, control)
    }
}

impl Qwen35Backbone {
    /// Sequential nested execution against the pinned Qwen3.5 backbone.
    ///
    /// Prefills the shared root once, then evaluates every question and
    /// candidate through immutable forks of the `BranchableState` continuation.
    /// This is the Phase 3.5 native correctness baseline for later batched
    /// question and candidate execution.
    ///
    /// # Errors
    /// Returns [`Qwen35Error`] for empty plans, identity mismatches, and
    /// backbone execution failures.
    pub fn evaluate_nested(
        &self,
        root_ids: &[u32],
        questions: &[NestedQuestion<'_>],
    ) -> Result<NestedRun<BackboneState>, Qwen35Error> {
        run_sequential_nested(self, root_ids, questions)
    }
}

impl SequentialNestedExecutor for Qwen35Backbone {
    type State = BackboneState;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        if input_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(
                "nested prefill requires a non-empty root sequence".into(),
            ));
        }
        let (output, state) = Qwen35Backbone::prefill(self, input_ids)?;
        Ok((output.final_token().to_vec(), state))
    }

    fn prefill_controlled(
        &self,
        input_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        if input_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(
                "nested prefill requires a non-empty root sequence".into(),
            ));
        }
        let (output, state) = self.prefill_controlled(input_ids, control)?;
        Ok((output.final_token().to_vec(), state))
    }

    fn continue_from(
        &self,
        state: &BackboneState,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        if suffix_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(
                "nested continuation requires a non-empty suffix".into(),
            ));
        }
        let (output, next_state) = Qwen35Backbone::continue_from(self, state, suffix_ids)?;
        Ok((output.final_token().to_vec(), next_state))
    }

    fn continue_from_controlled(
        &self,
        state: &BackboneState,
        suffix_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        if suffix_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(
                "nested continuation requires a non-empty suffix".into(),
            ));
        }
        let (output, next_state) = self.continue_from_controlled(state, suffix_ids, control)?;
        Ok((output.final_token().to_vec(), next_state))
    }
}
