//! Phase 3.8 execution-strategy layer: one decision request, three
//! parity-proven strategies, and a measured selection policy.
//!
//! A request (one rendered state root plus questions with candidate suffixes)
//! can execute three ways:
//!
//! * [`ExecutionStrategy::RepeatedFull`] — one full-sequence evaluation per
//!   candidate (the correctness oracle; lowest retained state, most work).
//! * [`ExecutionStrategy::NestedSequential`] — prefill once, fork/advance
//!   question then candidate (Phase 3.5).
//! * [`ExecutionStrategy::NestedBatched`] — prefill once, breadth-first
//!   `fork_batch` question lanes, then per-question candidate fan-outs
//!   (Phases 3.6/3.7; the vectorization-ready shape).
//!
//! All three produce identical candidate features; [`run_strategy`] dispatches
//! and accounts forward calls and staged tokens, and [`choose_strategy`] picks
//! a strategy from measured crossover thresholds and a state-memory ceiling.
//! The default thresholds are measured on the named M4 Max host (see the
//! `qwen35_scheduler_bench` example and the ROADMAP 3.8 checkpoint); a changed
//! host or precision profile must re-measure before trusting them.

use std::cell::Cell;

use crate::qwen35::{ExecutionControl, Qwen35Error};
use openkind_runtime::branch::BranchableState;
use openkind_runtime::BatchForwardMode;
pub use openkind_runtime::ExecutionPlan as ExecutionStrategy;

use super::nested::{BatchContinuation, NestedQuestion, SequentialNestedExecutor};
use super::{run_batched_nested, run_sequential_nested};

mod policy;

pub use policy::{
    choose_strategy, run_with_scheduler, ProcessMemoryEnvelope, RetentionEstimates,
    SchedulerConfig, StrategyDecision, StrategyEstimates, StrategyRequest,
};

/// Executor wrapper that counts forward invocations for cost accounting.
pub struct CountingExecutor<E> {
    inner: E,
    prefills: Cell<usize>,
    continues: Cell<usize>,
}

impl<E> CountingExecutor<E> {
    /// Wrap `inner` and start counting from zero.
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            prefills: Cell::new(0),
            continues: Cell::new(0),
        }
    }

    /// Total forward invocations seen so far.
    #[must_use]
    pub fn forward_calls(&self) -> usize {
        self.prefills.get() + self.continues.get()
    }
}

impl<E: SequentialNestedExecutor> SequentialNestedExecutor for CountingExecutor<E> {
    type State = E::State;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        self.prefills.set(self.prefills.get() + 1);
        self.inner.prefill(input_ids)
    }

    fn continue_from(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        self.continues.set(self.continues.get() + 1);
        self.inner.continue_from(state, suffix_ids)
    }

    fn continue_batch_from(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        let result = self.inner.continue_batch_from(states, suffix_ids)?;
        let (lanes, mode) = result.into_parts();
        let calls = match mode {
            BatchForwardMode::PerLane => lanes.len(),
            BatchForwardMode::Vectorized => 1,
        };
        self.continues
            .set(self.continues.get().saturating_add(calls));
        Ok(BatchContinuation::new(lanes, mode))
    }

    fn continue_batch_from_controlled(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
        control: &ExecutionControl,
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        let result = self
            .inner
            .continue_batch_from_controlled(states, suffix_ids, control)?;
        let (lanes, mode) = result.into_parts();
        let calls = match mode {
            BatchForwardMode::PerLane => lanes.len(),
            BatchForwardMode::Vectorized => 1,
        };
        self.continues
            .set(self.continues.get().saturating_add(calls));
        Ok(BatchContinuation::new(lanes, mode))
    }
}

/// Strategy output in one uniform shape for reporting and parity checks.
#[derive(Debug)]
pub struct StrategyOutput<S: BranchableState> {
    strategy: ExecutionStrategy,
    question_features: Vec<Vec<Vec<f32>>>,
    question_states: Vec<Vec<S>>,
    forward_calls: usize,
    batch_forward_mode: BatchForwardMode,
    staged_tokens: usize,
    retained_tensor_bytes: usize,
}

impl<S: BranchableState> StrategyOutput<S> {
    /// Strategy that produced this output.
    #[must_use]
    pub const fn strategy(&self) -> ExecutionStrategy {
        self.strategy
    }

    /// Candidate features per question, in plan order.
    #[must_use]
    pub fn question_features(&self) -> &[Vec<Vec<f32>>] {
        &self.question_features
    }

    /// Candidate continuation states per question, in plan order.
    #[must_use]
    pub fn question_states(&self) -> &[Vec<S>] {
        &self.question_states
    }

    /// Forward invocations (prefills plus continuations).
    #[must_use]
    pub const fn forward_calls(&self) -> usize {
        self.forward_calls
    }

    /// Physical batch-forward mode reported by the executor.
    #[must_use]
    pub const fn batch_forward_mode(&self) -> BatchForwardMode {
        self.batch_forward_mode
    }

    /// Non-root tokens executed. For [`ExecutionStrategy::RepeatedFull`] this
    /// counts every full sequence; for the shared strategies it excludes the
    /// single root prefill.
    #[must_use]
    pub const fn staged_tokens(&self) -> usize {
        self.staged_tokens
    }

    /// Continuation tensor bytes retained by the output's states (root-scale
    /// and question states included where the strategy produces them).
    #[must_use]
    pub const fn retained_tensor_bytes(&self) -> usize {
        self.retained_tensor_bytes
    }
}

/// Execute one request under an explicit strategy with call accounting.
///
/// # Errors
/// Returns [`Qwen35Error`] from the underlying strategy runner.
pub fn run_strategy<E: SequentialNestedExecutor>(
    executor: &E,
    strategy: ExecutionStrategy,
    root_ids: &[u32],
    plans: &[NestedQuestion<'_>],
) -> Result<StrategyOutput<E::State>, Qwen35Error>
where
    E::State: Clone,
{
    let counting = CountingExecutor::new(executor);
    let output = match strategy {
        ExecutionStrategy::RepeatedFull => repeated_full_output(&counting, root_ids, plans)?,
        ExecutionStrategy::NestedSequential => {
            let run = run_sequential_nested(&counting, root_ids, plans)?;
            let mut retained = run.root_state().tensor_storage_bytes();
            let mut question_features = Vec::with_capacity(run.questions().len());
            let mut question_states = Vec::with_capacity(run.questions().len());
            for question in run.questions() {
                retained += question.question_state().tensor_storage_bytes();
                let mut features = Vec::with_capacity(question.candidates().len());
                let mut states = Vec::with_capacity(question.candidates().len());
                for candidate in question.candidates() {
                    retained += candidate.state().tensor_storage_bytes();
                    features.push(candidate.feature().to_vec());
                    states.push(candidate.state().clone());
                }
                question_features.push(features);
                question_states.push(states);
            }
            let staged_tokens: usize = plans
                .iter()
                .map(|plan| {
                    plan.question_ids.len()
                        + plan
                            .candidate_suffix_ids
                            .iter()
                            .map(|s| s.len())
                            .sum::<usize>()
                })
                .sum();
            StrategyOutput {
                strategy,
                question_features,
                question_states,
                forward_calls: counting.forward_calls(),
                batch_forward_mode: BatchForwardMode::PerLane,
                staged_tokens,
                retained_tensor_bytes: retained,
            }
        }
        ExecutionStrategy::NestedBatched => {
            let run = run_batched_nested(&counting, root_ids, plans)?;
            let mut retained = run.root_state().tensor_storage_bytes();
            let mut question_features = Vec::with_capacity(run.questions().len());
            let mut question_states = Vec::with_capacity(run.questions().len());
            for question in run.questions() {
                retained += question.question_state().tensor_storage_bytes();
                let mut features = Vec::with_capacity(question.candidates().len());
                let mut states = Vec::with_capacity(question.candidates().len());
                for candidate in question.candidates() {
                    retained += candidate.state().tensor_storage_bytes();
                    features.push(candidate.feature().to_vec());
                    states.push(candidate.state().clone());
                }
                question_features.push(features);
                question_states.push(states);
            }
            let staged_tokens: usize = plans
                .iter()
                .map(|plan| {
                    plan.question_ids.len()
                        + plan
                            .candidate_suffix_ids
                            .iter()
                            .map(|s| s.len())
                            .sum::<usize>()
                })
                .sum();
            StrategyOutput {
                strategy,
                question_features,
                question_states,
                forward_calls: counting.forward_calls(),
                batch_forward_mode: run.batch_forward_mode(),
                staged_tokens,
                retained_tensor_bytes: retained,
            }
        }
    };
    Ok(output)
}

/// Execute one request with [`ExecutionStrategy::RepeatedFull`]: one
/// full-sequence evaluation per candidate.
///
/// This is the correctness oracle every sharing strategy must match.
///
/// # Errors
/// Returns [`Qwen35Error`] from the underlying executor.
pub fn run_repeated_full<E: SequentialNestedExecutor>(
    executor: &E,
    root_ids: &[u32],
    plans: &[NestedQuestion<'_>],
) -> Result<StrategyOutput<E::State>, Qwen35Error> {
    let counting = CountingExecutor::new(executor);
    repeated_full_output(&counting, root_ids, plans)
}

fn repeated_full_output<E: SequentialNestedExecutor>(
    executor: &CountingExecutor<E>,
    root_ids: &[u32],
    plans: &[NestedQuestion<'_>],
) -> Result<StrategyOutput<E::State>, Qwen35Error> {
    let mut question_features = Vec::with_capacity(plans.len());
    let mut question_states = Vec::with_capacity(plans.len());
    let mut staged_tokens = 0_usize;
    let mut retained_tensor_bytes = 0_usize;
    for (question_index, plan) in plans.iter().enumerate() {
        if plan.candidate_suffix_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(format!(
                "question {question_index} has no candidate suffixes"
            )));
        }
        let mut features = Vec::with_capacity(plan.candidate_suffix_ids.len());
        let mut states = Vec::with_capacity(plan.candidate_suffix_ids.len());
        for &suffix_ids in plan.candidate_suffix_ids.iter() {
            let mut full_ids = root_ids.to_vec();
            full_ids.extend_from_slice(plan.question_ids);
            full_ids.extend_from_slice(suffix_ids);
            let (feature, state) = executor.prefill(&full_ids)?;
            staged_tokens += full_ids.len();
            retained_tensor_bytes += state.tensor_storage_bytes();
            features.push(feature);
            states.push(state);
        }
        question_features.push(features);
        question_states.push(states);
    }
    Ok(StrategyOutput {
        strategy: ExecutionStrategy::RepeatedFull,
        question_features,
        question_states,
        forward_calls: executor.forward_calls(),
        batch_forward_mode: BatchForwardMode::PerLane,
        staged_tokens,
        retained_tensor_bytes,
    })
}
