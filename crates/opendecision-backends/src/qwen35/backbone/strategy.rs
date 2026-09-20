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

use crate::branch::BranchableState;
use crate::qwen35::Qwen35Error;

use super::nested::{NestedQuestion, SequentialNestedExecutor};
use super::{run_batched_nested, run_sequential_nested};

/// Execution strategy for one decision request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionStrategy {
    /// One full-sequence evaluation per candidate; the correctness oracle.
    RepeatedFull,
    /// One prefill, then sequential question and candidate forks (Phase 3.5).
    NestedSequential,
    /// One prefill, breadth-first batched question and candidate lanes
    /// (Phases 3.6/3.7).
    NestedBatched,
}

impl ExecutionStrategy {
    /// Stable lowercase identifier used in reports and JSON.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RepeatedFull => "repeated_full",
            Self::NestedSequential => "nested_sequential",
            Self::NestedBatched => "nested_batched",
        }
    }

    /// Every strategy, in dispatch-table order.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [
            Self::RepeatedFull,
            Self::NestedSequential,
            Self::NestedBatched,
        ]
    }
}

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
}

/// Strategy output in one uniform shape for reporting and parity checks.
#[derive(Debug)]
pub struct StrategyOutput<S: BranchableState> {
    strategy: ExecutionStrategy,
    question_features: Vec<Vec<Vec<f32>>>,
    question_states: Vec<Vec<S>>,
    forward_calls: usize,
    staged_tokens: usize,
    retained_state_bytes: usize,
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
    pub const fn retained_state_bytes(&self) -> usize {
        self.retained_state_bytes
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
            let mut retained = run.root_state().storage_bytes();
            let mut question_features = Vec::with_capacity(run.questions().len());
            let mut question_states = Vec::with_capacity(run.questions().len());
            for question in run.questions() {
                retained += question.question_state().storage_bytes();
                let mut features = Vec::with_capacity(question.candidates().len());
                let mut states = Vec::with_capacity(question.candidates().len());
                for candidate in question.candidates() {
                    retained += candidate.state().storage_bytes();
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
                staged_tokens,
                retained_state_bytes: retained,
            }
        }
        ExecutionStrategy::NestedBatched => {
            let run = run_batched_nested(&counting, root_ids, plans)?;
            let mut retained = run.root_state().storage_bytes();
            let mut question_features = Vec::with_capacity(run.questions().len());
            let mut question_states = Vec::with_capacity(run.questions().len());
            for question in run.questions() {
                retained += question.question_state().storage_bytes();
                let mut features = Vec::with_capacity(question.candidates().len());
                let mut states = Vec::with_capacity(question.candidates().len());
                for candidate in question.candidates() {
                    retained += candidate.state().storage_bytes();
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
                staged_tokens,
                retained_state_bytes: retained,
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
    let mut retained_state_bytes = 0_usize;
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
            retained_state_bytes += state.storage_bytes();
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
        staged_tokens,
        retained_state_bytes,
    })
}

/// Per-strategy cost estimates for one request, in the scheduler's analytic
/// token model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrategyEstimates {
    /// Tokens [`ExecutionStrategy::RepeatedFull`] processes: every full
    /// sequence for every candidate.
    pub repeated_tokens: usize,
    /// Tokens each shared strategy processes: one root prefill plus every
    /// question and candidate suffix.
    pub shared_tokens: usize,
    /// `repeated_tokens / shared_tokens`: the work multiplier sharing avoids.
    pub savings_ratio: f64,
}

/// Admission-relevant retained-state estimates, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetentionEstimates {
    /// [`ExecutionStrategy::RepeatedFull`]: every candidate state retained by
    /// the output.
    pub repeated_full_bytes: usize,
    /// [`ExecutionStrategy::NestedSequential`]: root plus every question and
    /// candidate state retained by the run.
    pub nested_sequential_bytes: usize,
    /// [`ExecutionStrategy::NestedBatched`]: root plus the question fan-out
    /// lanes plus every candidate state retained by the run.
    pub nested_batched_bytes: usize,
}

/// Selection decision with its measured-model rationale.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyDecision {
    /// Selected strategy.
    pub strategy: ExecutionStrategy,
    /// Human-readable reason referencing the measured model.
    pub rationale: String,
    /// Cost estimates behind the decision.
    pub estimates: StrategyEstimates,
    /// Retention estimates behind the decision.
    pub retention: RetentionEstimates,
}

/// Measured scheduler configuration.
///
/// `min_shared_savings_ratio` is measured on the named M4 Max host by the
/// `qwen35_scheduler_bench` example; the byte-model constants are pinned
/// properties of the selected profile's hybrid state. A different host or
/// precision profile must re-measure before trusting the crossover value.
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerConfig {
    /// Share state only when the repeated-token work exceeds the shared work
    /// by at least this ratio (measured crossover boundary).
    pub min_shared_savings_ratio: f64,
    /// Optional admission ceiling for retained branch-state bytes.
    pub max_state_bytes: Option<usize>,
    /// Position-independent bytes of one continuation state (recurrent plus
    /// convolution tensors).
    pub state_fixed_bytes: usize,
    /// Attention-KV bytes added per position token.
    pub state_bytes_per_token: usize,
}

impl SchedulerConfig {
    /// Measured crossover boundary for the named M4 Max host: every measured
    /// workload with a token-work ratio at or above 2.52 favored sharing by
    /// 1.25x-1.98x, so 2.0 selects sharing everywhere inside the measured
    /// envelope and falls back to the correctness oracle below it rather
    /// than extrapolate. See the ROADMAP 3.8 checkpoint.
    pub const MEASURED_MIN_SHARED_SAVINGS_RATIO: f64 = 2.0;

    /// Configuration for the pinned Qwen3.5 profile: the hybrid state's
    /// 24 DeltaNet recurrent tensors and 24 convolution tensors give
    /// `53,477,376` position-independent bytes per state, and 8 full-attention
    /// layers add `65,536` KV bytes per position token. The crossover ratio
    /// comes from the named-Mac measurement recorded in the ROADMAP 3.8
    /// checkpoint.
    #[must_use]
    pub const fn for_pinned_profile(
        min_shared_savings_ratio: f64,
        max_state_bytes: Option<usize>,
    ) -> Self {
        Self {
            min_shared_savings_ratio,
            max_state_bytes,
            state_fixed_bytes: 53_477_376,
            state_bytes_per_token: 65_536,
        }
    }
}

/// Request shape the scheduler plans for, in tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyRequest {
    /// Rendered state-root token count.
    pub root_tokens: usize,
    /// Per-question token counts.
    pub question_tokens: Vec<usize>,
    /// Per-question per-candidate suffix token counts.
    pub suffix_tokens: Vec<Vec<usize>>,
}

impl StrategyRequest {
    /// Token-count view of a concrete request.
    #[must_use]
    pub fn from_plans(root_tokens: usize, plans: &[NestedQuestion<'_>]) -> Self {
        Self {
            root_tokens,
            question_tokens: plans.iter().map(|plan| plan.question_ids.len()).collect(),
            suffix_tokens: plans
                .iter()
                .map(|plan| {
                    plan.candidate_suffix_ids
                        .iter()
                        .map(|ids| ids.len())
                        .collect()
                })
                .collect(),
        }
    }

    fn state_bytes(&self, config: &SchedulerConfig, position: usize) -> usize {
        config
            .state_fixed_bytes
            .saturating_add(config.state_bytes_per_token.saturating_mul(position))
    }
}

/// Estimate costs and select a strategy for `request`.
///
/// Policy order: (1) prefer repeated-full below the measured crossover when
/// its aggregate retained state fits; (2) prefer the breadth-first batched
/// path when its retained-state estimate fits the ceiling; (3) fall back to
/// the sequential shared path; (4) otherwise use repeated-full if it fits.
/// [`run_with_scheduler`] rejects execution when no strategy fits.
#[must_use]
pub fn choose_strategy(config: &SchedulerConfig, request: &StrategyRequest) -> StrategyDecision {
    let shared_tokens = request.root_tokens
        + request.question_tokens.iter().sum::<usize>()
        + request
            .suffix_tokens
            .iter()
            .map(|suffixes| suffixes.iter().sum::<usize>())
            .sum::<usize>();
    let repeated_tokens: usize = request
        .question_tokens
        .iter()
        .zip(&request.suffix_tokens)
        .map(|(&question, suffixes)| {
            suffixes
                .iter()
                .map(|&suffix| request.root_tokens + question + suffix)
                .sum::<usize>()
        })
        .sum();
    let savings_ratio = if shared_tokens == 0 {
        0.0
    } else {
        repeated_tokens as f64 / shared_tokens as f64
    };
    let estimates = StrategyEstimates {
        repeated_tokens,
        shared_tokens,
        savings_ratio,
    };

    let questions = request.question_tokens.len();
    let root_state_bytes = request.state_bytes(config, request.root_tokens);
    let mut sequential_bytes = root_state_bytes;
    let mut batched_question_bytes = 0_usize;
    for (&question_tokens, suffixes) in request.question_tokens.iter().zip(&request.suffix_tokens) {
        let question_position = request.root_tokens + question_tokens;
        let question_state_bytes = request.state_bytes(config, question_position);
        sequential_bytes += question_state_bytes;
        batched_question_bytes += question_state_bytes;
        for &suffix in suffixes {
            let candidate_bytes = request.state_bytes(config, question_position + suffix);
            sequential_bytes += candidate_bytes;
            batched_question_bytes += candidate_bytes;
        }
    }
    let batched_bytes = root_state_bytes * (1 + questions) + batched_question_bytes;
    let repeated_full_bytes = request
        .question_tokens
        .iter()
        .zip(&request.suffix_tokens)
        .flat_map(|(&question, suffixes)| {
            suffixes.iter().map(move |&suffix| {
                request.state_bytes(config, request.root_tokens + question + suffix)
            })
        })
        .fold(0_usize, usize::saturating_add);
    let retention = RetentionEstimates {
        repeated_full_bytes,
        nested_sequential_bytes: sequential_bytes,
        nested_batched_bytes: batched_bytes,
    };

    let fits = |bytes: usize| match config.max_state_bytes {
        Some(ceiling) => bytes <= ceiling,
        None => true,
    };

    if savings_ratio < config.min_shared_savings_ratio && fits(repeated_full_bytes) {
        return StrategyDecision {
            strategy: ExecutionStrategy::RepeatedFull,
            rationale: format!(
                "measured crossover: shared work would save less than the measured \
                 minimum ratio ({shared_tokens} shared vs {repeated_tokens} repeated tokens)"
            ),
            estimates,
            retention,
        };
    }
    if fits(batched_bytes) {
        return StrategyDecision {
            strategy: ExecutionStrategy::NestedBatched,
            rationale: format!(
                "sharing saves {savings_ratio}x repeated work and the batched lane \
                 peak ({batched_bytes} bytes) fits the state ceiling"
            ),
            estimates,
            retention,
        };
    }
    if fits(sequential_bytes) {
        return StrategyDecision {
            strategy: ExecutionStrategy::NestedSequential,
            rationale: format!(
                "sharing saves {savings_ratio}x repeated work; batched peak \
                 ({batched_bytes} bytes) exceeds the ceiling, sequential \
                 ({sequential_bytes} bytes) fits"
            ),
            estimates,
            retention,
        };
    }
    let (rationale, strategy) = if fits(repeated_full_bytes) {
        (
            format!(
                "retained shared state ({sequential_bytes} bytes) exceeds the ceiling; \
                 falling back to full-sequence execution ({repeated_full_bytes} bytes)"
            ),
            ExecutionStrategy::RepeatedFull,
        )
    } else {
        (
            format!(
                "no strategy fits the state ceiling; the repeated-full fallback requires \
                 {repeated_full_bytes} bytes"
            ),
            ExecutionStrategy::RepeatedFull,
        )
    };
    StrategyDecision {
        strategy,
        rationale,
        estimates,
        retention,
    }
}

/// Choose a strategy and execute the request under it.
///
/// # Errors
/// Returns [`Qwen35Error`] when no strategy fits the configured state-memory
/// ceiling or from the underlying strategy runner.
pub fn run_with_scheduler<E: SequentialNestedExecutor>(
    executor: &E,
    config: &SchedulerConfig,
    root_ids: &[u32],
    plans: &[NestedQuestion<'_>],
) -> Result<(StrategyDecision, StrategyOutput<E::State>), Qwen35Error>
where
    E::State: Clone,
{
    let request = StrategyRequest::from_plans(root_ids.len(), plans);
    let decision = choose_strategy(config, &request);
    let selected_bytes = match decision.strategy {
        ExecutionStrategy::RepeatedFull => decision.retention.repeated_full_bytes,
        ExecutionStrategy::NestedSequential => decision.retention.nested_sequential_bytes,
        ExecutionStrategy::NestedBatched => decision.retention.nested_batched_bytes,
    };
    if config
        .max_state_bytes
        .is_some_and(|ceiling| selected_bytes > ceiling)
    {
        return Err(Qwen35Error::InvalidInput(format!(
            "no execution strategy fits the state-memory ceiling: selected estimate \
             {selected_bytes} bytes"
        )));
    }
    let output = run_strategy(executor, decision.strategy, root_ids, plans)?;
    Ok((decision, output))
}
