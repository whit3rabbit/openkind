//! Cost estimation, retention modeling, and scheduler selection policy.

use crate::qwen35::Qwen35Error;
use openkind_runtime::{BackendCapabilities, BatchForwardMode};

use super::super::nested::{NestedQuestion, SequentialNestedExecutor};
use super::{run_strategy, ExecutionStrategy, StrategyOutput};

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

/// Admission-relevant continuation-tensor and process-peak estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetentionEstimates {
    /// Tensor payload retained by [`ExecutionStrategy::RepeatedFull`].
    pub repeated_full_tensor_bytes: usize,
    /// Tensor payload retained by [`ExecutionStrategy::NestedSequential`].
    pub nested_sequential_tensor_bytes: usize,
    /// Tensor payload retained by [`ExecutionStrategy::NestedBatched`].
    pub nested_batched_tensor_bytes: usize,
    /// Estimated process peak for [`ExecutionStrategy::RepeatedFull`].
    pub repeated_full_process_peak_bytes: Option<usize>,
    /// Estimated process peak for [`ExecutionStrategy::NestedSequential`].
    pub nested_sequential_process_peak_bytes: Option<usize>,
    /// Estimated process peak for [`ExecutionStrategy::NestedBatched`].
    pub nested_batched_process_peak_bytes: Option<usize>,
}

/// Measured non-state memory used to turn tensor accounting into admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemoryEnvelope {
    /// Observed resident bytes before request execution.
    pub observed_resident_bytes: usize,
    /// Measured or conservatively budgeted forward scratch.
    pub forward_scratch_bytes: usize,
    /// Allocator fragmentation and bookkeeping headroom.
    pub allocator_headroom_bytes: usize,
    /// Hard process-memory ceiling.
    pub max_process_bytes: usize,
}

impl ProcessMemoryEnvelope {
    fn estimated_peak(self, tensor_bytes: usize) -> usize {
        self.observed_resident_bytes
            .saturating_add(self.forward_scratch_bytes)
            .saturating_add(self.allocator_headroom_bytes)
            .saturating_add(tensor_bytes)
    }

    /// Refresh the loaded-process baseline and divide remaining headroom across
    /// the maximum number of native requests that may execute concurrently.
    ///
    /// The resulting ceiling is deliberately per request. If every admitted
    /// request reaches its estimate at once, their combined incremental memory
    /// still fits the original process ceiling.
    #[must_use]
    pub fn for_concurrent_requests(
        mut self,
        observed_resident_bytes: usize,
        max_concurrent_requests: usize,
    ) -> Self {
        self.observed_resident_bytes = self.observed_resident_bytes.max(observed_resident_bytes);
        let remaining = self
            .max_process_bytes
            .saturating_sub(self.observed_resident_bytes);
        let per_request = remaining / max_concurrent_requests.max(1);
        self.max_process_bytes = self.observed_resident_bytes.saturating_add(per_request);
        self
    }
}

/// Selection decision with its measured-model rationale.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyDecision {
    /// Selected strategy.
    pub strategy: ExecutionStrategy,
    /// Physical compute mode the selected plan executes in. A plan name is
    /// state topology; this records how lanes actually advanced, so a
    /// `nested_batched` run on a per-lane CPU backend is not mistaken for a
    /// vectorized one on an accelerated backend.
    pub batch_forward_mode: BatchForwardMode,
    /// Whether the plan came from a diagnostic override instead of the
    /// measured policy.
    pub forced: bool,
    /// Human-readable reason referencing the measured model.
    pub rationale: String,
    /// Cost estimates behind the decision.
    pub estimates: StrategyEstimates,
    /// Retention estimates behind the decision.
    pub retention: RetentionEstimates,
    /// Whether the selected plan fits every configured admission ceiling.
    pub admitted: bool,
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
    /// Optional admission ceiling for continuation tensor payload bytes.
    pub max_tensor_storage_bytes: Option<usize>,
    /// Optional measured process-memory envelope used for peak admission.
    pub process_memory: Option<ProcessMemoryEnvelope>,
    /// Concrete backend compute capabilities. State lanes alone do not enable batching.
    pub backend_capabilities: BackendCapabilities,
    /// Position-independent bytes of one continuation state (recurrent plus
    /// convolution tensors).
    pub state_fixed_tensor_bytes: usize,
    /// Attention-KV bytes added per position token.
    pub state_tensor_bytes_per_token: usize,
    /// Diagnostic execution-plan override. When set, the measured savings
    /// ratio and the scheduler's vectorized-preference policy are bypassed,
    /// but admission (tensor and process-memory ceilings) is still enforced
    /// and the backend's real capabilities still bound the physical forward
    /// mode recorded in the decision.
    pub forced_strategy: Option<ExecutionStrategy>,
}

impl SchedulerConfig {
    /// Lowest token-work ratio actually measured on the named M4 Max host.
    /// Requests below this boundary are not treated as measured evidence.
    pub const LOWEST_MEASURED_SHARED_SAVINGS_RATIO: f64 = 2.52;

    /// Configuration for the pinned Qwen3.5 profile: the hybrid state's
    /// 24 DeltaNet recurrent tensors and 24 convolution tensors give
    /// `53,477,376` position-independent bytes per state, and 8 full-attention
    /// layers add `65,536` KV bytes per position token. The crossover ratio
    /// comes from the named-Mac measurement recorded in the ROADMAP 3.8
    /// checkpoint.
    #[must_use]
    pub const fn for_pinned_profile(
        min_shared_savings_ratio: f64,
        max_tensor_storage_bytes: Option<usize>,
    ) -> Self {
        Self {
            min_shared_savings_ratio,
            max_tensor_storage_bytes,
            process_memory: None,
            backend_capabilities: BackendCapabilities::per_lane(),
            state_fixed_tensor_bytes: 53_477_376,
            state_tensor_bytes_per_token: 65_536,
            forced_strategy: None,
        }
    }

    /// Override the concrete backend's compute capabilities.
    #[must_use]
    pub const fn with_backend_capabilities(mut self, capabilities: BackendCapabilities) -> Self {
        self.backend_capabilities = capabilities;
        self
    }

    /// Add measured process-memory admission inputs.
    #[must_use]
    pub const fn with_process_memory(mut self, process_memory: ProcessMemoryEnvelope) -> Self {
        self.process_memory = Some(process_memory);
        self
    }

    /// Force one execution plan for diagnostics and reproducibility.
    ///
    /// The override bypasses only the scheduler's profitability policy — the
    /// measured savings ratio and its preference for vectorized-capable
    /// backends. Admission ceilings still apply, and the decision records the
    /// physical [`BatchForwardMode`] the backend will actually use.
    #[must_use]
    pub const fn with_forced_strategy(mut self, plan: Option<ExecutionStrategy>) -> Self {
        self.forced_strategy = plan;
        self
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
            .state_fixed_tensor_bytes
            .saturating_add(config.state_tensor_bytes_per_token.saturating_mul(position))
    }
}

/// Estimate costs and select a strategy for `request`.
///
/// Policy order: (0) a configured [`SchedulerConfig::forced_strategy`]
/// override short-circuits the policy for diagnostics and reproducibility —
/// bypassing profitability only, never admission or the backend's real
/// capabilities; (1) share only when the measured crossover threshold says
/// sharing recovers its overhead; (2) prefer the breadth-first batched path
/// when its retained-state estimate fits the ceiling; (3) fall back to the
/// sequential shared path; (4) otherwise repeat full sequences.
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
    // The repeated-full executor returns every candidate state in its output,
    // so admission must account for all of those states rather than only the
    // largest one. Use saturating arithmetic so an unrepresentable aggregate
    // fails any finite ceiling instead of wrapping into an admissible value.
    let repeated_full_tensor_bytes = request
        .question_tokens
        .iter()
        .zip(&request.suffix_tokens)
        .fold(0_usize, |request_bytes, (&question, suffixes)| {
            suffixes.iter().fold(request_bytes, |bytes, &suffix| {
                bytes.saturating_add(
                    request.state_bytes(
                        config,
                        request
                            .root_tokens
                            .saturating_add(question)
                            .saturating_add(suffix),
                    ),
                )
            })
        });
    let process_peak = |tensor_bytes| {
        config
            .process_memory
            .map(|envelope| envelope.estimated_peak(tensor_bytes))
    };
    let retention = RetentionEstimates {
        repeated_full_tensor_bytes,
        nested_sequential_tensor_bytes: sequential_bytes,
        nested_batched_tensor_bytes: batched_bytes,
        repeated_full_process_peak_bytes: process_peak(repeated_full_tensor_bytes),
        nested_sequential_process_peak_bytes: process_peak(sequential_bytes),
        nested_batched_process_peak_bytes: process_peak(batched_bytes),
    };

    let fits = |tensor_bytes: usize| {
        let tensor_fits = config
            .max_tensor_storage_bytes
            .is_none_or(|ceiling| tensor_bytes <= ceiling);
        let process_fits = config.process_memory.is_none_or(|envelope| {
            envelope.estimated_peak(tensor_bytes) <= envelope.max_process_bytes
        });
        tensor_fits && process_fits
    };
    let max_candidates = request
        .suffix_tokens
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    let min_candidates = request
        .suffix_tokens
        .iter()
        .map(Vec::len)
        .min()
        .unwrap_or(0);
    // Every question lane forks the same immutable root, so the input states
    // share a position even when their suffix lengths differ. MLX right-pads
    // those suffixes and restores each lane's true position afterward.
    let question_lanes_share_start_position = true;
    let mode_for = |plan: ExecutionStrategy| {
        plan.batch_forward_mode(
            config.backend_capabilities,
            questions,
            question_lanes_share_start_position,
            min_candidates,
            max_candidates,
        )
    };

    if let Some(forced) = config.forced_strategy {
        // A diagnostic override bypasses profitability only: the measured
        // savings ratio and the vectorized-preference policy. Admission and
        // the backend's real capabilities still bound the decision, and the
        // physical forward mode is recorded so a per-lane `nested_batched`
        // run cannot be mistaken for a vectorized one.
        let tensor_bytes = match forced {
            ExecutionStrategy::RepeatedFull => repeated_full_tensor_bytes,
            ExecutionStrategy::NestedSequential => sequential_bytes,
            ExecutionStrategy::NestedBatched => batched_bytes,
        };
        let admitted = fits(tensor_bytes);
        let batch_forward_mode = mode_for(forced);
        return StrategyDecision {
            strategy: forced,
            batch_forward_mode,
            forced: true,
            rationale: format!(
                "forced override bypasses the measured savings-ratio and vectorized-preference \
                 policy only; retained tensor payload ({tensor_bytes} bytes) admission={admitted}; \
                 physical forward mode is {}",
                batch_forward_mode.as_str()
            ),
            estimates,
            retention,
            admitted,
        };
    }

    if savings_ratio < config.min_shared_savings_ratio {
        let admitted = fits(repeated_full_tensor_bytes);
        return StrategyDecision {
            strategy: ExecutionStrategy::RepeatedFull,
            batch_forward_mode: mode_for(ExecutionStrategy::RepeatedFull),
            forced: false,
            rationale: format!(
                "measured crossover: shared work would save less than the measured \
                 minimum ratio ({shared_tokens} shared vs {repeated_tokens} repeated tokens); \
                 repeated-full admission={admitted}"
            ),
            estimates,
            retention,
            admitted,
        };
    }
    let vectorized_shape_supported = config
        .backend_capabilities
        .supports_vectorized_nested_forward()
        && questions >= 2
        && questions <= config.backend_capabilities.max_vectorized_question_lanes()
        && question_lanes_share_start_position
        && min_candidates >= 2
        && max_candidates <= config.backend_capabilities.max_vectorized_candidate_lanes();
    if vectorized_shape_supported && fits(batched_bytes) {
        return StrategyDecision {
            strategy: ExecutionStrategy::NestedBatched,
            batch_forward_mode: mode_for(ExecutionStrategy::NestedBatched),
            forced: false,
            rationale: format!(
                "sharing saves {savings_ratio}x repeated work, the backend has vectorized \
                 question/candidate forward kernels, and the batched tensor payload \
                 ({batched_bytes} bytes) fits admission"
            ),
            estimates,
            retention,
            admitted: true,
        };
    }
    if fits(sequential_bytes) {
        return StrategyDecision {
            strategy: ExecutionStrategy::NestedSequential,
            batch_forward_mode: mode_for(ExecutionStrategy::NestedSequential),
            forced: false,
            rationale: format!(
                "sharing saves {savings_ratio}x repeated work; the backend lacks complete \
                 vectorized nested forward support or the batched tensor payload \
                 ({batched_bytes} bytes) exceeds admission, while sequential \
                 ({sequential_bytes} bytes) fits"
            ),
            estimates,
            retention,
            admitted: true,
        };
    }
    let admitted = fits(repeated_full_tensor_bytes);
    StrategyDecision {
        strategy: ExecutionStrategy::RepeatedFull,
        batch_forward_mode: mode_for(ExecutionStrategy::RepeatedFull),
        forced: false,
        rationale: if admitted {
            format!(
                "retained shared tensor payload ({sequential_bytes} bytes) exceeds admission; \
                 falling back to repeated full-sequence execution"
            )
        } else {
            format!(
                "no execution plan fits admission: repeated-full tensor payload is \
                 {repeated_full_tensor_bytes} bytes and shared tensor payload starts at \
                 {sequential_bytes} bytes"
            )
        },
        estimates,
        retention,
        admitted,
    }
}

/// Choose a strategy and execute the request under it.
///
/// # Errors
/// Returns [`Qwen35Error`] from the underlying strategy runner.
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
    if !decision.admitted {
        return Err(Qwen35Error::InvalidInput(decision.rationale.clone()));
    }
    let output = run_strategy(executor, decision.strategy, root_ids, plans)?;
    if output.batch_forward_mode() != decision.batch_forward_mode {
        return Err(Qwen35Error::InvalidInput(format!(
            "scheduler selected {} batch forwarding but executor reported {}",
            decision.batch_forward_mode.as_str(),
            output.batch_forward_mode().as_str()
        )));
    }
    Ok((decision, output))
}
