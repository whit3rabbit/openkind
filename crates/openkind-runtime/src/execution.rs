//! Backend-neutral execution capabilities and planner vocabulary.

/// Execution plan selected for one decision request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionPlan {
    /// Evaluate every complete candidate sequence independently.
    RepeatedFull,
    /// Share one root while advancing question and candidate branches one lane at a time.
    NestedSequential,
    /// Share one root and submit compatible question and candidate lanes to batch-forward kernels.
    NestedBatched,
}

impl ExecutionPlan {
    /// Stable lowercase identifier used in reports and persisted measurements.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RepeatedFull => "repeated_full",
            Self::NestedSequential => "nested_sequential",
            Self::NestedBatched => "nested_batched",
        }
    }

    /// All plans in stable reporting order.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [
            Self::RepeatedFull,
            Self::NestedSequential,
            Self::NestedBatched,
        ]
    }

    /// Physical compute mode this plan executes in under `capabilities`.
    ///
    /// `NestedBatched` is vectorized only when every question and candidate
    /// fan-out has at least two lanes and the request fits the advertised lane
    /// ceilings; otherwise the complete nested plan is classified per-lane.
    /// The two single-lane plans are per-lane by definition.
    #[must_use]
    pub const fn batch_forward_mode(
        self,
        capabilities: BackendCapabilities,
        questions: usize,
        question_lanes_share_start_position: bool,
        min_candidates: usize,
        max_candidates: usize,
    ) -> BatchForwardMode {
        match self {
            Self::RepeatedFull | Self::NestedSequential => BatchForwardMode::PerLane,
            Self::NestedBatched => {
                if capabilities.supports_vectorized_nested_forward()
                    && questions >= 2
                    && questions <= capabilities.max_vectorized_question_lanes()
                    && question_lanes_share_start_position
                    && min_candidates >= 2
                    && max_candidates >= min_candidates
                    && max_candidates <= capabilities.max_vectorized_candidate_lanes()
                {
                    BatchForwardMode::Vectorized
                } else {
                    BatchForwardMode::PerLane
                }
            }
        }
    }
}

/// Physical compute mode a selected [`ExecutionPlan`] actually executed in.
///
/// A plan name describes state topology — how branches share the immutable
/// root — not how the model forward advanced lanes. The same plan legitimately
/// executes per-lane on a backend without vectorized kernels and vectorized on
/// one that has them, so telemetry and evidence artifacts must record the plan
/// and this mode as siblings: an accelerated backend can otherwise appear to
/// test the same execution graph as the CPU reference when it does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchForwardMode {
    /// At least one lane group used separate model calls, or no group had
    /// enough lanes for a multi-lane forward. A nested-plan result can include
    /// vectorized calls for other groups.
    PerLane,
    /// This batch used one multi-lane model forward. For a nested-plan result,
    /// every question and candidate fan-out used such a forward.
    Vectorized,
}

impl BatchForwardMode {
    /// Stable lowercase identifier used in reports and evidence artifacts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PerLane => "per_lane",
            Self::Vectorized => "vectorized",
        }
    }
}

/// Model-forward capabilities advertised by one concrete backend.
///
/// State fan-out alone does not imply compute batching. The two flags are
/// separate because a backend may vectorize questions but still execute
/// candidate suffixes lane by lane, or vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BackendCapabilities {
    vectorized_question_forward: bool,
    vectorized_candidate_forward: bool,
    max_vectorized_question_lanes: usize,
    max_vectorized_candidate_lanes: usize,
}

impl BackendCapabilities {
    /// Construct an explicit capability declaration.
    #[must_use]
    pub const fn new(
        vectorized_question_forward: bool,
        vectorized_candidate_forward: bool,
    ) -> Self {
        Self {
            vectorized_question_forward,
            vectorized_candidate_forward,
            max_vectorized_question_lanes: if vectorized_question_forward {
                usize::MAX
            } else {
                1
            },
            max_vectorized_candidate_lanes: if vectorized_candidate_forward {
                usize::MAX
            } else {
                1
            },
        }
    }

    /// Capability declaration for the current per-lane CPU reference path.
    #[must_use]
    pub const fn per_lane() -> Self {
        Self::new(false, false)
    }

    /// Capability declaration for a backend that consumes both lane levels in one model call.
    #[must_use]
    pub const fn fully_vectorized() -> Self {
        Self::new(true, true)
    }

    /// Set explicit vectorized lane ceilings advertised by the backend.
    #[must_use]
    pub const fn with_lane_limits(
        mut self,
        max_question_lanes: usize,
        max_candidate_lanes: usize,
    ) -> Self {
        self.max_vectorized_question_lanes = max_question_lanes;
        self.max_vectorized_candidate_lanes = max_candidate_lanes;
        self
    }

    /// Whether question suffix lanes are consumed by one vectorized model forward.
    #[must_use]
    pub const fn supports_vectorized_question_forward(self) -> bool {
        self.vectorized_question_forward
    }

    /// Whether candidate suffix lanes are consumed by one vectorized model forward.
    #[must_use]
    pub const fn supports_vectorized_candidate_forward(self) -> bool {
        self.vectorized_candidate_forward
    }

    /// Whether the complete nested batched plan has real compute batching at both levels.
    #[must_use]
    pub const fn supports_vectorized_nested_forward(self) -> bool {
        self.vectorized_question_forward && self.vectorized_candidate_forward
    }

    /// Maximum question lanes accepted by one vectorized forward.
    #[must_use]
    pub const fn max_vectorized_question_lanes(self) -> usize {
        self.max_vectorized_question_lanes
    }

    /// Maximum candidate lanes accepted by one vectorized forward.
    #[must_use]
    pub const fn max_vectorized_candidate_lanes(self) -> usize {
        self.max_vectorized_candidate_lanes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_lane_and_vectorized_capabilities_are_distinct() {
        assert!(!BackendCapabilities::per_lane().supports_vectorized_nested_forward());
        assert!(BackendCapabilities::fully_vectorized().supports_vectorized_nested_forward());
        let limited = BackendCapabilities::fully_vectorized().with_lane_limits(8, 64);
        assert_eq!(limited.max_vectorized_question_lanes(), 8);
        assert_eq!(limited.max_vectorized_candidate_lanes(), 64);
        assert_eq!(ExecutionPlan::NestedBatched.as_str(), "nested_batched");
    }

    #[test]
    fn batch_forward_mode_separates_plan_topology_from_physical_compute() {
        let per_lane = BackendCapabilities::per_lane();
        for plan in ExecutionPlan::all() {
            assert_eq!(
                plan.batch_forward_mode(per_lane, 2, true, 2, 16),
                BatchForwardMode::PerLane,
                "per-lane backends advance every plan one lane at a time"
            );
        }

        let vectorized = BackendCapabilities::fully_vectorized();
        assert_eq!(
            ExecutionPlan::RepeatedFull.batch_forward_mode(vectorized, 2, true, 2, 16),
            BatchForwardMode::PerLane
        );
        assert_eq!(
            ExecutionPlan::NestedSequential.batch_forward_mode(vectorized, 2, true, 2, 16),
            BatchForwardMode::PerLane
        );
        assert_eq!(
            ExecutionPlan::NestedBatched.batch_forward_mode(vectorized, 2, true, 2, 16),
            BatchForwardMode::Vectorized
        );

        // Advertised kernels but a request shape beyond the lane ceilings still
        // executes per-lane.
        let limited = vectorized.with_lane_limits(8, 64);
        assert_eq!(
            ExecutionPlan::NestedBatched.batch_forward_mode(limited, 16, true, 2, 16),
            BatchForwardMode::PerLane
        );
        assert_eq!(
            ExecutionPlan::NestedBatched.batch_forward_mode(limited, 8, true, 2, 128),
            BatchForwardMode::PerLane
        );
        assert_eq!(
            ExecutionPlan::NestedBatched.batch_forward_mode(limited, 8, true, 2, 64),
            BatchForwardMode::Vectorized
        );
        assert_eq!(
            ExecutionPlan::NestedBatched.batch_forward_mode(vectorized, 1, true, 2, 2),
            BatchForwardMode::PerLane
        );
        assert_eq!(
            ExecutionPlan::NestedBatched.batch_forward_mode(vectorized, 2, true, 1, 2),
            BatchForwardMode::PerLane
        );
        assert_eq!(
            ExecutionPlan::NestedBatched.batch_forward_mode(vectorized, 2, true, 1, 1),
            BatchForwardMode::PerLane
        );
        assert_eq!(
            ExecutionPlan::NestedBatched.batch_forward_mode(vectorized, 2, false, 2, 2),
            BatchForwardMode::PerLane
        );
        assert_eq!(BatchForwardMode::PerLane.as_str(), "per_lane");
        assert_eq!(BatchForwardMode::Vectorized.as_str(), "vectorized");
    }
}
