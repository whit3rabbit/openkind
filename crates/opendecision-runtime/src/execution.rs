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
}
