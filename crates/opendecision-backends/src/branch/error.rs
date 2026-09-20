//! Failures raised by the backend-neutral branch-state contract.

use thiserror::Error;

/// A branch-state identity, fork, batch, select, or gather operation failed.
///
/// Failures are reported instead of silently degrading isolation: a state that
/// cannot be proven branchable is never treated as branchable.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StateError {
    /// A state carried a different profile/model/tokenizer/renderer/arithmetic
    /// identity than the operation required.
    #[error("state identity mismatch: expected {expected}, found {actual}")]
    IdentityMismatch {
        /// Required identity.
        expected: String,
        /// Observed identity.
        actual: String,
    },

    /// A required identity field was empty.
    #[error("state identity field `{field}` cannot be empty")]
    EmptyIdentityField {
        /// Name of the rejected field.
        field: &'static str,
    },

    /// A lane index was outside the batch.
    #[error("lane index {index} is out of range for a {lanes}-lane branch batch")]
    LaneIndexOutOfBounds {
        /// Number of lanes held by the batch.
        lanes: usize,
        /// Rejected lane index.
        index: usize,
    },

    /// A batched fork requested fewer than one lane.
    #[error("batched fork requires at least one lane, requested {requested}")]
    InvalidLaneCount {
        /// Rejected lane count.
        requested: usize,
    },

    /// A gather listed no lane indices.
    #[error("gather requires at least one lane index")]
    EmptyGather,
}
