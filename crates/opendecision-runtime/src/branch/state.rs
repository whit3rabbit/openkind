//! Backend-neutral branchable continuation-state traits.

use super::{ProfileId, SchedulingFingerprint, StateError};

/// A complete, branchable model continuation state.
pub trait BranchableState: Send + Sync {
    /// Batch handle produced by [`BranchableState::fork_batch`].
    type Batch: BranchBatch<State = Self>;

    /// Profile identity required before states can be combined.
    fn profile_id(&self) -> &ProfileId;
    /// Next absolute token position for a continuation suffix.
    fn position(&self) -> usize;
    /// Exact continuation tensor payload bytes, excluding process overhead.
    fn tensor_storage_bytes(&self) -> usize;
    /// Cheap scheduling fingerprint. This is never a persistent content key.
    fn scheduling_fingerprint(&self) -> SchedulingFingerprint;
    /// Fork into one independently mutable child while leaving the source unchanged.
    fn fork_one(&self) -> Result<Self, StateError>
    where
        Self: Sized;
    /// Fork into independently mutable lanes while leaving the source unchanged.
    fn fork_batch(&self, lanes: usize) -> Result<Self::Batch, StateError>
    where
        Self: Sized;
}

/// An ordered set of independent sibling states forked from one root.
pub trait BranchBatch {
    /// State type held by each lane.
    type State: BranchableState<Batch = Self>;
    /// Number of independent lanes.
    fn lanes(&self) -> usize;
    /// Exact tensor payload bytes held by all lanes combined.
    fn tensor_storage_bytes(&self) -> usize;
    /// Copy one lane without consuming the batch.
    fn select(&self, index: usize) -> Result<Self::State, StateError>;
    /// Reorder, subsample, or duplicate lanes into a new batch.
    fn gather(&self, indices: &[usize]) -> Result<Self, StateError>
    where
        Self: Sized;
}
