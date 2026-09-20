//! Backend-neutral branchable continuation-state traits.

use super::{ProfileId, StateError, StateFingerprint};

/// A complete, branchable model continuation state.
///
/// The abstraction describes continuation state, not attention KV
/// specifically. Implementations must bind every mutable component a
/// continuation suffix would read; for the selected Qwen3.5 profile that is
/// attention KV, DeltaNet recurrent state, and convolution state together
/// with the logical position and profile identity. Cloning or gathering only
/// attention KV does not isolate branches.
pub trait BranchableState: Send + Sync {
    /// Batch handle produced by [`BranchableState::fork_batch`].
    type Batch: BranchBatch<State = Self>;

    /// Profile identity that must match before a state is continued or
    /// combined with another state.
    fn profile_id(&self) -> &ProfileId;

    /// Next absolute token position for a continuation suffix.
    fn position(&self) -> usize;

    /// Exact continuation tensor bytes held by this state, excluding
    /// fixed-size metadata bookkeeping.
    fn storage_bytes(&self) -> usize;

    /// Cheap structural fingerprint, stable under `Clone` and unchanged by
    /// fork operations until the child advances.
    fn fingerprint(&self) -> StateFingerprint;

    /// Fork this state into one independent child. The source is unchanged,
    /// and the child owns every continuation tensor.
    ///
    /// # Errors
    /// Returns [`StateError`] when the implementation cannot prove complete
    /// isolation of the child.
    fn fork_one(&self) -> Result<Self, StateError>
    where
        Self: Sized;

    /// Fork this state into `lanes` independent children in one batch. The
    /// source root is unchanged, and every lane is independently mutable.
    ///
    /// # Errors
    /// Returns [`StateError::InvalidLaneCount`] for an empty batch, and
    /// [`StateError`] when the implementation cannot prove complete lane
    /// isolation.
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

    /// Exact continuation tensor bytes held by all lanes combined.
    fn storage_bytes(&self) -> usize;

    /// Copy lane `index` out of the batch without consuming the batch or the
    /// remaining lanes.
    ///
    /// # Errors
    /// Returns [`StateError::LaneIndexOutOfBounds`] when the index is outside
    /// the batch.
    fn select(&self, index: usize) -> Result<Self::State, StateError>;

    /// Reorder, subsample, or duplicate lanes into a new batch. Every lane in
    /// the returned batch preserves the position and fingerprint of its
    /// source lane.
    ///
    /// # Errors
    /// Returns [`StateError::EmptyGather`] for an empty index list and
    /// [`StateError::LaneIndexOutOfBounds`] for any out-of-range index.
    fn gather(&self, indices: &[usize]) -> Result<Self, StateError>
    where
        Self: Sized;
}
