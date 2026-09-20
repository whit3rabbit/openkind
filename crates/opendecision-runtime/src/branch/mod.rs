//! Backend-neutral branchable continuation-state contract.
//!
//! This runtime-owned module defines profile-bound state identity, typed
//! scheduling and content fingerprints, exact tensor-payload accounting,
//! immutable-root fork, batched fan-out, and gather/select. Concrete model
//! crates implement these contracts without making the runtime depend on a
//! backend.

mod cache;
mod error;
mod fingerprint;
mod identity;
mod state;

pub use cache::{BranchStateCache, CacheError, StateCacheKey};
pub use error::StateError;
pub use fingerprint::{
    ContentFingerprint, ContentFingerprintBuilder, SchedulingFingerprint,
    SchedulingFingerprintBuilder,
};
pub use identity::{ProfileId, StateIdentity, StateLineage, TensorStorageBreakdown};
pub use state::{BranchBatch, BranchableState};
