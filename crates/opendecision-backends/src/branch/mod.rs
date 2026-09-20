//! Backend-neutral branchable continuation-state contract.
//!
//! This module defines the Phase 3.4 contract that lifts backend-specific
//! continuation state into a shared branching vocabulary: profile-bound state
//! identity, stable fingerprints, exact storage accounting, immutable-root
//! fork, batched fan-out, and gather/select. It describes continuation state
//! in general; the selected Qwen3.5 hybrid profile binds attention KV,
//! DeltaNet recurrent state, and convolution state, and cloning or gathering
//! only attention KV would not isolate branches.
//!
//! The contract does not register a backend, map native state onto the Jev
//! wire format, or claim Metal execution.

mod error;
mod fingerprint;
mod identity;
mod state;

pub use error::StateError;
pub use fingerprint::StateFingerprint;
pub use identity::{ProfileId, StateIdentity, StateLineage, StorageBreakdown};
pub use state::{BranchBatch, BranchableState};

pub(crate) use fingerprint::FingerprintMaterial;
