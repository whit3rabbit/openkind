//! Profile-bound state identity, fork lineage, and storage accounting.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use super::StateError;

/// Immutable profile identifier bound to every branch state.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProfileId(String);

impl ProfileId {
    /// Construct a profile identifier, rejecting empty values.
    pub fn new(id: impl Into<String>) -> Result<Self, StateError> {
        let id = id.into();
        if id.is_empty() {
            return Err(StateError::EmptyIdentityField {
                field: "profile_id",
            });
        }
        Ok(Self(id))
    }

    /// Profile identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProfileId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Complete execution identity carried by every continuation state.
///
/// The identity records the profile, model, tokenizer, renderer, and
/// arithmetic execution contract that produced the state. A state may only be
/// continued, forked, or batched together with states of an equal identity.
/// A changed device, precision, or kernel path is a new arithmetic identity
/// rather than an in-place upgrade.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateIdentity {
    profile: ProfileId,
    backbone_id: String,
    backbone_revision: String,
    renderer_id: String,
    tokenizer_digest: String,
    arithmetic_id: String,
}

impl StateIdentity {
    /// Construct a state identity, rejecting empty fields.
    pub fn new(
        profile: impl Into<String>,
        backbone_id: impl Into<String>,
        backbone_revision: impl Into<String>,
        renderer_id: impl Into<String>,
        tokenizer_digest: impl Into<String>,
        arithmetic_id: impl Into<String>,
    ) -> Result<Self, StateError> {
        let identity = Self {
            profile: ProfileId::new(profile)?,
            backbone_id: backbone_id.into(),
            backbone_revision: backbone_revision.into(),
            renderer_id: renderer_id.into(),
            tokenizer_digest: tokenizer_digest.into(),
            arithmetic_id: arithmetic_id.into(),
        };
        for (field, value) in [
            ("backbone_id", &identity.backbone_id),
            ("backbone_revision", &identity.backbone_revision),
            ("renderer_id", &identity.renderer_id),
            ("tokenizer_digest", &identity.tokenizer_digest),
            ("arithmetic_id", &identity.arithmetic_id),
        ] {
            if value.is_empty() {
                return Err(StateError::EmptyIdentityField { field });
            }
        }
        Ok(identity)
    }

    /// Immutable model/execution profile identifier.
    #[must_use]
    pub fn profile(&self) -> &ProfileId {
        &self.profile
    }

    /// Pinned base-model identifier.
    #[must_use]
    pub fn backbone_id(&self) -> &str {
        &self.backbone_id
    }

    /// Immutable base-model revision.
    #[must_use]
    pub fn backbone_revision(&self) -> &str {
        &self.backbone_revision
    }

    /// Renderer identity, including ordering semantics.
    #[must_use]
    pub fn renderer_id(&self) -> &str {
        &self.renderer_id
    }

    /// Digest of the pinned offline tokenizer artifact.
    #[must_use]
    pub fn tokenizer_digest(&self) -> &str {
        &self.tokenizer_digest
    }

    /// Arithmetic/device execution identity of the producing backend.
    #[must_use]
    pub fn arithmetic_id(&self) -> &str {
        &self.arithmetic_id
    }
}

impl fmt::Display for StateIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "profile `{profile}` backbone `{backbone}`@`{revision}` renderer `{renderer}` \
             tokenizer `{tokenizer}` arithmetic `{arithmetic}`",
            profile = self.profile,
            backbone = self.backbone_id,
            revision = self.backbone_revision,
            renderer = self.renderer_id,
            tokenizer = self.tokenizer_digest,
            arithmetic = self.arithmetic_id,
        )
    }
}

static NEXT_ROOT_ID: AtomicU64 = AtomicU64::new(1);

/// Branch lineage shared by a prefill root and every state forked from it.
///
/// The root ID is assigned once per prefill from a process-local counter and
/// is not stable across processes; the strict content fingerprint is the
/// cross-process identity. Continuation advances extend a state's own
/// timeline and do not change its lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateLineage {
    root_id: u64,
    fork_depth: u32,
}

impl StateLineage {
    /// Assign a fresh root lineage from the process-local counter.
    #[must_use]
    pub fn new_root() -> Self {
        Self {
            root_id: NEXT_ROOT_ID.fetch_add(1, Ordering::Relaxed),
            fork_depth: 0,
        }
    }

    /// Lineage of one further fork from this state; the source is unchanged.
    #[must_use]
    pub fn forked(&self) -> Self {
        Self {
            root_id: self.root_id,
            fork_depth: self.fork_depth + 1,
        }
    }

    /// Process-local identifier of the prefill root.
    #[must_use]
    pub fn root_id(self) -> u64 {
        self.root_id
    }

    /// Number of fork operations between the root and this state.
    #[must_use]
    pub fn fork_depth(self) -> u32 {
        self.fork_depth
    }
}

/// Exact byte accounting for one hybrid continuation state.
///
/// The field names are the three tensor families of the selected Qwen3.5
/// hybrid profile. `tensor_bytes` and [`StorageBreakdown::total_bytes`] are
/// exact sums; `metadata_bytes` reports the fixed-size identity, lineage, and
/// position bookkeeping and is excluded from the comparable tensor total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageBreakdown {
    /// Full-attention key/value tensor bytes.
    pub attention_kv_bytes: usize,
    /// DeltaNet recurrent state tensor bytes.
    pub recurrent_bytes: usize,
    /// Causal-convolution state tensor bytes.
    pub convolution_bytes: usize,
    /// Fixed-size identity, lineage, and position bookkeeping bytes.
    pub metadata_bytes: usize,
}

impl StorageBreakdown {
    /// Exact continuation tensor bytes, excluding metadata bookkeeping.
    #[must_use]
    pub fn tensor_bytes(&self) -> usize {
        self.attention_kv_bytes + self.recurrent_bytes + self.convolution_bytes
    }

    /// Exact total bytes including metadata bookkeeping.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.tensor_bytes() + self.metadata_bytes
    }
}
