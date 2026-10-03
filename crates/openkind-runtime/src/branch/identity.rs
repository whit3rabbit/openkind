//! Profile-bound state identity, fork lineage, and tensor accounting.

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
            "profile `{}` backbone `{}`@`{}` renderer `{}` tokenizer `{}` arithmetic `{}`",
            self.profile,
            self.backbone_id,
            self.backbone_revision,
            self.renderer_id,
            self.tokenizer_digest,
            self.arithmetic_id
        )
    }
}

static NEXT_ROOT_ID: AtomicU64 = AtomicU64::new(1);

/// Branch lineage shared by a prefill root and states forked from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateLineage {
    root_id: u64,
    fork_depth: u32,
}

impl StateLineage {
    /// Assign a fresh process-local root lineage.
    #[must_use]
    pub fn new_root() -> Self {
        Self {
            root_id: NEXT_ROOT_ID.fetch_add(1, Ordering::Relaxed),
            fork_depth: 0,
        }
    }
    /// Lineage of one further fork from this state.
    #[must_use]
    pub fn forked(&self) -> Self {
        Self {
            root_id: self.root_id,
            fork_depth: self.fork_depth + 1,
        }
    }
    /// Process-local identifier of the prefill root.
    #[must_use]
    pub const fn root_id(self) -> u64 {
        self.root_id
    }
    /// Number of fork operations between root and state.
    #[must_use]
    pub const fn fork_depth(self) -> u32 {
        self.fork_depth
    }
}

/// Exact tensor-payload accounting for one hybrid continuation state.
///
/// This deliberately excludes allocator overhead, `Vec`/`Arc` headers,
/// identity strings, Candle objects, mapped model weights, and forward
/// scratch. Admission adds those separately through a measured process-memory
/// envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorStorageBreakdown {
    /// Full-attention key/value tensor bytes.
    pub attention_kv_bytes: usize,
    /// DeltaNet recurrent-state tensor bytes.
    pub recurrent_bytes: usize,
    /// Causal-convolution-state tensor bytes.
    pub convolution_bytes: usize,
}

impl TensorStorageBreakdown {
    /// Exact total tensor payload bytes.
    #[must_use]
    pub const fn tensor_storage_bytes(&self) -> usize {
        self.attention_kv_bytes
            .saturating_add(self.recurrent_bytes)
            .saturating_add(self.convolution_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_storage_total_saturates_instead_of_wrapping() {
        let breakdown = TensorStorageBreakdown {
            attention_kv_bytes: usize::MAX,
            recurrent_bytes: 1,
            convolution_bytes: 1,
        };
        assert_eq!(breakdown.tensor_storage_bytes(), usize::MAX);
    }

    fn identity() -> StateIdentity {
        StateIdentity::new(
            "profile-a",
            "backbone",
            "rev-1",
            "renderer",
            "tokenizer-digest",
            "arithmetic",
        )
        .expect("complete identity")
    }

    #[test]
    fn state_identity_preserves_every_field() {
        let identity = identity();
        assert_eq!(identity.profile().as_str(), "profile-a");
        assert_eq!(identity.backbone_id(), "backbone");
        assert_eq!(identity.backbone_revision(), "rev-1");
        assert_eq!(identity.renderer_id(), "renderer");
        assert_eq!(identity.tokenizer_digest(), "tokenizer-digest");
        assert_eq!(identity.arithmetic_id(), "arithmetic");
        let display = identity.to_string();
        assert!(
            display.contains("profile `profile-a`") && display.contains("arithmetic `arithmetic`"),
            "display covers every field: {display}"
        );
    }

    #[test]
    fn state_identity_rejects_each_empty_field_by_name() {
        for (field, args) in [
            ("profile_id", ("", "b", "r", "e", "t", "a")),
            ("backbone_id", ("p", "", "r", "e", "t", "a")),
            ("backbone_revision", ("p", "b", "", "e", "t", "a")),
            ("renderer_id", ("p", "b", "r", "", "t", "a")),
            ("tokenizer_digest", ("p", "b", "r", "e", "", "a")),
            ("arithmetic_id", ("p", "b", "r", "e", "t", "")),
        ] {
            let error = StateIdentity::new(args.0, args.1, args.2, args.3, args.4, args.5)
                .expect_err("an empty field must fail");
            assert!(
                matches!(error, StateError::EmptyIdentityField { field: name } if name == field),
                "expected `{field}`: {error}"
            );
        }
    }

    #[test]
    fn state_lineage_tracks_root_and_depth_across_forks() {
        let root = StateLineage::new_root();
        assert_eq!(root.fork_depth(), 0);
        let fork = root.forked();
        assert_eq!(fork.root_id(), root.root_id(), "forks keep the root id");
        assert_eq!(fork.fork_depth(), 1);
        let deeper = fork.forked();
        assert_eq!(deeper.root_id(), root.root_id());
        assert_eq!(deeper.fork_depth(), 2);
        // Roots are process-unique.
        let other = StateLineage::new_root();
        assert_ne!(other.root_id(), root.root_id());
    }
}
