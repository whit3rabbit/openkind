//! Phase 3M MLX/Metal **parity** backend for the selected Qwen3.5 profile.
//!
//! This module adds a second execution backend behind the same frozen Phase 3B
//! contract: MLX arrays and kernels (built from the pinned mlx-c release inside
//! `mlx-sys`) driven by the Rust Qwen3.5 layer semantics already proven on the
//! Candle CPU oracle. The production path uses the faster ordinary-ops
//! recurrence. Generic masked/vector-gate reduction-tree kernels and a packed
//! FP32 `Dk = Dv = 128` sequence kernel are available as qualified tuning
//! candidates, but benchmark evidence has not justified promoting them.
//!
//! The FP32 reference-ops path includes a vectorized continuation candidate
//! for same-position lane batches with right-padded suffixes. It remains
//! fail-closed at capability dispatch until model-backed parity and workload
//! gates qualify it. BF16 and FP32 MetalTree stay per-lane. Candle CPU remains
//! the correctness oracle; every comparison target is a Phase 3B saved vector,
//! never a live Candle model in this process.
//!
//! The whole module requires macOS arm64 and the `mlx` cargo feature.

use thiserror::Error;

/// Branchable state and batch execution abstractions for the MLX backend.
pub mod branch_state;
/// Sequential and scheduled execution drivers for the MLX backbone.
pub mod executor;
/// Neural network layer implementations (DeltaNet, attention, RMSNorm, MLP) in MLX.
pub mod layers;
/// MLX backbone model definitions and evaluation state containers.
pub mod model;
/// MLX device runtime, memory tracker, and toolchain qualification state.
pub mod runtime;
/// Weight loading, layout transformation, and safetensors mapping for MLX.
pub mod weights;

pub use branch_state::MlxBranchBatch;
pub use model::{
    MlxBackboneOutput, MlxBackboneState, MlxBackboneTrace, MlxOperationTrace, MlxQwen35Backbone,
};
pub use runtime::{
    MlxMemorySnapshot, MlxRuntime, MlxRuntimeConfig, SharedMlxRuntime,
    DEFAULT_INACTIVE_CACHE_LIMIT_BYTES, MLX_CORE_VERSION,
};
pub use weights::{MlxCheckpointFormat, MlxSurveyCheckpoint, MlxWeightLoadReport, MlxWeightStore};

/// Execution precision of the MLX backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxPrecision {
    /// Weights and activations in FP32 (BF16 widened exactly on load): the
    /// correctness-reference mode expected to pass the frozen gates.
    Fp32,
    /// Source-checkpoint BF16 weights and activations: the separately gated
    /// `mlx-native-bf16` candidate profile. The explicit runtime BF16
    /// preflight must pass before any model workload may run in this mode;
    /// the full Gate A example remains the promotion check for a toolchain.
    NativeBf16,
}

/// Gated DeltaNet recurrence implementation selected for an MLX runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxGatedDeltaKernel {
    /// Ordinary MLX array operations, retained as the independent fallback
    /// and differential comparator.
    ReferenceOps,
    /// Fused Metal reduction-tree kernels where model-backed qualification
    /// passes. FP32 dispatches the packed `Dk = Dv = 128` specialization.
    /// The generic BF16 kernel remains an unpromoted differential candidate,
    /// so native BF16 model loads currently fall back to `ReferenceOps`.
    MetalTree,
}

impl MlxGatedDeltaKernel {
    /// Stable kernel-family identifier used in execution identity.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceOps => "reference-ops",
            Self::MetalTree => "metal-tree-v1",
        }
    }
}

impl MlxPrecision {
    /// Stable identifier used in arithmetic identities and reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::NativeBf16 => "bf16",
        }
    }
}

/// Pinned `mlx-rs`/`mlx-sys` crate version.
///
/// `mlx-sys` 0.6.0 vendors and builds mlx-c release `v0.6.0-7-gc74db53`,
/// which corresponds to MLX core 0.32.2. The linked runtime is therefore
/// identified by this pin; formal evidence additionally records archive and
/// `mlx.metallib` hashes plus the Xcode/Metal toolchain identity.
pub const MLX_RS_VERSION: &str = "0.32.0";

/// mlx-c release vendored by `mlx-sys` 0.6.0, with its MLX core version.
pub const MLX_C_RELEASE: &str = "v0.6.0-7-gc74db53 (MLX 0.32.2)";

/// mlx-lm source commit used as the reference authority for Qwen3.5 recurrent
/// q/k normalization and offset-RMSNorm semantics.
///
/// No mlx-lm code executes in this backend; the commit is recorded because it
/// contains the upstream Qwen recurrent q/k normalization fix that earlier
/// mlx-lm releases (0.31.3) applied incorrectly.
pub const MLX_LM_REFERENCE_COMMIT: &str = "a63e24c389382619eb6d9af656e3b46024be217a";

/// Base arithmetic identity of the MLX FP32 reference-ops execution path.
///
/// Kernel family is part of the identity: opt-in fused Gated-DeltaNet
/// execution receives its own distinct value instead of silently sharing
/// this one. A loaded backbone appends the runtime's Xcode/Metal toolchain
/// identity before storing this in continuation state.
pub const MLX_ARITHMETIC_ID_FP32_REFERENCE: &str = "mlx-core-0.32.2/fp32/reference-ops";

/// Base arithmetic identity of the MLX native-BF16 reference-ops candidate profile.
pub const MLX_ARITHMETIC_ID_BF16_REFERENCE: &str = "mlx-core-0.32.2/bf16/reference-ops";

/// Base arithmetic identity of the FP32 packed Metal reduction-tree path.
pub const MLX_ARITHMETIC_ID_FP32_METAL_TREE: &str =
    "mlx-core-0.32.2/fp32/metal-tree-packed-dk128-v1";

/// Base arithmetic identity of the BF16 generic Metal reduction-tree path.
pub const MLX_ARITHMETIC_ID_BF16_METAL_TREE: &str = "mlx-core-0.32.2/bf16/metal-tree-generic-v1";

/// Errors raised by the MLX parity backend.
#[derive(Debug, Error)]
pub enum MlxError {
    /// An MLX operation failed.
    #[error("MLX operation `{operation}` failed: {message}")]
    Operation {
        /// Name of the failing operation.
        operation: &'static str,
        /// Exception text captured from the MLX runtime.
        message: String,
    },
    /// The linked MLX runtime does not match the pinned version.
    #[error("MLX runtime version mismatch: expected {expected}, found {found}")]
    VersionMismatch {
        /// Expected pinned version string.
        expected: &'static str,
        /// Version reported by the linked runtime.
        found: String,
    },
    /// A required tensor or shape was invalid.
    #[error("invalid MLX tensor state: {0}")]
    InvalidState(String),
}
