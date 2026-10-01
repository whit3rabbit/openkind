//! Phase 3 parity implementation for the selected Qwen3.5 state-first profile.
//!
//! This module implements the exported tokenizer/renderer contract, verified
//! FP32 CPU backbone and continuation path, sequential nested execution over
//! the backend-neutral branch contract, feature-to-probability readout, and a
//! direct [`DecisionEngine`](openkind_engine::DecisionEngine) adapter for
//! explicit offline service registration. It does not expose a Metal path or
//! claim release promotion.

mod backbone;
mod engine;
mod evidence;
/// Offline scoring experiments over the pinned base checkpoint.
pub mod experimental;
mod head;
mod identity;
/// Phase 3M MLX/Metal parity backend (available on macOS arm64 with feature `mlx`).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx;
mod profile;
mod tokenizer;

pub use backbone::{
    choose_strategy, run_batched_candidates, run_batched_nested, run_batched_nested_pooled,
    run_batched_questions, run_flat_batched_candidates, run_repeated_full, run_sequential_nested,
    run_strategy, run_with_scheduler, BackboneOutput, BackboneReference, BackboneState,
    BatchContinuation, BatchedCandidateResult, BatchedCandidates, BatchedNestedRun,
    BatchedQuestionResult, BatchedQuestions, CountingExecutor, EmbeddingOutput, ExecutionStrategy,
    FlatBatchRun, FullSequenceRecord, Layer0Output, LayerKind, NestedCandidateResult,
    NestedQuestion, NestedQuestionResult, NestedRun, ProcessMemoryEnvelope, Qwen35Backbone,
    Qwen35BranchBatch, Qwen35Embedding, Qwen35Geometry, Qwen35Layer0, RetentionEstimates,
    SchedulerConfig, SequentialNestedExecutor, StageComparison, StrategyDecision,
    StrategyEstimates, StrategyOutput, StrategyRequest, TraceStage,
};
pub(crate) use backbone::{EmbeddingLayout, TextBackbone};
// Clef-family GGUF execution reuses the parity-verified forward kernels.
pub(crate) use backbone::{
    apply_rotary, causal_depthwise_conv_silu_with_state, causal_grouped_query_attention,
    gated_delta_recurrent_with_state, rms_norm_heads, rms_norm_zero_centered, split_query_gate,
};
pub use engine::{Qwen35Backend, Qwen35DecisionEngine, Qwen35EngineConfig, SEMANTIC_NONE_OPTION};
pub use evidence::{native_profile_record, BACKEND_IMPLEMENTATION};
pub use head::{HeadEvaluation, PolicyAction, PrimitiveKind, ScoreSummaryHead, FEATURE_WIDTH};
pub use identity::ExecutionIdentity;
pub use profile::{ReferenceBundle, DECLARED_PROBABILITY_SPACE};
pub use tokenizer::{
    CandidateText, Qwen35Tokenizer, StateFirstSegments, MAX_CANDIDATES, MAX_SEQUENCE_TOKENS,
    STATE_FIRST_RENDERER_ID, TOKENIZER_BACKEND_SHA256, TOKENIZER_JSON_SHA256,
};

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use openkind_engine::ProfileValidationError;
use openkind_runtime::branch::{StateError, StateIdentity};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Selected immutable model/execution profile ID.
pub const PROFILE_ID: &str = "a047d6802c3f06f085b8";
/// SHA-256 of the exported selected-model bundle archive.
pub const BUNDLE_SHA256: &str = "4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332";
/// Public repository containing the selected reference bundle.
pub const REFERENCE_REPOSITORY: &str = "cowWhySo/OpenKind-Qwen3.5-4B-StateFirst";
/// Immutable repository revision used by the vendored parity fixtures.
pub const REFERENCE_REVISION: &str = "20974648aa087369645494e898351253248627a0";
/// Pinned frozen base-model ID.
pub const BACKBONE_ID: &str = "Qwen/Qwen3.5-4B-Base";
/// Pinned frozen base-model revision.
pub const BACKBONE_REVISION: &str = "1001bb4d826a52d1f399e183466143f4da7b741b";
/// Selected fitted head artifact.
pub const HEAD_FILE: &str = "score_summary_seed17.safetensors";
/// Selected calibration temperature.
pub const CALIBRATION_TEMPERATURE: f64 = 1.818_679_991_044_277_7;
/// Selected application-policy threshold.
pub const POLICY_THRESHOLD: f64 = 0.98;
/// Exported full-distribution parity tolerance.
pub const PROBABILITY_TOLERANCE: f64 = 0.005;
/// Exported ordering/logit comparison tolerance.
pub const ORDERING_TOLERANCE: f64 = 0.000_01;
/// Arithmetic/device identity of the native CPU continuation execution path.
///
/// A changed device (for example Metal), precision, or kernel path is a new
/// arithmetic identity: states produced under a different value are rejected
/// instead of silently mixed.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32";

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub use mlx::{
    MLX_ARITHMETIC_ID_BF16_METAL_TREE, MLX_ARITHMETIC_ID_BF16_REFERENCE,
    MLX_ARITHMETIC_ID_FP32_METAL_TREE, MLX_ARITHMETIC_ID_FP32_REFERENCE, MLX_C_RELEASE,
    MLX_LM_REFERENCE_COMMIT, MLX_RS_VERSION,
};

const MANIFEST_SHA256: &str = "dd42289e525d82a1ab8d55efd3843970e6c31a23059512a2c7e4ee7ca6459f78";

/// Errors raised while loading or evaluating the selected Qwen3.5 profile.
#[derive(Debug, Error)]
pub enum Qwen35Error {
    /// A required artifact could not be read.
    #[error("failed to read `{path}`: {source}")]
    Io {
        /// Artifact path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },

    /// A JSON contract artifact could not be decoded.
    #[error("failed to decode JSON artifact `{path}`: {source}")]
    Json {
        /// Artifact path.
        path: PathBuf,
        /// Underlying JSON error.
        source: serde_json::Error,
    },

    /// The pinned tokenizer artifact could not be decoded or executed.
    #[error("tokenizer failure: {0}")]
    Tokenizer(String),

    /// A safetensors artifact was malformed.
    #[error("invalid safetensors artifact: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    /// Native tensor execution failed.
    #[error("native Qwen tensor execution failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// MLX parity-backend execution failed (feature `mlx`, macOS arm64).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    #[error("MLX execution failed: {0}")]
    Mlx(#[from] mlx::MlxError),

    /// The operating system could not provide process-memory evidence.
    #[error("failed to observe native process memory: {0}")]
    MemoryObservation(#[from] std::io::Error),

    /// Engine-level profile validation failed.
    #[error("invalid model execution profile: {0}")]
    Profile(#[from] ProfileValidationError),

    /// A pinned contract field did not match the selected profile.
    #[error("reference contract mismatch for `{field}`: expected `{expected}`, found `{actual}`")]
    ContractMismatch {
        /// Name of the mismatched field.
        field: &'static str,
        /// Required selected-profile value.
        expected: String,
        /// Observed artifact value.
        actual: String,
    },

    /// A required file was not covered by the bundle manifest.
    #[error("bundle manifest does not contain required artifact `{0}`")]
    MissingManifestEntry(String),

    /// An artifact did not match its pinned digest.
    #[error("SHA-256 mismatch for `{path}`: expected {expected}, found {actual}")]
    DigestMismatch {
        /// Bundle-relative artifact path.
        path: String,
        /// Required digest.
        expected: String,
        /// Observed digest.
        actual: String,
    },

    /// The head contained an unexpected tensor set.
    #[error("head tensor inventory mismatch: expected {expected:?}, found {actual:?}")]
    TensorSetMismatch {
        /// Required tensor names.
        expected: Vec<String>,
        /// Observed tensor names.
        actual: Vec<String>,
    },

    /// A fitted tensor had the wrong dtype or shape.
    #[error("invalid tensor `{name}`: {message}")]
    InvalidTensor {
        /// Tensor name.
        name: String,
        /// Validation failure.
        message: String,
    },

    /// Candidate features did not satisfy the selected head contract.
    #[error("invalid head input: {0}")]
    InvalidInput(String),

    /// A disconnected caller requested cooperative cancellation of native work.
    #[error("native evaluation was cancelled")]
    ExecutionCancelled,

    /// The configured queue-inclusive evaluation deadline elapsed.
    #[error("native evaluation exceeded its configured deadline")]
    ExecutionDeadlineExceeded,

    /// A branch-state contract operation failed.
    #[error("branch state contract violated: {0}")]
    State(#[from] StateError),

    /// A persisted continuation-state snapshot was malformed or incompatible.
    #[error("invalid persisted Qwen state: {0}")]
    StatePersistence(String),

    /// Numerical evaluation failed instead of returning a finite distribution.
    #[error("head evaluation produced invalid arithmetic: {0}")]
    Numerical(String),
}

/// Request-scoped cancellation and deadline checks shared with native execution.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ExecutionControl {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl ExecutionControl {
    pub(crate) fn new(cancelled: Arc<AtomicBool>, deadline: Option<Instant>) -> Self {
        Self {
            cancelled,
            deadline,
        }
    }

    pub(crate) fn check(&self) -> Result<(), Qwen35Error> {
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(Qwen35Error::ExecutionDeadlineExceeded);
        }
        if self.cancelled.load(Ordering::Acquire) {
            return Err(Qwen35Error::ExecutionCancelled);
        }
        Ok(())
    }
}

/// Pinned branch-state identity of the selected profile's native CPU path.
///
/// Every `BackboneState` produced by `Qwen35Backbone` carries this identity,
/// and continuation rejects states carrying any other value.
#[must_use]
pub fn pinned_state_identity() -> StateIdentity {
    StateIdentity::new(
        PROFILE_ID,
        BACKBONE_ID,
        BACKBONE_REVISION,
        STATE_FIRST_RENDERER_ID,
        TOKENIZER_JSON_SHA256,
        EXECUTION_ARITHMETIC_ID,
    )
    .expect("pinned identity constants are non-empty")
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn require_equal(
    field: &'static str,
    expected: impl ToString,
    actual: impl ToString,
) -> Result<(), Qwen35Error> {
    let expected = expected.to_string();
    let actual = actual.to_string();
    if expected == actual {
        Ok(())
    } else {
        Err(Qwen35Error::ContractMismatch {
            field,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod execution_control_tests {
    use super::*;

    #[test]
    fn execution_control_distinguishes_deadline_from_cancellation() {
        let not_cancelled = Arc::new(AtomicBool::new(false));
        let deadline = ExecutionControl::new(not_cancelled.clone(), Some(Instant::now()));
        assert!(matches!(
            deadline.check(),
            Err(Qwen35Error::ExecutionDeadlineExceeded)
        ));

        let cancelled = Arc::new(AtomicBool::new(true));
        let control = ExecutionControl::new(cancelled, None);
        assert!(matches!(
            control.check(),
            Err(Qwen35Error::ExecutionCancelled)
        ));

        let active = ExecutionControl::new(not_cancelled, None);
        assert!(active.check().is_ok());
    }
}
