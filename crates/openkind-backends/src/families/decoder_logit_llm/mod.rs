//! Family: `decoder-logit-llm`.
//!
//! The GGUF-binding sibling of [`decoder-logit-letter`](crate::families::decoder_logit_letter):
//! a chat/instruct decoder loaded from a GGUF checkpoint through the candle
//! quantized-runner binding, with the identical letter-logit readout. The
//! pinned profile uses `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (q8_0 quantization)
//! at a frozen revision.
//!
//! Binding decision: the surveyed page left the binding open
//! (`llama-cpp-2` versus a thin `libllama` wrapper). This profile resolves
//! the question by using candle's quantized GGUF runner, which keeps the
//! build surface inside the existing Rust dependency set and the FP32
//! dequantization arithmetic deterministic per checkpoint. The profile
//! carries its own calibration temperature; quantization changes the logit
//! scale relative to the unquantized decoder-logit-letter profile.
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`].
//! - Continuation state: KV cache only, cleared after every question.
//! - Text generation: none — the readout never samples a token.

mod engine;
mod model;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::DecoderLlmEngine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "decoder-logit-llm";
/// Pinned backbone repository (GGUF export).
pub const BACKBONE_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
/// Pinned immutable backbone revision.
pub const BACKBONE_REVISION: &str = "9217f5db79a29953eb74d5343926648285ec7e67";
/// Pinned GGUF quantization variant.
pub const GGUF_VARIANT: &str = "qwen2.5-0.5b-instruct-q8_0.gguf";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "465963d705b6f35d6208";
/// SHA-256 of the pinned GGUF checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
/// SHA-256 of the pinned `tokenizer.json` (exported from the unquantized
/// source checkpoint; the GGUF file shares the tokenizer).
pub const TOKENIZER_JSON_SHA256: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-q8_0-qwen2";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Frozen maximum rendered prompt length. Longer requests fail closed;
/// truncation is forbidden.
pub const MAX_SEQUENCE_TOKENS: usize = 8_192;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this profile.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Calibration temperature fitted offline on the pinned calibration workload.
///
/// Fitted at 0.98179538607801242 by minimizing mean NLL over the shared
/// 23-case letter-family calibration workload; see `docs/BENCHMARKS.md` for
/// the recorded fit. Loaders reject any other temperature.
pub const CALIBRATION_TEMPERATURE: f64 = 0.981_795_386_078_012_4;

/// GGUF metadata values the loader enforces before trusting the checkpoint.
pub(crate) fn pinned_metadata() -> Vec<(&'static str, &'static str)> {
    vec![
        ("general.architecture", "qwen2"),
        ("qwen2.embedding_length", "896"),
        ("qwen2.block_count", "24"),
        ("qwen2.attention.head_count", "14"),
        ("qwen2.attention.head_count_kv", "2"),
    ]
}

/// Errors raised while loading or evaluating the pinned GGUF profile.
#[derive(Debug, Error)]
pub enum DecoderLlmError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned decoder-logit-llm profile.
#[derive(Debug, Clone)]
pub struct DecoderLlmEngineConfig {
    /// Model root containing the pinned `qwen2.5-0.5b-instruct-q8_0.gguf`
    /// and `tokenizer.json` at [`BACKBONE_REVISION`]. Artifacts are verified
    /// in place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DecoderLlmEngineConfig {
    /// Fill admission defaults around the given model root.
    pub fn new(model_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 2,
                retry_after_ms: 1_000,
                evaluation_timeout: Some(Duration::from_secs(600)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_id_matches_the_derivation_rule() {
        assert_eq!(
            PROFILE_ID,
            crate::families::support::derive_profile_id(
                FAMILY_SLUG,
                BACKBONE_ID,
                BACKBONE_REVISION
            )
        );
    }

    #[test]
    fn declared_probability_space_is_conditional() {
        assert_eq!(
            DECLARED_PROBABILITY_SPACE.as_str(),
            ProbabilitySpace::ConditionalOnOfferedOptions.as_str()
        );
    }
}
