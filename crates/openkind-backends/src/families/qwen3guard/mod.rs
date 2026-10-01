//! Family: `qwen3guard`.
//!
//! A guardrail decision family served through the Qwen3Guard-Stream
//! checkpoint: a Qwen3 dense backbone whose per-token hidden states feed a
//! classification head producing a three-way risk distribution
//! (`Safe` / `Unsafe` / `Controversial`). The Stream readout never
//! generates a verdict string, so the profile satisfies the workspace
//! no-generation invariant — the surveyed `gen` variant's string-parse readout
//! is explicitly not implemented.
//!
//! Pinned profile: `Qwen/Qwen3Guard-Stream-0.6B` at
//! `419364a715de9840d47b1457982f64ff37f90ed4` (Apache-2.0), query-side
//! moderation of the state text, FP32 on CPU.
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the readout is a
//!   softmax over the offered class logits.
//! - Wire mapping: `Noul` maps to the `Unsafe` class probability.
//!   `Choice` options map by label (`safe`/`unsafe`/`controversial`,
//!   case-insensitive) to their class logits; the reserved `__none__` key
//!   maps to the `Controversial` class, which the checkpoint trains as the
//!   failure-to-decide mass. Offering both `controversial` and `__none__`
//!   fails closed. `Score` questions are rejected: the fixed preset has no
//!   ordinal contract.
//! - Continuation state: none; every question is one full forward pass.

/// Dense Qwen3 decoder forward, shared with the raw letter-logit control
/// family (`decoder-logit-qwen3`).
pub(crate) mod arch;
mod engine;
mod model;
#[cfg(feature = "onnx")]
#[doc(hidden)]
pub mod onnx;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::Qwen3GuardEngine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "qwen3guard";
/// Pinned backbone repository.
pub const BACKBONE_ID: &str = "Qwen/Qwen3Guard-Stream-0.6B";
/// Pinned immutable backbone revision.
pub const BACKBONE_REVISION: &str = "419364a715de9840d47b1457982f64ff37f90ed4";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "0fcf416cab16d94f933d";
/// SHA-256 of the pinned `model.safetensors` checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "e0a2eac6cc79cca5bf35bf8fb356f94c333dd9715f4f0dd58883b01f2fe33419";
/// SHA-256 of the pinned `config.json`.
pub const CONFIG_JSON_SHA256: &str =
    "b3bcfee86ed04c86e3c92000a2768d3c879564ac45e7e97909bce2e5767503cd";
/// SHA-256 of the pinned `tokenizer.json`.
pub const TOKENIZER_JSON_SHA256: &str =
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen3";
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

/// Calibration temperature applied to the risk-level class logits.
///
/// Pinned at `1.0` deliberately: on the pinned calibration workload the
/// checkpoint already assigns ≥ 0.994 to the correct side of every case, so
/// the NLL fit is degenerate (it drives to a hard one-hot through extreme
/// sharpening, T → 0.048, without improving decisions). The raw logits are
/// used as-is and the fit is recorded in `docs/BENCHMARKS.md`.
pub const CALIBRATION_TEMPERATURE: f64 = 1.0;

/// Risk-level class logits in checkpoint order.
pub(crate) const CLASS_SAFE: usize = 0;
pub(crate) const CLASS_UNSAFE: usize = 1;
pub(crate) const CLASS_CONTROVERSIAL: usize = 2;

/// Pinned `config.json` values the loader enforces before trusting weights.
pub(crate) fn pinned_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("architectures[0]", json!("Qwen3ForGuardModel")),
        ("model_type", json!("qwen3")),
        ("hidden_size", json!(1024)),
        ("num_hidden_layers", json!(28)),
        ("num_attention_heads", json!(16)),
        ("num_key_value_heads", json!(8)),
        ("head_dim", json!(128)),
        ("intermediate_size", json!(3072)),
        ("vocab_size", json!(151936)),
        ("rms_norm_eps", json!(1e-06)),
        ("rope_theta", json!(1000000)),
        ("max_position_embeddings", json!(8192)),
        ("guard_inner_size", json!(512)),
        ("num_risk_level", json!(3)),
        ("tie_word_embeddings", json!(true)),
    ]
}

/// Errors raised while loading or evaluating the pinned guard profile.
#[derive(Debug, Error)]
pub enum Qwen3GuardError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned guard profile.
#[derive(Debug, Clone)]
pub struct Qwen3GuardEngineConfig {
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// and `tokenizer.json` at [`BACKBONE_REVISION`]. Artifacts are verified
    /// in place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl Qwen3GuardEngineConfig {
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
