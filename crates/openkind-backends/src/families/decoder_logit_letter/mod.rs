//! Family: `decoder-logit-letter`.
//!
//! A decoder backbone prompts the question with lettered options and reads
//! the next-token logits at the answer slot, restricted to option-letter
//! tokens. No output token is ever sampled: the answer is the temperature-
//! calibrated distribution over letter logits. The pinned profile uses
//! `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) at a frozen revision through
//! the candle `qwen2` implementation, FP32 on CPU.
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the distribution
//!   over offered options sums to one and no semantic-none mass exists. A
//!   offered `__none__` key is scored as an ordinary option.
//! - Continuation state: KV cache only, cleared after every question; no
//!   state is retained across questions or requests.
//! - Text generation: none.

mod engine;
mod model;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::DecoderLetterEngine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "decoder-logit-letter";
/// Pinned backbone repository.
pub const BACKBONE_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";
/// Pinned immutable backbone revision.
pub const BACKBONE_REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "5492c97dfcdaf3fe9439";
/// SHA-256 of the pinned `model.safetensors` checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe";
/// SHA-256 of the pinned `tokenizer.json`.
pub const TOKENIZER_JSON_SHA256: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
/// SHA-256 of the pinned `config.json`.
pub const CONFIG_JSON_SHA256: &str =
    "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen2";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Letter vocabulary size: option letters `A`..`Z`.
pub const MAX_OPTIONS: usize = 26;
/// Minimum candidates per question, mirroring the Jev wire contract.
pub const MIN_CANDIDATES: usize = 2;
/// Frozen maximum rendered prompt length. Longer requests fail closed;
/// truncation is forbidden.
pub const MAX_SEQUENCE_TOKENS: usize = 8_192;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this profile. The threshold exists so telemetry and evidence artifacts
/// can record acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Calibration temperature fitted offline on the pinned calibration workload.
///
/// Fitted at 0.76926237511012552 by minimizing mean NLL over 23 deterministic
/// synthetic decision cases whose correct option is stated verbatim in the
/// state document (12 `Choice`, 6 `Noul`, 5 `Score`); see
/// `docs/BENCHMARKS.md` for the recorded fit. The value is frozen here so
/// loaders reject any other temperature.
pub const CALIBRATION_TEMPERATURE: f64 = 0.769_262_375_110_125_5;

/// Pinned `config.json` values the loader enforces before trusting weights.
pub(crate) fn pinned_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("architectures[0]", json!("Qwen2ForCausalLM")),
        ("model_type", json!("qwen2")),
        ("hidden_size", json!(896)),
        ("intermediate_size", json!(4864)),
        ("num_hidden_layers", json!(24)),
        ("num_attention_heads", json!(14)),
        ("num_key_value_heads", json!(2)),
        ("vocab_size", json!(151936)),
        ("max_position_embeddings", json!(32768)),
        ("tie_word_embeddings", json!(true)),
        ("rope_theta", json!(1000000.0)),
        ("rms_norm_eps", json!(1e-06)),
        ("sliding_window", json!(32768)),
    ]
}

/// Errors raised while loading or evaluating the pinned decoder-letter profile.
#[derive(Debug, Error)]
pub enum DecoderLetterError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned decoder-letter profile.
#[derive(Debug, Clone)]
pub struct DecoderLetterEngineConfig {
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// and `tokenizer.json` at [`BACKBONE_REVISION`]. Artifacts are verified
    /// in place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DecoderLetterEngineConfig {
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
