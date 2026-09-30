//! Family: `decoder-logit-qwen35`.
//!
//! The frozen 32-layer Qwen3.5 hybrid text backbone (24 gated-DeltaNet linear
//! attention layers, 8 grouped-query attention layers, hidden size 2,560)
//! prompts the question as a JSON decision payload and reads the next-token
//! logits at the answer slot, restricted to the option-letter tokens of the
//! SemIf protocol (16 letters per pass). No output token is ever sampled: the
//! answer is the temperature-calibrated distribution over letter logits.
//! Questions with more than 16 options are combined with the reference
//! runtime's knockout schedule. The pinned profile serves
//! `alibiserikbay/JevK5` (Apache-2.0), whose merged weights run on the shared
//! native CPU backbone in [`crate::qwen35::backbone`].
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the distribution
//!   over offered options sums to one and no semantic-none mass exists. An
//!   offered `__none__` key is scored as an ordinary option.
//! - Continuation state: none — every pass is an independent full-sequence
//!   forward and nothing is retained across questions or requests.
//! - Text generation: none.
//!
//! Reference contract: the `jevk5` runtime (`github.com/allebee/jevk5`,
//! Apache-2.0), whose `prompt.py` pins the exact prompt bytes, option
//! mapping, and knockout combination this profile reproduces.

mod engine;
mod knockout;
mod model;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::DecoderLogitQwen35Engine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "decoder-logit-qwen35";
/// Pinned checkpoint repository of the first profile (JevK5 v0.3).
pub const BACKBONE_ID: &str = "alibiserikbay/JevK5";
/// Pinned immutable checkpoint revision.
pub const BACKBONE_REVISION: &str = "c4f7fdb3aeab5582336406e78d3bef11bf98833d";
/// Stable derived profile ID for the pinned JevK5 profile.
pub const PROFILE_ID: &str = "415bcf4a064e6dadcf85";
/// SHA-256 of the pinned single-file BF16 checkpoint (`model.safetensors`).
pub const CHECKPOINT_SHA256: &str =
    "13824e47f2e40fe052f06943976cf742cb366ba305741a111e75a8ebae907a9c";
/// SHA-256 of the pinned `tokenizer.json`.
pub const TOKENIZER_JSON_SHA256: &str =
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";
/// SHA-256 of the pinned `config.json`.
pub const CONFIG_JSON_SHA256: &str =
    "63f47812d0f11118e4d252d2b3ad488707eb9287a11589f4fd382a1d31182724";
/// SHA-256 of the pinned `jevk5_config.json` holding the served temperatures.
pub const RUNTIME_CONFIG_SHA256: &str =
    "0d689fd13d15dc962265e2ae10b56359706ab5d05ad24e00e6334e4c19cf83d2";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen35-text";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Option letters of the SemIf protocol: one pass scores at most 16 options.
pub const LETTERS: &[u8] = b"ABCDEFGHIJKLMNOP";
/// Maximum options in one forward pass.
pub const MAX_OPTIONS_PER_PASS: usize = 16;
/// Maximum candidates per question. This bounds the number of independent
/// full-sequence forwards performed by the knockout schedule.
pub const MAX_CANDIDATES: usize = 32;
/// Minimum candidates per question, mirroring the Jev wire contract.
pub const MIN_CANDIDATES: usize = 2;
/// Frozen maximum rendered prompt length per pass. Longer requests fail
/// closed; truncation is forbidden.
pub const MAX_SEQUENCE_TOKENS: usize = 512;
/// Maximum aggregate prompt tokens evaluated for one question, including
/// every knockout pass.
pub const MAX_QUESTION_TOKENS: u64 = 1_536;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this profile. The threshold exists so telemetry and evidence artifacts
/// can record acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Letter-logit calibration temperature served by the reference runtime
/// (`jevk5_config.json`: `"temperature": 1.22`), fitted by the author on
/// held-out decisions and pinned here so loaders reject any other value.
pub const CALIBRATION_TEMPERATURE: f64 = 1.22;

/// Sharpening temperature for the knockout combination of more than 16
/// options (`jevk5_config.json`: `"knockout_temperature": 0.93`).
pub const KNOCKOUT_TEMPERATURE: f64 = 0.93;

/// Pinned `config.json` values the loader enforces before trusting weights.
///
/// These freeze the exact architecture the shared native backbone executes:
/// any drift in the layer geometry or attention schedule fails the load
/// instead of silently changing the forward graph.
pub(crate) fn pinned_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("architectures[0]", json!("Qwen3_5ForCausalLM")),
        ("model_type", json!("qwen3_5_text")),
        ("dtype", json!("bfloat16")),
        ("hidden_size", json!(2_560)),
        ("intermediate_size", json!(9_216)),
        ("num_hidden_layers", json!(32)),
        ("full_attention_interval", json!(4)),
        ("num_attention_heads", json!(16)),
        ("num_key_value_heads", json!(4)),
        ("head_dim", json!(256)),
        ("linear_num_key_heads", json!(16)),
        ("linear_num_value_heads", json!(32)),
        ("linear_key_head_dim", json!(128)),
        ("linear_value_head_dim", json!(128)),
        ("linear_conv_kernel_dim", json!(4)),
        ("vocab_size", json!(248_320)),
        ("tie_word_embeddings", json!(true)),
        ("rms_norm_eps", json!(1e-06)),
        ("partial_rotary_factor", json!(0.25)),
        ("rope_parameters.rope_theta", json!(10_000_000)),
        ("attn_output_gate", json!(true)),
        ("attention_bias", json!(false)),
    ]
}

/// Errors raised while loading or evaluating the pinned profile.
#[derive(Debug, Error)]
pub enum DecoderLogitQwen35Error {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned JevK5 profile.
#[derive(Debug, Clone)]
pub struct DecoderLogitQwen35EngineConfig {
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// `jevk5_config.json`, and `tokenizer.json` at [`BACKBONE_REVISION`].
    /// Artifacts are verified in place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DecoderLogitQwen35EngineConfig {
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
    fn temperatures_match_the_pinned_runtime_config_values() {
        assert_eq!(CALIBRATION_TEMPERATURE, 1.22);
        assert_eq!(KNOCKOUT_TEMPERATURE, 0.93);
    }

    #[test]
    fn letters_cover_the_pass_vocabulary() {
        assert_eq!(LETTERS.len(), MAX_OPTIONS_PER_PASS);
    }
}
