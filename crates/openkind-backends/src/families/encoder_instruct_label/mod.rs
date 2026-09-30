//! Family: `encoder-instruct-label`.
//!
//! A label-marker decision family served through a GLiClass-style
//! uni-encoder checkpoint: one forward pass carries every candidate as a
//! `<<LABEL>>`-marked span ahead of a `<<SEP>>` boundary and the state text;
//! the encoder's class-token positions are pooled, projected, and scored
//! against the pooled text representation with a dot-product scorer. No
//! candidate count multiplies the forward passes, and nothing generates text.
//!
//! Pinned profile: `knowledgator/gliclass-modern-base-v3.0` at
//! `ac369222ca4375ca66ebaf7fb5220f223514c035` (Apache-2.0), a ModernBERT-base
//! encoder (22 layers, hidden 768) hand-implemented for the pinned candle
//! 0.8.0 line in [`arch`], FP32 on CPU. The checkpoint selection evidence
//! (and the rejected smaller-edge and DeBERTa-v3 alternatives) is recorded
//! on the family page in `docs/families`.
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — every offered
//!   candidate is scored as its own label marker; the reported distribution
//!   is normalized over the offered options only, and the reserved
//!   `__none__` key is an ordinary marker when offered.
//! - Wire mapping: `Choice` renders every offered criterion (or label text)
//!   as a marker and renormalizes the temperature-calibrated sigmoids over
//!   the offered set. `Score` renders the ordered level criteria and takes a
//!   temperature softmax over their logits. `Noul` renders one proposition
//!   marker — the caller's `true` criterion when explicit criteria are
//!   supplied, the question instruction verbatim otherwise — and reports its
//!   calibrated support sigmoid.
//! - Readout determinism: marker scores shift with marker order, so the
//!   profile owns a fixed order (sorted labels for `Choice`, level order for
//!   `Score`) and probabilities are only defined under that order.
//! - Continuation state: none; every question is one full forward pass.

mod engine;
#[doc(hidden)]
pub mod model;
#[doc(hidden)]
pub mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::EncoderInstructLabelEngine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "encoder-instruct-label";
/// Pinned backbone repository (GLiClass uni-encoder over ModernBERT-base).
pub const BACKBONE_ID: &str = "knowledgator/gliclass-modern-base-v3.0";
/// Pinned immutable backbone revision.
pub const BACKBONE_REVISION: &str = "ac369222ca4375ca66ebaf7fb5220f223514c035";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "9fd68313a5606eca42f2";
/// SHA-256 of the pinned `model.safetensors` checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "b83af831dc664ff552fc52054f9ec0def73b83d81ce554ae1c91b86ade7cd369";
/// SHA-256 of the pinned `config.json`.
pub const CONFIG_JSON_SHA256: &str =
    "f28b93a1ab70736f6b0f38d9da058bdabc045238cd8c50bb6c334197c797d1e8";
/// SHA-256 of the pinned `tokenizer.json`.
pub const TOKENIZER_JSON_SHA256: &str =
    "7c1979be5ac04a6681dbfbb98a2b01b176883a298f2a7240c93048b244118a01";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-modernbert";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Frozen maximum encoded marker+state sequence length, matching the
/// reference pipeline's 1024-token budget. Longer requests fail closed;
/// truncation is forbidden.
pub const MAX_SEQUENCE_TOKENS: usize = 1024;
/// Maximum candidates per question, equal to the checkpoint's trained
/// `max_num_classes` of 25; more markers leave the head's training envelope.
pub const MAX_CANDIDATES: usize = 25;

/// Token id of the `<<LABEL>>` class marker in the pinned vocabulary.
pub(crate) const CLASS_TOKEN_ID: u32 = 50368;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this profile. The threshold exists so telemetry and evidence artifacts
/// can record acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Calibration temperature applied to the raw marker logits.
///
/// Fitted at 0.44530092168688412 by minimizing mean NLL over the pinned
/// stated-fact calibration workload (15 synthetic decision cases: 6
/// `Choice`, 5 `Noul`, 4 `Score`) with the exact wire readout of each
/// primitive — renormalized sigmoids for `Choice`, softmax for `Score`, the
/// support sigmoid for `Noul`. Mean NLL improves from 0.194 (T=1) to 0.151
/// at the optimum; the optimizer settles interior, so the fit is not the
/// degenerate sharpening case. Recorded in `docs/BENCHMARKS.md`; loaders
/// reject any other temperature.
pub const CALIBRATION_TEMPERATURE: f64 = 0.445_300_921_686_884;

/// Pinned `config.json` values the loader enforces before trusting weights.
pub(crate) fn pinned_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("architecture_type", json!("uni-encoder")),
        ("architectures[0]", json!("GLiClassModel")),
        ("class_token_index", json!(50368)),
        ("text_token_index", json!(50369)),
        ("embed_class_token", json!(true)),
        ("prompt_first", json!(true)),
        ("pooling_strategy", json!("first")),
        ("scorer_type", json!("simple")),
        ("normalize_features", json!(false)),
        ("max_num_classes", json!(25)),
        ("hidden_size", json!(768)),
        ("vocab_size", json!(50370)),
        ("encoder_config.model_type", json!("modernbert")),
        ("encoder_config.hidden_size", json!(768)),
        ("encoder_config.num_hidden_layers", json!(22)),
        ("encoder_config.num_attention_heads", json!(12)),
        ("encoder_config.intermediate_size", json!(1152)),
        ("encoder_config.local_attention", json!(128)),
        ("encoder_config.global_attn_every_n_layers", json!(3)),
        ("encoder_config.global_rope_theta", json!(160000.0)),
        ("encoder_config.local_rope_theta", json!(10000.0)),
        ("encoder_config.norm_eps", json!(1e-05)),
        ("encoder_config.position_embedding_type", json!("absolute")),
        ("encoder_config.attention_bias", json!(false)),
        ("encoder_config.mlp_bias", json!(false)),
        ("encoder_config.norm_bias", json!(false)),
        ("encoder_config.vocab_size", json!(50370)),
    ]
}

/// Errors raised while loading or evaluating the pinned label-marker profile.
#[derive(Debug, Error)]
pub enum EncoderInstructLabelError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// MLX/Metal execution backend for the pinned profile (feature `mlx`).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
impl From<crate::qwen35::mlx::MlxError> for EncoderInstructLabelError {
    fn from(error: crate::qwen35::mlx::MlxError) -> Self {
        Self::Family(crate::families::support::FamilyError::from(error))
    }
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub use self::mlx::{
    EncoderInstructLabelMlxEngine, EncoderInstructLabelMlxEngineConfig, MLX_EXECUTION_ARITHMETIC_ID,
};

/// Filesystem configuration for the pinned label-marker profile.
#[derive(Debug, Clone)]
pub struct EncoderInstructLabelEngineConfig {
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// and `tokenizer.json` at [`BACKBONE_REVISION`]. Artifacts are verified
    /// in place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl EncoderInstructLabelEngineConfig {
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

    #[test]
    fn candidate_bounds_match_the_checkpoint_envelope() {
        assert_eq!(MAX_CANDIDATES, 25);
        const { assert!(MAX_CANDIDATES <= MAX_SEQUENCE_TOKENS) };
    }
}
