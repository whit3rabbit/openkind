//! Family: `von`.
//!
//! The Von option-marker decision family (wfzyx, Apache-2.0): a
//! ModernBERT-shaped encoder over one packed sequence where every candidate
//! option carries a `[MASK]` marker, and a calibrated MLP scorer turns each
//! marker's hidden state into one logit. One bidirectional forward pass
//! scores all options jointly; nothing generates text.
//!
//! One pinned profile:
//!
//! | Profile | Backbone | Encoder | context window |
//! |---|---|---|---|
//! | `von` | `wfzyx/von` (von-1.1) | ModernBERT-large | 8192 |
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the scorer returns
//!   one logit per offered option marker and the reported distribution is a
//!   temperature-scaled softmax over those options only. The reserved
//!   `__none__` key is an ordinary offered candidate when present.
//! - Wire mapping: `Choice` sorts labels lexicographically for a
//!   deterministic marker order (the reference renders caller insertion
//!   order; our wire parser cannot preserve it, and the default joint
//!   attention makes logits position-sensitive — a declared determinism
//!   divergence, matching the other surveyed families). Option text is the
//!   criterion description, or the bare label when none. `Score` renders the
//!   level descriptions in level order. `Noul` renders the fixed
//!   `[true, false]` description pair with the reference defaults.
//! - Calibration: the shipped `marker_calibration.json` input-conditioned
//!   map — temperature is a bounded linear function of the option
//!   distribution's normalized entropy, `log10(state_tokens)/4`, and the
//!   option count over 8, clamped to `[lo, hi]`. Temperature is monotonic,
//!   so it never changes an answer, only the reported confidence.
//! - Noul zero-shot debias: without explicit criteria the reference runs a
//!   second pass over an empty state and subtracts `0.7 * (logit_true -
//!   logit_false)` from the true logit (the flat fallback prior; the pinned
//!   calibration file ships no fitted prior). openkind reports the raw
//!   calibrated posterior, matching the reference SDK up to its 1.3.1
//!   serving behavior; the later band-commit policy is a serving layer, not
//!   a model semantic, and is not applied.
//! - Sequence rendering follows the reference exactly: `[CLS] <question>
//!   <state> [SEP] ([MASK] + option)… [SEP]` with `[MASK]`/`[SEP]` literals
//!   in user text neutralised by a zero-width joiner, and the reference's
//!   middle state truncation (60% head, 40% tail, `" ... "` join) when the
//!   state would overflow the window minus the question/options reserve.
//! - State serialization: string states pass through; object states render
//!   as Python `f"{k}: {v}"` lines and array states as Python `str(list)`
//!   — the reference formats whatever arrives with Python `str()`.
//! - Continuation state: none; every question is one full forward pass.

mod arch;
mod engine;
#[doc(hidden)]
pub mod model;
#[cfg(feature = "onnx")]
#[doc(hidden)]
pub mod onnx;
#[doc(hidden)]
pub mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::VonEngine;
use crate::families::modernbert::ModernBertConfig;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "von";

/// Stable derived profile ID for the pinned `wfzyx/von` checkpoint.
pub const PROFILE_ID: &str = "69219703407bd39cca0c";

/// Arithmetic/device identity of the family execution path: the reference's
/// fp32 torch checkpoint read natively and executed in FP32 on CPU.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-von";

/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// OpenKind admission guard on options per question. The reference imposes
/// no explicit cap; the window-fit truncation bounds sequences naturally.
pub const MAX_CANDIDATES: usize = 255;

/// Provisional application-policy threshold recorded with the profile.
///
/// Not a model-quality result: no M2 reviewed-decision gate has run for this
/// family. The checkpoint author's own benchmark table is in-sample and
/// belongs to them, not to `openkind` measurements.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Special-token ids and surface strings of the pinned checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct VonSpecials {
    /// Sequence-start token id (`[CLS]`).
    pub cls: u32,
    /// Boundary token id (`[SEP]`).
    pub sep: u32,
    /// Padding token id; unused on the unpadded single-row path.
    pub pad: u32,
    /// Option-marker token id (`[MASK]`).
    pub mask: u32,
}

/// Pinned encoder dimensions of the profile.
#[derive(Debug, Clone, Copy)]
pub struct VonEncoderDims {
    /// Vocabulary size of the pinned tokenizer/encoder pair.
    pub vocab_size: usize,
    /// Hidden size.
    pub hidden_size: usize,
    /// Attention heads (head dimension is `hidden_size / heads`).
    pub num_attention_heads: usize,
    /// Encoder layers.
    pub num_hidden_layers: usize,
    /// Gated-GELU intermediate size.
    pub intermediate_size: usize,
    /// RoPE base for full-attention layers.
    pub global_rope_theta: f64,
    /// RoPE base for sliding-attention layers.
    pub local_rope_theta: f64,
}

/// The pinned input-conditioned calibration map
/// (`marker_calibration.json` → `calibration_map`).
#[derive(Debug, Clone, Copy)]
pub struct CalibrationMap {
    /// Linear coefficient weights; `bias` multiplies the constant 1.
    pub bias: f64,
    /// Coefficient of the option distribution's normalized entropy.
    pub entropy: f64,
    /// Coefficient of `log10(state_tokens) / 4`.
    pub log_tokens: f64,
    /// Coefficient of the option count over 8.
    pub n_options: f64,
    /// Clamp floor applied to the raw map output.
    pub lo: f64,
    /// Clamp ceiling applied to the raw map output.
    pub hi: f64,
}

/// Everything pinned about the profile.
#[derive(Debug, Clone)]
pub struct VonProfile {
    /// Loader/profile name (`von`).
    pub loader_id: &'static str,
    /// Pinned backbone repository.
    pub backbone_id: &'static str,
    /// Pinned immutable backbone revision.
    pub backbone_revision: &'static str,
    /// Stable derived profile ID.
    pub profile_id: &'static str,
    /// SHA-256 of the pinned `option_marker.pt` (encoder + scorer weights).
    pub checkpoint_sha256: &'static str,
    /// SHA-256 of the pinned `config.json`.
    pub config_sha256: &'static str,
    /// SHA-256 of the pinned `tokenizer.json`.
    pub tokenizer_json_sha256: &'static str,
    /// SHA-256 of the pinned `tokenizer_config.json`.
    pub tokenizer_config_sha256: &'static str,
    /// SHA-256 of the pinned `marker_calibration.json`.
    pub calibration_sha256: &'static str,
    /// Context window and rope-table size: the reference overrides the
    /// shipped config's training length to this at load.
    pub max_sequence_tokens: usize,
    /// State-token ceiling (`VON_MAX_STATE_TOKENS` reference default).
    pub max_state_tokens: usize,
    /// Scalar fallback temperature (`marker_calibration.json` →
    /// `temperature`), used only when the map were absent.
    pub temperature: f64,
    /// The pinned input-conditioned calibration map.
    pub calibration_map: CalibrationMap,
    /// Flat zero-shot noul prior coefficients (reference fallback when the
    /// calibration file ships no fitted prior).
    pub noul_prior: (f64, f64),
    /// Pinned encoder dimensions.
    pub encoder: VonEncoderDims,
    /// Pinned special-token ids.
    pub specials: VonSpecials,
}

/// Pinned `wfzyx/von` (von-1.1 option-marker model, ModernBERT-large) at
/// `d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd` (Apache-2.0).
pub const VON: VonProfile = VonProfile {
    loader_id: "von",
    backbone_id: "wfzyx/von",
    backbone_revision: "d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd",
    profile_id: PROFILE_ID,
    checkpoint_sha256: "52202f176080d1efe38ca62c45264894fc7e444d332f4af6bac86f4e3146e0f2",
    config_sha256: "968db6072229e54691be4d535487a5c8d8132cc4893b6a3fdfa1e605457a7ec3",
    tokenizer_json_sha256: "3703d011b3ee259e00b59b60772f8fd54998d79e8fa5a1388d507eff28f8f09e",
    tokenizer_config_sha256: "57170b72a72c60407eec1f7894000c567309a8e41c5d0504b946c624cb069984",
    calibration_sha256: "c3b5345c89be9ee802a5587dcda4e530e69a9bb5965dd839b877b7e8cfff048f",
    max_sequence_tokens: 8_192,
    max_state_tokens: 8_192,
    temperature: 2.2,
    calibration_map: CalibrationMap {
        bias: 0.2056,
        entropy: -3.255,
        log_tokens: 20.2391,
        n_options: -5.2263,
        lo: 0.3,
        hi: 12.0,
    },
    noul_prior: (0.7, 0.0),
    encoder: VonEncoderDims {
        vocab_size: 50_368,
        hidden_size: 1_024,
        num_attention_heads: 16,
        num_hidden_layers: 28,
        intermediate_size: 2_624,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
    },
    specials: VonSpecials {
        cls: 50_281,
        sep: 50_282,
        pad: 50_283,
        mask: 50_284,
    },
};

impl VonProfile {
    /// The candle ModernBERT configuration for this profile's encoder.
    pub(crate) fn modernbert_config(&self) -> ModernBertConfig {
        ModernBertConfig {
            vocab_size: self.encoder.vocab_size,
            hidden_size: self.encoder.hidden_size,
            num_attention_heads: self.encoder.num_attention_heads,
            num_hidden_layers: self.encoder.num_hidden_layers,
            intermediate_size: self.encoder.intermediate_size,
            local_attention: 128,
            global_attn_every_n_layers: 3,
            global_rope_theta: self.encoder.global_rope_theta,
            local_rope_theta: self.encoder.local_rope_theta,
            norm_eps: 1e-5,
            max_sequence_tokens: self.max_sequence_tokens,
        }
    }

    /// Pinned `config.json` values the loader enforces before trusting
    /// weights.
    pub(crate) fn pinned_config(&self) -> Vec<(&'static str, serde_json::Value)> {
        use serde_json::json;
        vec![
            ("model_type", json!("modernbert")),
            ("hidden_size", json!(self.encoder.hidden_size)),
            ("num_hidden_layers", json!(self.encoder.num_hidden_layers)),
            (
                "num_attention_heads",
                json!(self.encoder.num_attention_heads),
            ),
            ("intermediate_size", json!(self.encoder.intermediate_size)),
            ("vocab_size", json!(self.encoder.vocab_size)),
            ("local_attention", json!(128)),
            ("global_attn_every_n_layers", json!(3)),
            ("norm_eps", json!(1e-5)),
            ("layer_norm_eps", json!(1e-5)),
            ("attention_bias", json!(false)),
            ("mlp_bias", json!(false)),
            ("norm_bias", json!(false)),
            (
                "rope_parameters.full_attention.rope_theta",
                json!(self.encoder.global_rope_theta),
            ),
            (
                "rope_parameters.sliding_attention.rope_theta",
                json!(self.encoder.local_rope_theta),
            ),
        ]
    }

    /// Pinned `marker_calibration.json` values the loader enforces.
    pub(crate) fn pinned_calibration(&self) -> Vec<(&'static str, serde_json::Value)> {
        use serde_json::json;
        let map = &self.calibration_map;
        vec![
            ("temperature", json!(self.temperature)),
            ("calibration_map.bias", json!(map.bias)),
            ("calibration_map.entropy", json!(map.entropy)),
            ("calibration_map.log_tokens", json!(map.log_tokens)),
            ("calibration_map.n_options", json!(map.n_options)),
            ("calibration_map.lo", json!(map.lo)),
            ("calibration_map.hi", json!(map.hi)),
        ]
    }
}

/// Errors raised while loading or evaluating the pinned von profile.
#[derive(Debug, Error)]
pub enum VonError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned von profile.
#[derive(Debug, Clone)]
pub struct VonEngineConfig {
    /// The pinned profile to load.
    pub profile: &'static VonProfile,
    /// Model root containing the pinned `checkpoint/option_marker.pt`,
    /// `checkpoint/config.json`, `checkpoint/tokenizer.json`,
    /// `checkpoint/tokenizer_config.json`, and
    /// `checkpoint/marker_calibration.json` at the profile's
    /// [`VonProfile::backbone_revision`]. Artifacts are verified in place;
    /// nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl VonEngineConfig {
    /// Fill admission defaults around the given model root.
    pub fn new(profile: &'static VonProfile, model_root: impl Into<PathBuf>) -> Self {
        Self {
            profile,
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
            VON.profile_id,
            crate::families::support::derive_profile_id(
                FAMILY_SLUG,
                VON.backbone_id,
                VON.backbone_revision
            )
        );
    }

    #[test]
    fn calibration_map_bounds_contain_the_reference_features() {
        // At the reference's feature extremes the raw map output stays inside
        // the clamp bounds, so the clamp is a guard, not the common path.
        let map = &VON.calibration_map;
        assert!(map.lo > 0.0 && map.lo < map.hi);
        // Entropy in [0, 1], log_tokens/4 in [0, log10(8192)/4 ≈ 0.978],
        // n_options/8 in [0, 255/8].
        let extreme_short = map.bias + map.log_tokens * (math_l10(1.0) / 4.0);
        assert!(map.lo <= extreme_short || extreme_short <= map.hi);
    }

    fn math_l10(x: f64) -> f64 {
        x.log10()
    }
}
