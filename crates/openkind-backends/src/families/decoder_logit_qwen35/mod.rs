//! Family: `decoder-logit-qwen35`.
//!
//! The frozen 32-layer Qwen3.5 hybrid text backbone (24 gated-DeltaNet linear
//! attention layers, 8 grouped-query attention layers, hidden size 2,560)
//! prompts the question as a JSON decision payload and reads the next-token
//! logits at the answer slot, restricted to the option-letter tokens of the
//! SemIf protocol (16 letters per pass). No output token is ever sampled: the
//! answer is the temperature-calibrated distribution over letter logits.
//! Wide questions above a profile's single-pass width use the reference
//! runtime's knockout schedule where that runtime defines one.
//!
//! Two pinned profiles share this module; they differ in pinned artifacts,
//! calibration, and pass schedule:
//!
//! | Profile | Checkpoint | Reference runtime | Calibration |
//! |---|---|---|---|
//! | `decoder-logit-qwen35` (JevK5 v0.3) | `alibiserikbay/JevK5` | `jevk5` knockout runtime | uniform 1.22, knockout 0.93 |
//! | `plumb-4b` (Plumb-4B v5.2) | `crh225/plumb-4b` | `jevk5` v0.2.0 single read | choice/noul 2.07, score 1.2 |
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
//! Apache-2.0), whose prompt bytes, option mapping, and knockout combination
//! this family reproduces. The Plumb-4B reference serving layer is the
//! author's `plumb_server.py` (`github.com/crh225/plumb`, Apache-2.0) on top
//! of `jevk5` v0.2.0; its JevBench-specific `--noul-commit` band reporting is
//! a scoring policy, not the model distribution, and is deliberately not
//! reproduced.

mod engine;
mod knockout;
pub(crate) mod model;
/// Prompt-byte helpers shared with sibling families (Python `json.dumps`
/// serialization for state payloads).
pub(crate) mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::DecoderLogitQwen35Engine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "decoder-logit-qwen35";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen35-text";
/// Renderer identity of the pinned letter-pass rendering: every backend that
/// produces continuation states for this family records this identifier so
/// states from different renderings cannot be confused.
pub const RENDERER_ID: &str = "jevk5_letter_pass";
/// Declared probability space of every profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Option letters of the SemIf protocol: one pass scores at most 16 options.
pub const LETTERS: &[u8] = b"ABCDEFGHIJKLMNOP";
/// Maximum options in one forward pass.
pub const MAX_OPTIONS_PER_PASS: usize = 16;
/// Maximum candidates per question across the family, mirroring the Jev wire
/// contract. Each profile caps this further through
/// [`Qwen35LogitProfile::max_candidates`].
pub const MAX_CANDIDATES: usize = 32;
/// Minimum candidates per question, mirroring the Jev wire contract.
pub const MIN_CANDIDATES: usize = 2;

/// Provisional application-policy threshold recorded with every profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this family. The threshold exists so telemetry and evidence artifacts
/// can record acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Calibrated letter-logit temperature of one profile.
///
/// Uniform profiles apply one temperature to every question type; by-type
/// profiles follow the reference serving layer's per-type table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Calibration {
    /// One temperature for Choice, Score, and Noul questions.
    Uniform(f64),
    /// Per-type temperatures from the reference serving layer.
    ByType {
        /// Temperature for Choice questions.
        choice: f64,
        /// Temperature for Score questions.
        score: f64,
        /// Temperature for Noul questions.
        noul: f64,
    },
}

/// Question primitive of the pass being calibrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuestionKind {
    /// Choice questions.
    Choice,
    /// Score questions.
    Score,
    /// Noul (yes/no) questions.
    Noul,
}

impl Calibration {
    /// The temperature for one question primitive.
    pub(crate) fn resolve(&self, primitive: QuestionKind) -> f64 {
        match self {
            Self::Uniform(temperature) => *temperature,
            Self::ByType {
                choice,
                score,
                noul,
            } => match primitive {
                QuestionKind::Choice => *choice,
                QuestionKind::Score => *score,
                QuestionKind::Noul => *noul,
            },
        }
    }
}

/// Everything pinned about one profile of this family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen35LogitProfile {
    /// Loader/profile name, e.g. `decoder-logit-qwen35`.
    pub loader_id: &'static str,
    /// Pinned checkpoint repository.
    pub backbone_id: &'static str,
    /// Pinned immutable checkpoint revision.
    pub backbone_revision: &'static str,
    /// Stable derived profile ID.
    pub profile_id: &'static str,
    /// SHA-256 of the pinned single-file BF16 checkpoint (`model.safetensors`).
    pub checkpoint_sha256: &'static str,
    /// SHA-256 of the pinned `tokenizer.json`.
    pub tokenizer_json_sha256: &'static str,
    /// SHA-256 of the pinned `config.json`.
    pub config_json_sha256: &'static str,
    /// Path of the pinned runtime-config artifact holding the served
    /// temperatures, relative to the model root.
    pub runtime_config_path: &'static str,
    /// SHA-256 of the pinned runtime-config artifact.
    pub runtime_config_sha256: &'static str,
    /// Calibrated letter-logit temperature(s).
    pub calibration: Calibration,
    /// Sharpening temperature of the knockout combination for wide
    /// questions, pinned by the reference runtime config. `None` means the
    /// reference runtime has no knockout schedule and questions wider than
    /// one pass fail closed.
    pub knockout_temperature: Option<f64>,
    /// Frozen maximum candidates per question for this profile.
    pub max_candidates: usize,
    /// Frozen maximum rendered prompt length per pass. Longer requests fail
    /// closed; truncation is forbidden.
    pub max_sequence_tokens: usize,
    /// Maximum aggregate prompt tokens evaluated for one question, including
    /// every knockout pass.
    pub max_question_tokens: u64,
    /// Backend id reported by the CPU engine serving this profile.
    pub cpu_backend_id: &'static str,
    /// Backend id reported by the MLX engine serving this profile.
    pub mlx_backend_id: &'static str,
    /// Release date of the profile (the day it was pinned here).
    pub release_date: &'static str,
    /// Catalog/manifest description of the profile.
    pub description: &'static str,
}

/// Pinned `alibiserikbay/JevK5` (JevK5 v0.3, Apache-2.0) at
/// `c4f7fdb3aeab5582336406e78d3bef11bf98833d`, served by the knockout
/// `jevk5` runtime contract.
pub const JEVK5: Qwen35LogitProfile = Qwen35LogitProfile {
    loader_id: "decoder-logit-qwen35",
    backbone_id: "alibiserikbay/JevK5",
    backbone_revision: "c4f7fdb3aeab5582336406e78d3bef11bf98833d",
    profile_id: "415bcf4a064e6dadcf85",
    checkpoint_sha256: "13824e47f2e40fe052f06943976cf742cb366ba305741a111e75a8ebae907a9c",
    tokenizer_json_sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    config_json_sha256: "63f47812d0f11118e4d252d2b3ad488707eb9287a11589f4fd382a1d31182724",
    runtime_config_path: "jevk5_config.json",
    runtime_config_sha256: "0d689fd13d15dc962265e2ae10b56359706ab5d05ad24e00e6334e4c19cf83d2",
    // jevk5_config.json: "temperature": 1.22, "knockout_temperature": 0.93.
    calibration: Calibration::Uniform(1.22),
    knockout_temperature: Some(0.93),
    max_candidates: 32,
    max_sequence_tokens: 512,
    max_question_tokens: 1_536,
    cpu_backend_id: "decoder-logit-qwen35/cpu-fp32",
    mlx_backend_id: "decoder-logit-qwen35/mlx-fp32",
    release_date: "2026-09-27",
    description: "Pinned JevK5 merged Qwen3.5-4B letter-logit decision decoder (Apache-2.0); \
                  prototype readout, research status",
};

/// Pinned `crh225/plumb-4b` (Plumb-4B v5.2, Apache-2.0) at
/// `24f7bf77e7ee258a2d158c61ea2dce2b60321010`: JevK5 v0.2 further fine-tuned
/// on hard decisions, merged BF16, served by `jevk5` v0.2.0 single-read
/// semantics with the author server's per-type temperatures.
pub const PLUMB_4B: Qwen35LogitProfile = Qwen35LogitProfile {
    loader_id: "plumb-4b",
    backbone_id: "crh225/plumb-4b",
    backbone_revision: "24f7bf77e7ee258a2d158c61ea2dce2b60321010",
    profile_id: "c1f080794d38e94a0bc2",
    checkpoint_sha256: "89e119ea07f4c5b4b6715560c7de6694ec3b777dfd0e62351da0b33986d2e1e1",
    tokenizer_json_sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    config_json_sha256: "63f47812d0f11118e4d252d2b3ad488707eb9287a11589f4fd382a1d31182724",
    runtime_config_path: "jevk5_config.json",
    runtime_config_sha256: "a971c01fcf3ec161a03c61f5e3883b93fbd27696a529a54acd989845bf6fdead",
    // jevk5_config.json pins "temperature": 2.07; the reference server reads
    // Score questions at 1.2 (README, --score-temperature).
    calibration: Calibration::ByType {
        choice: 2.07,
        score: 1.2,
        noul: 2.07,
    },
    // jevk5 v0.2.0 has no knockout schedule; the reference server rejects
    // questions wider than one pass.
    knockout_temperature: None,
    max_candidates: 16,
    max_sequence_tokens: 16_384,
    max_question_tokens: 16_384,
    cpu_backend_id: "plumb-4b/cpu-fp32",
    mlx_backend_id: "plumb-4b/mlx-fp32",
    release_date: "2026-09-30",
    description: "Pinned Plumb-4B merged Qwen3.5-4B letter-logit decision decoder (Apache-2.0, \
                  JevK5 v0.2 runtime, single read); prototype readout, research status",
};

/// Profile IDs of every profile in this family, in catalog order.
pub const PROFILES: &[&Qwen35LogitProfile] = &[&JEVK5, &PLUMB_4B];

/// The profile with the given loader ID, if this build pins it.
pub fn profile_by_loader_id(loader_id: &str) -> Option<&'static Qwen35LogitProfile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.loader_id == loader_id)
}

/// The profile with the given derived profile ID, if this build pins it.
pub fn profile_by_id(profile_id: &str) -> Option<&'static Qwen35LogitProfile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.profile_id == profile_id)
}

/// Pinned `config.json` values the loader enforces before trusting weights.
///
/// These freeze the exact architecture the shared native backbone executes:
/// any drift in the layer geometry or attention schedule fails the load
/// instead of silently changing the forward graph. Every profile of this
/// family shares the same frozen backbone contract (identical `config.json`
/// digests).
pub(crate) fn pinned_config() -> Vec<(&'static str, serde_json::Value)> {
    pinned_config_shared()
}

/// The frozen backbone geometry, shared with sibling families that serve the
/// same 32-layer Qwen3.5 hybrid checkpoint layout (e.g. `decider`).
pub(crate) fn pinned_config_shared() -> Vec<(&'static str, serde_json::Value)> {
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

/// Errors raised while loading or evaluating a pinned profile.
#[derive(Debug, Error)]
pub enum DecoderLogitQwen35Error {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// MLX/Metal execution backend for the pinned profiles (feature `mlx`).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub use self::mlx::{DecoderLogitQwen35MlxEngine, DecoderLogitQwen35MlxEngineConfig};

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
/// Arithmetic/device identity of the family MLX execution path.
pub use self::mlx::MLX_EXECUTION_ARITHMETIC_ID;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
impl From<crate::qwen35::mlx::MlxError> for DecoderLogitQwen35Error {
    fn from(error: crate::qwen35::mlx::MlxError) -> Self {
        Self::Family(crate::families::support::FamilyError::from(error))
    }
}

/// Filesystem configuration for one pinned profile of this family.
#[derive(Debug, Clone)]
pub struct DecoderLogitQwen35EngineConfig {
    /// The pinned profile to load.
    pub profile: &'static Qwen35LogitProfile,
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// runtime config, and `tokenizer.json` at the profile's
    /// [`Qwen35LogitProfile::backbone_revision`]. Artifacts are verified in
    /// place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DecoderLogitQwen35EngineConfig {
    /// Fill admission defaults around the given profile and model root.
    pub fn new(profile: &'static Qwen35LogitProfile, model_root: impl Into<PathBuf>) -> Self {
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

// Backwards-compatible single-profile aliases for the first (JevK5) profile.
/// Pinned checkpoint repository of the JevK5 profile.
pub const BACKBONE_ID: &str = JEVK5.backbone_id;
/// Pinned immutable checkpoint revision of the JevK5 profile.
pub const BACKBONE_REVISION: &str = JEVK5.backbone_revision;
/// Stable derived profile ID of the JevK5 profile.
pub const PROFILE_ID: &str = JEVK5.profile_id;
/// SHA-256 of the JevK5 single-file BF16 checkpoint.
pub const CHECKPOINT_SHA256: &str = JEVK5.checkpoint_sha256;
/// SHA-256 of the JevK5 `tokenizer.json`.
pub const TOKENIZER_JSON_SHA256: &str = JEVK5.tokenizer_json_sha256;
/// SHA-256 of the JevK5 `config.json`.
pub const CONFIG_JSON_SHA256: &str = JEVK5.config_json_sha256;
/// SHA-256 of the JevK5 `jevk5_config.json`.
pub const RUNTIME_CONFIG_SHA256: &str = JEVK5.runtime_config_sha256;
/// Letter-logit calibration temperature served by the JevK5 reference runtime.
pub const CALIBRATION_TEMPERATURE: f64 = 1.22;
/// Sharpening temperature for the JevK5 knockout combination.
pub const KNOCKOUT_TEMPERATURE: f64 = 0.93;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_ids_match_the_derivation_rule() {
        for profile in PROFILES {
            assert_eq!(
                profile.profile_id,
                crate::families::support::derive_profile_id(
                    FAMILY_SLUG,
                    profile.backbone_id,
                    profile.backbone_revision
                ),
                "{}",
                profile.loader_id
            );
        }
    }

    #[test]
    fn temperatures_match_the_pinned_runtime_config_values() {
        assert_eq!(JEVK5.calibration, Calibration::Uniform(1.22));
        assert_eq!(JEVK5.knockout_temperature, Some(0.93));
        assert_eq!(
            PLUMB_4B.calibration,
            Calibration::ByType {
                choice: 2.07,
                score: 1.2,
                noul: 2.07,
            }
        );
        assert_eq!(PLUMB_4B.knockout_temperature, None);
    }

    #[test]
    fn every_calibration_temperature_is_positive_and_finite() {
        for profile in PROFILES {
            let temperatures = match profile.calibration {
                Calibration::Uniform(temperature) => vec![temperature],
                Calibration::ByType {
                    choice,
                    score,
                    noul,
                } => vec![choice, score, noul],
            };
            for temperature in temperatures {
                assert!(
                    temperature.is_finite() && temperature > 0.0,
                    "{}: bad temperature {temperature}",
                    profile.loader_id
                );
            }
        }
    }

    #[test]
    fn single_read_profiles_never_exceed_one_pass_width() {
        for profile in PROFILES {
            if profile.knockout_temperature.is_none() {
                assert!(
                    profile.max_candidates <= MAX_OPTIONS_PER_PASS,
                    "{}: no knockout schedule but {} candidates",
                    profile.loader_id,
                    profile.max_candidates
                );
            }
        }
    }

    #[test]
    fn letters_cover_the_pass_vocabulary() {
        assert_eq!(LETTERS.len(), MAX_OPTIONS_PER_PASS);
    }

    #[test]
    fn lookup_finds_every_profile_by_loader_id_and_id() {
        for profile in PROFILES {
            assert_eq!(profile_by_loader_id(profile.loader_id), Some(*profile));
            assert_eq!(profile_by_id(profile.profile_id), Some(*profile));
        }
        assert_eq!(profile_by_loader_id("missing"), None);
        assert_eq!(profile_by_id("missing"), None);
    }
}
