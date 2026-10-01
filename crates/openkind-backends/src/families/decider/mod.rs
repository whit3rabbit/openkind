//! Family: `decider`.
//!
//! The frozen 32-layer Qwen3.5 hybrid text backbone (the same geometry the
//! `decoder-logit-qwen35` family serves) prompts each question as a plain
//! state-first decision block and reads the next-token logits at the answer
//! slot — the final `(` of `Answer: (` — restricted to uppercase label
//! tokens. No chat template wraps the prompt and no output token is ever
//! sampled: the answer is the temperature-calibrated distribution over label
//! logits. `Score` questions are judged one level per row (a yes/no "does
//! this level fit" read each) and the per-level fit probabilities normalize
//! into the level distribution, mirroring the reference `isolated_levels`
//! readout.
//!
//! One pinned profile:
//!
//! | Profile | Checkpoint | Reference runtime | Calibration |
//! |---|---|---|---|
//! | `decider-4b` (4b-v2.1) | `Mapika/decider-4b` | `decider-ai` >= 1.4 (`github.com/Mapika/decider`) | choice 1.11, noul 1.56, score 1.287 |
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the distribution
//!   over offered options sums to one and no semantic-none mass exists. An
//!   offered `__none__` key is scored as an ordinary option.
//! - Continuation state: none — every row is an independent full-sequence
//!   forward and nothing is retained across questions or requests.
//! - Text generation: none.
//!
//! Reference contract: the `decider` package shipped next to the checkpoint
//! (`decider/prompt.py`, `decider/systemone.py`, `decider/infer.py`,
//! Apache-2.0), whose state serialization, option rendering, isolated-level
//! rows, and slot readout this family reproduces. Serving always renders
//! options in the order given (the reference `_NoShuffle` path); our wire
//! criteria map is a hash map, so Choice options render in the wire's sorted
//! label order — a declared determinism difference, as in the other surveyed
//! families.

mod engine;
mod model;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::DeciderEngine;
use crate::families::decoder_logit_qwen35::Calibration;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "decider";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen35-text";
/// Declared probability space of every profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Provisional application-policy threshold recorded with every profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this family. The threshold exists so telemetry and evidence artifacts
/// can record acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Everything pinned about one profile of this family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeciderProfile {
    /// Loader/profile name, e.g. `decider-4b`.
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
    /// SHA-256 of the pinned `decider_config.json`.
    pub decider_config_sha256: &'static str,
    /// Calibrated slot-logit temperature(s) from `decider_config.json`.
    pub calibration: Calibration,
    /// Frozen maximum candidates per Choice question (`decider_config.json`:
    /// `"max_options": 255`).
    pub max_candidates: usize,
    /// Frozen maximum Score levels per question (reference `MAX_LEVELS`).
    pub max_score_levels: usize,
    /// Frozen token cap of the serialized state (`decider_config.json`:
    /// `"max_state_tokens": 32768`; keep-first truncation).
    pub max_state_tokens: usize,
    /// Frozen maximum rendered row length. Longer requests fail closed;
    /// truncation is forbidden beyond the state cap.
    pub max_row_tokens: usize,
    /// Frozen minimum length of a state array that gets `_index` annotations
    /// (reference `ANNOTATE_MIN`).
    pub annotate_min: usize,
    /// Backend id reported by the CPU engine serving this profile.
    pub cpu_backend_id: &'static str,
    /// Backend id reported by the CUDA engine serving this profile
    /// (`cuda` feature).
    pub cuda_backend_id: &'static str,
    /// Release date of the profile (the day it was pinned here).
    pub release_date: &'static str,
    /// Catalog/manifest description of the profile.
    pub description: &'static str,
}

/// Pinned `Mapika/decider-4b` (4b-v2.1, Apache-2.0) at
/// `eb5fbdfc9448473ec25e399882912863afbdb70e`: Qwen3.5-4B-Base plus a
/// supervised pass and a rank-64 LoRA, merged; plain state-first layout,
/// isolated Score levels, `neutralize_none` off.
pub const DECIDER_4B: DeciderProfile = DeciderProfile {
    loader_id: "decider-4b",
    backbone_id: "Mapika/decider-4b",
    backbone_revision: "eb5fbdfc9448473ec25e399882912863afbdb70e",
    profile_id: "0529bf6f2bed84641701",
    checkpoint_sha256: "ee8ce585b3cedd93206dd149b09b4bdd683174874211f85c77a36090b90c9fdd",
    tokenizer_json_sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    config_json_sha256: "63f47812d0f11118e4d252d2b3ad488707eb9287a11589f4fd382a1d31182724",
    decider_config_sha256: "fc83293b6f707172e20176d603ea88dd1a3be2abd3596fdb1ccbec9404626baf",
    // decider_config.json: "temperature": 1.099 with "temperature_by_type":
    // {"choice": 1.11, "noul": 1.56, "score": 1.287}.
    calibration: Calibration::ByType {
        choice: 1.11,
        score: 1.287,
        noul: 1.56,
    },
    max_candidates: 255,
    max_score_levels: 10,
    max_state_tokens: 32_768,
    max_row_tokens: 65_536,
    annotate_min: 8,
    cpu_backend_id: "decider-4b/cpu-fp32",
    cuda_backend_id: "decider-4b/cuda-fp32",
    release_date: "2026-09-30",
    description: "Pinned decider-4b merged Qwen3.5-4B slot-logit decision decoder (Apache-2.0, \
                  plain state-first layout, isolated score levels); prototype readout, research \
                  status",
};

/// Profiles of this family, in catalog order.
pub const PROFILES: &[&DeciderProfile] = &[&DECIDER_4B];

/// The profile with the given loader ID, if this build pins it.
pub fn profile_by_loader_id(loader_id: &str) -> Option<&'static DeciderProfile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.loader_id == loader_id)
}

/// Narrow option count of the reference renderer: up to ten options render
/// as `(A) text` lines tokenized as one string; wider sets use the
/// single-token label table.
pub const NARROW_OPTIONS: usize = 10;

/// Pinned `config.json` values the loader enforces before trusting weights
/// (the same frozen backbone geometry as the `decoder-logit-qwen35` family).
pub(crate) fn pinned_config() -> Vec<(&'static str, serde_json::Value)> {
    crate::families::decoder_logit_qwen35::pinned_config_shared()
}

/// Pinned `decider_config.json` values the loader enforces before trusting
/// weights. These freeze the serving semantics the engine reproduces: a
/// config that flips the layout, re-enables none-neutralization, or changes
/// the option geometry fails the load instead of silently changing answers.
pub(crate) fn pinned_decider_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("version", json!("4b-v2.1")),
        ("layout", json!("plain")),
        ("neutralize_none", json!(false)),
        ("schema_first", json!(false)),
        ("schema_first_trained", json!(false)),
        ("isolated_levels", json!(true)),
        ("max_options", json!(255)),
        ("max_state_tokens", json!(32_768)),
    ]
}

/// Errors raised while loading or evaluating a pinned profile.
#[derive(Debug, Error)]
pub enum DeciderError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for one pinned profile of this family.
#[derive(Debug, Clone)]
pub struct DeciderEngineConfig {
    /// The pinned profile to load.
    pub profile: &'static DeciderProfile,
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// `decider_config.json`, and `tokenizer.json` at the profile's
    /// [`DeciderProfile::backbone_revision`]. Artifacts are verified in
    /// place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DeciderEngineConfig {
    /// Fill admission defaults around the given profile and model root.
    pub fn new(profile: &'static DeciderProfile, model_root: impl Into<PathBuf>) -> Self {
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
    fn temperatures_match_the_pinned_decider_config_values() {
        assert_eq!(
            DECIDER_4B.calibration,
            Calibration::ByType {
                choice: 1.11,
                score: 1.287,
                noul: 1.56,
            }
        );
    }

    #[test]
    fn score_levels_stay_inside_the_label_table() {
        const {
            assert!(DECIDER_4B.max_score_levels <= NARROW_OPTIONS);
            assert!(DECIDER_4B.max_candidates <= renderer::MAX_LABELS);
        }
    }

    #[test]
    fn lookup_finds_every_profile() {
        for profile in PROFILES {
            assert_eq!(profile_by_loader_id(profile.loader_id), Some(*profile));
        }
        assert_eq!(profile_by_loader_id("missing"), None);
    }
}
