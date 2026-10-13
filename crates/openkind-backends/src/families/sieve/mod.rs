//! Family: `sieve`.
//!
//! The sthanika-ai Sieve checkpoints prompt every question as a tag-delimited
//! branch over a shared `<state>` prefix — the chat bytes are assembled
//! directly (no chat template file), option lines carry single-token answer
//! codes (`A..Z` then single-token two-letter codes), and a trained 255-row
//! float32 head (`head.safetensors`, initialised from the base's `lm_head`
//! rows of those codes) is read at the final branch position. The serving
//! checkpoint is a LoRA adapter over a pinned base plus the head: the merge
//! (`base + (B @ A) · alpha/rank`, PEFT `merge_and_unload` semantics) runs
//! once per load through the shared [`MergedLoRABackend`]. Attention stays
//! fully **causal** — the prefix-plus-branch row is one ordinary causal
//! sequence — so this family reuses the standard backbone path and keeps
//! every mask invariant intact.
//!
//! One pinned profile:
//!
//! | Profile | Checkpoint | Reference runtime | Calibration |
//! |---|---|---|---|
//! | `sieve-4b` (v1) | `sthanika-ai/Sieve-4B` over `Qwen/Qwen3.5-4B` | the shipped `sieve4b` package (Apache-2.0) | choice 1.1608, noul 1.4403, score 1.0 |
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the softmax over
//!   the offered options' code logits sums to one. An offered `__none__`
//!   key is scored as an ordinary option.
//! - Continuation state: none — every branch is an independent
//!   prefix-plus-branch forward; nothing is retained across questions or
//!   requests (the reference's shared-prefix KV cache is a serving
//!   optimization this family does not need).
//! - Text generation: none.
//! - Noul: the reference answers every Noul question **twice** — once with
//!   the options listed Yes-first, once No-first — and averages the two
//!   distributions to cancel listing bias; this family reproduces that.
//!
//! Reference contract: the `sieve4b` package shipped beside the checkpoint
//! (`prompt.py`, `model.py`, `postprocess.py`, Apache-2.0), whose state
//! fencing (`defang`), tag-delimited branches, code-table derivation,
//! per-type temperatures, and two-order noul this family reproduces. Our
//! wire criteria map is a hash map, so Choice options render in the wire's
//! sorted label order; answers map through openkind's own wire contract
//! (entropy confidence, unrounded distributions) rather than the
//! reference's TypeSafe confidence formulas and 4-decimal rounding — the
//! same declared differences as the other surveyed families.

mod engine;
mod model;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::{plan_request as plan_sieve_request, SieveEngine, SieveRequestPlan};
pub use self::renderer::{derive_answer_codes as derive_sieve_answer_codes, SieveRenderer};
use crate::families::decoder_logit_qwen35::Calibration;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "sieve";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen35-text-lora";
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
pub struct SieveProfile {
    /// Loader/profile name, e.g. `sieve-4b`.
    pub loader_id: &'static str,
    /// Pinned serving checkpoint repository (adapter + head).
    pub backbone_id: &'static str,
    /// Pinned immutable serving revision.
    pub backbone_revision: &'static str,
    /// Stable derived profile ID.
    pub profile_id: &'static str,
    /// Frozen base-checkpoint geometry (causal).
    pub geometry: crate::qwen35::Qwen35Geometry,
    /// Pinned base-checkpoint repository (`sieve_config.json` `base`).
    pub base_id: &'static str,
    /// Pinned base-checkpoint revision (`sieve_config.json` `revision`).
    pub base_revision: &'static str,
    /// Base checkpoint shards as `(path, bytes, sha256)` under the install's
    /// `base/` directory.
    pub base_shards: &'static [(&'static str, u64, &'static str)],
    /// SHA-256 of the pinned base `config.json`.
    pub base_config_sha256: &'static str,
    /// Serving checkpoint files as `(path, bytes, sha256)` under the
    /// install's `checkpoint/` directory.
    pub checkpoint_files: &'static [(&'static str, u64, &'static str)],
    /// LoRA rank of the pinned adapter (`adapter_config.json` `r`).
    pub lora_rank: usize,
    /// LoRA alpha of the pinned adapter (`adapter_config.json`
    /// `lora_alpha`).
    pub lora_alpha: usize,
    /// LoRA scale (`lora_alpha / lora_rank`) of the pinned adapter; the
    /// loader rejects a profile where the two disagree.
    pub lora_scale: f64,
    /// Frozen readout width (`sieve_config.json` `MAX_OPTIONS`).
    pub max_candidates: usize,
    /// Frozen maximum Score levels per question.
    pub max_score_levels: usize,
    /// Reference `contract.py:MAX_BRANCH_TOKENS`: the state tokens plus the
    /// request's longest question branch, preamble excluded. Longer requests
    /// fail closed before any forward because the reference never truncates.
    pub max_branch_tokens: usize,
    /// Reference `contract.py:MAX_REQUEST_TOKENS`: the state tokens once plus
    /// every question's longest branch, preamble excluded.
    pub max_request_tokens: usize,
    /// Calibrated per-type temperatures from `sieve_config.json`.
    pub calibration: Calibration,
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

/// Pinned `sthanika-ai/Sieve-4B` (Apache-2.0) at
/// `ea7cd1edee29590a82ed9fbfcb644227a8eed5f1`: a rank-32 LoRA (alpha 64,
/// the full hybrid projection set) over `Qwen/Qwen3.5-4B` at
/// `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` plus a trained `[255, 2560]`
/// float32 head.
pub const SIEVE_4B: SieveProfile = SieveProfile {
    loader_id: "sieve-4b",
    backbone_id: "sthanika-ai/Sieve-4B",
    backbone_revision: "ea7cd1edee29590a82ed9fbfcb644227a8eed5f1",
    profile_id: "8b97b56524b55c41fbf4",
    geometry: crate::qwen35::Qwen35Geometry::PINNED,
    base_id: "Qwen/Qwen3.5-4B",
    base_revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    base_shards: &[
        (
            "model.safetensors-00001-of-00002.safetensors",
            5_329_398_688,
            "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
        ),
        (
            "model.safetensors-00002-of-00002.safetensors",
            3_990_429_408,
            "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
        ),
    ],
    base_config_sha256: "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    checkpoint_files: &[
        (
            "adapter_model.safetensors",
            259_784_032,
            "2a36be08bc4d87cf2fd580a2447d6ec4571cf98edefe4291de7d3e23b2b34439",
        ),
        (
            "head.safetensors",
            2_611_288,
            "b4e82bbe75f6ce5364ebdf85da83867ac2ba34874a7349e2ecb1dbf86a0fe6d5",
        ),
        (
            "adapter_config.json",
            1_381,
            "dea5991c0d64fac1393dca174ae57c407c28288008fbc41c3478ec865458187b",
        ),
        (
            "sieve_config.json",
            6_135,
            "222bb07cc8eeb455dd43feb5dde71421191f45b896015d69d7db70a9d08b6143",
        ),
        (
            "tokenizer_config.json",
            16_710,
            "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
        ),
        (
            "provenance.json",
            4_735,
            "a4290f8129e5940eee6b1efc11dd01a08149c6c399793d4bebf00ad9a74dc95b",
        ),
        (
            "LICENSE",
            10_745,
            "8489ebe7c8abe903fcdaace4080ef1660bba98ea44c8007923bfb91777945e37",
        ),
        (
            "tokenizer.json",
            12_807_982,
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        ),
    ],
    lora_rank: 32,
    lora_alpha: 64,
    lora_scale: 2.0,
    max_candidates: 255,
    max_score_levels: 10,
    max_branch_tokens: 32_768,
    max_request_tokens: 65_536,
    // sieve_config.json "temperatures".
    calibration: Calibration::ByType {
        choice: 1.1608255534843908,
        score: 1.0,
        noul: 1.4402880083655132,
    },
    cpu_backend_id: "sieve-4b/cpu-fp32",
    cuda_backend_id: "sieve-4b/cuda-fp32",
    release_date: "2026-10-12",
    description: "Pinned sieve-4b causal LoRA-merged Qwen3.5-4B code-readout decision engine \
                  (Apache-2.0, sieve4b package contract); prototype readout, research status",
};

/// Profiles of this family, in catalog order.
pub const PROFILES: &[&SieveProfile] = &[&SIEVE_4B];

/// The profile with the given loader ID, if this build pins it.
pub fn profile_by_loader_id(loader_id: &str) -> Option<&'static SieveProfile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.loader_id == loader_id)
}

/// The profile with the given derived profile ID, if this build pins it.
pub fn profile_by_id(profile_id: &str) -> Option<&'static SieveProfile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.profile_id == profile_id)
}

/// Width of the code readout and of the derived answer-code table.
pub const CODE_TABLE_WIDTH: usize = 255;

/// Errors raised while loading or evaluating a pinned profile.
#[derive(Debug, Error)]
pub enum SieveError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for one pinned profile of this family.
///
/// The model root is the installation root: `base/` holds the pinned base
/// checkpoint shards and config, `checkpoint/` holds the serving checkpoint
/// (adapter, head, configs, tokenizer). Artifacts are verified in place;
/// nothing is copied or downloaded.
#[derive(Debug, Clone)]
pub struct SieveEngineConfig {
    /// The pinned profile to load.
    pub profile: &'static SieveProfile,
    /// Installation root containing `base/` and `checkpoint/`.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl SieveEngineConfig {
    /// Fill admission defaults around the given profile and model root.
    pub fn new(profile: &'static SieveProfile, model_root: impl Into<PathBuf>) -> Self {
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
    fn lookup_finds_every_profile() {
        for profile in PROFILES {
            assert_eq!(profile_by_loader_id(profile.loader_id), Some(*profile));
            assert_eq!(profile_by_id(profile.profile_id), Some(*profile));
        }
        assert_eq!(profile_by_loader_id("missing"), None);
        assert_eq!(profile_by_id("missing"), None);
    }

    #[test]
    fn every_profile_is_causal_with_pinned_calibration() {
        for profile in PROFILES {
            assert!(
                profile.geometry.causal,
                "{} serves a causal checkpoint",
                profile.loader_id
            );
            assert_eq!(
                profile.calibration,
                Calibration::ByType {
                    choice: 1.1608255534843908,
                    score: 1.0,
                    noul: 1.4402880083655132,
                }
            );
            assert_eq!(profile.max_candidates, CODE_TABLE_WIDTH);
        }
    }

    #[test]
    fn every_profile_scale_is_alpha_over_rank() {
        // PEFT merges with `lora_alpha / r` (no rslora); a profile whose
        // scale disagrees would merge a different model than it pins.
        for profile in PROFILES {
            assert_eq!(
                profile.lora_scale,
                profile.lora_alpha as f64 / profile.lora_rank as f64,
                "{}",
                profile.loader_id
            );
        }
    }

    #[test]
    fn token_caps_match_the_reference_contract() {
        // Reference `contract.py`: MAX_BRANCH_TOKENS and MAX_REQUEST_TOKENS.
        assert_eq!(SIEVE_4B.max_branch_tokens, 32_768);
        assert_eq!(SIEVE_4B.max_request_tokens, 65_536);
    }
}
