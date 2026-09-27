//! Family: `kev`.
//!
//! The published Kev decision-model line (github.com/jaredpalmer/kev,
//! Apache-2.0): a LoRA adapter plus a pointer head on a Qwen base, trained
//! against the public `/v1/systemone` decision contract. The family page's
//! 2026-09-26 blocker — "no open checkpoint reproducing the reference
//! exists" — lapsed when the author published the checkpoints, the
//! reference implementation, the training config, and the evaluation
//! protocol under Apache-2.0. This profile pins the small member,
//! `jaredpalmer/kev-0.6b`, on `Qwen/Qwen3-0.6B-Base`.
//!
//! Architecture (per the published reference): a prefill-only causal
//! backbone that never generates text. One sequence per question — the
//! state document as a shared prefix, then the question branch
//! (`<|fim_middle|>` instruction, `<|box_start|>`/`<|box_end|>`-wrapped
//! options, `<|fim_suffix|>` decide token) — with a block-causal mask. This
//! port uses the reference's row form, proven equivalent to the packed
//! block-causal form by the reference's own tests: every question is one
//! causal row of state + branch, so no packed mask is needed on the CPU
//! path. The pointer head scores each option's `</opt>` boundary hidden
//! state against the decide hidden state
//! (`k(h_opt) · q(h_decide) / 16`), and the option softmax is the answer
//! distribution.
//!
//! Pinned profile artifacts (all verified in place, nothing downloaded at
//! load):
//! - adapter + tokenizer: `jaredpalmer/kev-0.6b` at
//!   `dece6dba8d43f0f7ded45e9f5b9df12474d90843` (Apache-2.0);
//! - base: `Qwen/Qwen3-0.6B-Base` at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
//!   (Apache-2.0), the exact revision recorded in the checkpoint's
//!   `head.pt`;
//! - pointer head: `head.safetensors`, a one-time operator conversion of
//!   the checkpoint's `head.pt` (torch pickle, source digest recorded
//!   below) via `safetensors.torch.save_file`; the conversion copies the
//!   four FP32 tensors verbatim.
//!
//! Wire mapping (per the reference `kev.api`): `Noul` scores the two
//! options `no`/`yes` (with the caller's rendered criteria when supplied)
//! and reports `p(true)`. `Choice` scores one option per offered criterion
//! (`name` or `name: description`) and reports probabilities by key.
//! `Score` scores the rendered level criteria and reports the expected
//! level. The reference's JSON text rendering (`kev.api.render`) is
//! reproduced for states, instructions, and criteria.

mod arch;
mod engine;
mod model;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::KevEngine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "kev";
/// Pinned checkpoint repository (adapter, tokenizer, head).
pub const BACKBONE_ID: &str = "jaredpalmer/kev-0.6b";
/// Pinned immutable checkpoint revision.
pub const BACKBONE_REVISION: &str = "dece6dba8d43f0f7ded45e9f5b9df12474d90843";
/// Pinned base-model repository, recorded inside the checkpoint's `head.pt`.
pub const BASE_MODEL_ID: &str = "Qwen/Qwen3-0.6B-Base";
/// Pinned immutable base-model revision (the checkpoint's `base_revision`).
pub const BASE_MODEL_REVISION: &str = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "39d88c11faeb4ac165fa";
/// SHA-256 of the pinned base checkpoint (`model.safetensors`).
pub const BASE_CHECKPOINT_SHA256: &str =
    "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba";
/// SHA-256 of the pinned base `config.json`.
pub const BASE_CONFIG_SHA256: &str =
    "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59";
/// SHA-256 of the pinned LoRA adapter (`adapter_model.safetensors`).
pub const ADAPTER_SHA256: &str = "deaab63b61f95d628503831e8b336e1c3e334b9a42d3530d979a5149b429c492";
/// SHA-256 of the pinned tokenizer (`tokenizer.json`).
pub const TOKENIZER_JSON_SHA256: &str =
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4";
/// SHA-256 of the pinned pointer head (`head.safetensors`), converted from
/// the checkpoint's `head.pt`.
pub const HEAD_SHA256: &str = "006201e630a8cdef3386e803e38d6374786c557cf7b5d8637a4774b52f075202";
/// SHA-256 of the original `head.pt` the converted head derives from,
/// recorded for provenance; loaders accept either artifact in the model
/// root and verify whichever is present.
pub const HEAD_PT_SOURCE_SHA256: &str =
    "ce6cd9ffc54db41c179b65a33d60973dc8280c29e886218eb3ec07dc28d6f28b";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen3";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Frozen maximum state-document length in tokens (the reference training
/// context's `max_state`). Longer requests fail closed; truncation is
/// forbidden.
pub const MAX_STATE_TOKENS: usize = 384;
/// Frozen maximum row length (state + one question branch), the reference
/// training context's `max_branch`.
pub const MAX_ROW_TOKENS: usize = 1024;
/// LoRA rank of the pinned adapter.
pub(crate) const LORA_RANK: usize = 16;
/// LoRA scale (`lora_alpha / r` = 32 / 16).
pub(crate) const LORA_SCALE: f64 = 2.0;
/// Pointer dimension of the pinned head.
pub(crate) const POINTER_DIM: usize = 256;

/// Calibration temperature applied to the pointer logits.
///
/// Pinned at `1.0` deliberately: the checkpoint ships raw logits (its
/// `head.pt` records no fitted temperature; the published ECE numbers are
/// raw), and the openkind NLL fit over the stated-fact calibration workload
/// is degenerate — the merged model already assigns ≥ 0.95 to the correct
/// side of 14 of 15 cases (≥ 0.99 on 12), so the fit drives to a hard
/// one-hot (T → 0.048, mean NLL → 0) without improving any decision. Raw
/// logits are used as-is; the fit is recorded in `docs/BENCHMARKS.md`.
pub const CALIBRATION_TEMPERATURE: f64 = 1.0;

/// Pinned base `config.json` values the loader enforces before trusting
/// weights.
pub(crate) fn pinned_base_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("model_type", json!("qwen3")),
        ("hidden_size", json!(1024)),
        ("num_hidden_layers", json!(28)),
        ("num_attention_heads", json!(16)),
        ("num_key_value_heads", json!(8)),
        ("head_dim", json!(128)),
        ("intermediate_size", json!(3072)),
        ("rms_norm_eps", json!(1e-06)),
        ("rope_theta", json!(1000000)),
        ("vocab_size", json!(151936)),
        ("tie_word_embeddings", json!(true)),
    ]
}

/// Errors raised while loading or evaluating the pinned kev profile.
#[derive(Debug, Error)]
pub enum KevError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned kev profile.
#[derive(Debug, Clone)]
pub struct KevEngineConfig {
    /// Model root containing the pinned checkpoint artifacts
    /// (`adapter_model.safetensors`, `tokenizer.json`, and
    /// `head.safetensors` or the original `head.pt`) at
    /// [`BACKBONE_REVISION`]. Verified in place; nothing is copied or
    /// downloaded.
    pub model_root: PathBuf,
    /// Base-model root containing the pinned `Qwen3-0.6B-Base`
    /// `model.safetensors` and `config.json` at [`BASE_MODEL_REVISION`].
    pub base_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl KevEngineConfig {
    /// Fill admission defaults around the given roots.
    pub fn new(model_root: impl Into<PathBuf>, base_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            base_root: base_root.into(),
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
