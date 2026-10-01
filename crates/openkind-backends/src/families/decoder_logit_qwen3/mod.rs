//! Family: `decoder-logit-qwen3`.
//!
//! The raw direct-logit controls of the JevBench board: unmodified dense
//! Qwen3 instruction/back checkpoints prompted with the family's letter-pass
//! decision renderer and read through the tied embedding rows of the option
//! letters at the answer slot. No fine-tuned decision weights exist for these
//! checkpoints — their board rows measure how far a stock model plus a
//! constrained scoring readout goes — and the profiles pin that control
//! setup: temperature 1.0 (no calibration is fitted for a raw control), one
//! forward per question, no knockout schedule (questions above one pass
//! width fail closed), and no sampling.
//!
//! Three pinned profiles:
//!
//! | Profile | Checkpoint | Assistant tail |
//! |---|---|---|
//! | `decoder-logit-qwen3-06b` | `Qwen/Qwen3-0.6B` | thinking-capable: empty `<think>` block |
//! | `decoder-logit-qwen3-17b` | `Qwen/Qwen3-1.7B` | thinking-capable: empty `<think>` block |
//! | `decoder-logit-qwen3-4b` | `Qwen/Qwen3-4B-Instruct-2507` | non-thinking-only: bare assistant turn |
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the distribution
//!   over offered options sums to one and no semantic-none mass exists. An
//!   offered `__none__` key is scored as an ordinary option.
//! - Continuation state: none — every question is one independent
//!   full-sequence forward.
//! - Text generation: none.
//!
//! Renderer contract: the prompt bytes replicate the reference chat
//! templates of the pinned checkpoints exactly (system + user JSON payload,
//! thinking off where the template supports it), and the decision payload
//! reuses the `jevk5` letter-pass shape (`github.com/allebee/jevk5`,
//! Apache-2.0, which documents SemIf as its protocol origin) so the controls
//! are methodology-comparable with the fine-tuned letter-logit profiles of
//! the `decoder-logit-qwen35` family. The renderer is a declared
//! construction of this family: the upstream checkpoints ship no serving
//! runtime of their own.

mod engine;
mod model;
#[cfg(feature = "onnx")]
#[doc(hidden)]
pub mod onnx;
/// Prompt-byte helpers, exposed for offline renderer tests.
#[doc(hidden)]
pub mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::DecoderLogitQwen3Engine;
use crate::families::decoder_logit_qwen35::Calibration;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "decoder-logit-qwen3";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen3";
/// Declared probability space of every profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Option letters of the SemIf protocol: one pass scores at most 16 options.
pub const LETTERS: &[u8] = b"ABCDEFGHIJKLMNOP";
/// Maximum options in the single forward pass; wider questions fail closed.
pub const MAX_OPTIONS_PER_PASS: usize = 16;
/// Minimum candidates per question, mirroring the Jev wire contract.
pub const MIN_CANDIDATES: usize = 2;
/// Frozen maximum rendered prompt length per pass. Longer requests fail
/// closed; truncation is forbidden.
pub const MAX_SEQUENCE_TOKENS: usize = 512;

/// Provisional application-policy threshold recorded with every profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this family, and a raw control carries no deployment claim at all.
/// The threshold exists so telemetry and evidence artifacts can record
/// acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// The pinned chat-template assistant tail of one profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistantTail {
    /// Thinking-capable templates rendered with thinking off: the empty
    /// `<think>` block closes the assistant turn.
    EmptyThinkBlock,
    /// Non-thinking-only templates (Qwen3 Instruct 2507): the assistant turn
    /// ends at the bare generation prompt.
    BareGenerationPrompt,
}

impl AssistantTail {
    /// The literal template tail bytes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EmptyThinkBlock => "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            Self::BareGenerationPrompt => "<|im_start|>assistant\n",
        }
    }
}

/// Everything pinned about one profile of this family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen3LogitProfile {
    /// Loader/profile name, e.g. `decoder-logit-qwen3-06b`.
    pub loader_id: &'static str,
    /// Pinned checkpoint repository.
    pub backbone_id: &'static str,
    /// Pinned immutable checkpoint revision.
    pub backbone_revision: &'static str,
    /// Stable derived profile ID.
    pub profile_id: &'static str,
    /// Pinned shard file names, relative to the model root, in load order.
    pub checkpoint_shards: &'static [&'static str],
    /// SHA-256 of every pinned shard, in [`Self::checkpoint_shards`] order.
    pub checkpoint_sha256s: &'static [&'static str],
    /// SHA-256 of the pinned `tokenizer.json`.
    pub tokenizer_json_sha256: &'static str,
    /// SHA-256 of the pinned `config.json`.
    pub config_json_sha256: &'static str,
    /// Assistant chat-template tail (thinking-capable vs 2507 bare).
    pub assistant_tail: AssistantTail,
    /// Pinned dense geometry for the shared forward.
    pub hidden_size: usize,
    /// Pinned intermediate size for MLP layers.
    pub intermediate_size: usize,
    /// Number of transformer hidden layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads.
    pub num_key_value_heads: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Base frequency for RoPE rotary embeddings.
    pub rope_theta: f64,
    /// Backend id reported by the CPU engine serving this profile.
    pub cpu_backend_id: &'static str,
    /// Release date of the profile (the day it was pinned here).
    pub release_date: &'static str,
    /// Catalog/manifest description of the profile.
    pub description: &'static str,
}

impl Qwen3LogitProfile {
    /// The pinned calibration of a raw control: raw logits, no fitted
    /// temperature.
    pub fn calibration(&self) -> Calibration {
        Calibration::Uniform(1.0)
    }
}

/// Pinned `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`
/// (Apache-2.0): the board's 0.6B raw-logit control.
pub const QWEN3_06B: Qwen3LogitProfile = Qwen3LogitProfile {
    loader_id: "decoder-logit-qwen3-06b",
    backbone_id: "Qwen/Qwen3-0.6B",
    backbone_revision: "c1899de289a04d12100db370d81485cdf75e47ca",
    profile_id: "d900f4af57509fe02e62",
    checkpoint_shards: &["model.safetensors"],
    checkpoint_sha256s: &["f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"],
    tokenizer_json_sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    config_json_sha256: "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    assistant_tail: AssistantTail::EmptyThinkBlock,
    hidden_size: 1024,
    intermediate_size: 3072,
    num_hidden_layers: 28,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    head_dim: 128,
    vocab_size: 151_936,
    rope_theta: 1_000_000.0,
    cpu_backend_id: "decoder-logit-qwen3-06b/cpu-fp32",
    release_date: "2026-10-01",
    description: "Pinned raw Qwen3-0.6B direct-logit decision control (Apache-2.0, temperature \
                  1.0, single read); untrained control readout, research status",
};

/// Pinned `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
/// (Apache-2.0): the board's 1.7B raw-logit control.
pub const QWEN3_17B: Qwen3LogitProfile = Qwen3LogitProfile {
    loader_id: "decoder-logit-qwen3-17b",
    backbone_id: "Qwen/Qwen3-1.7B",
    backbone_revision: "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    profile_id: "8119b9271f8d011e7d03",
    checkpoint_shards: &[
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ],
    checkpoint_sha256s: &[
        "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
    ],
    tokenizer_json_sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    config_json_sha256: "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
    assistant_tail: AssistantTail::EmptyThinkBlock,
    hidden_size: 2048,
    intermediate_size: 6144,
    num_hidden_layers: 28,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    head_dim: 128,
    vocab_size: 151_936,
    rope_theta: 1_000_000.0,
    cpu_backend_id: "decoder-logit-qwen3-17b/cpu-fp32",
    release_date: "2026-10-01",
    description: "Pinned raw Qwen3-1.7B direct-logit decision control (Apache-2.0, temperature \
                  1.0, single read); untrained control readout, research status",
};

/// Pinned `Qwen/Qwen3-4B-Instruct-2507` at
/// `cdbee75f17c01a7cc42f958dc650907174af0554` (Apache-2.0): the board's
/// highest-ranked raw-logit control (#21), non-thinking-only.
pub const QWEN3_4B: Qwen3LogitProfile = Qwen3LogitProfile {
    loader_id: "decoder-logit-qwen3-4b",
    backbone_id: "Qwen/Qwen3-4B-Instruct-2507",
    backbone_revision: "cdbee75f17c01a7cc42f958dc650907174af0554",
    profile_id: "9dfaf11792a8d061b6b8",
    checkpoint_shards: &[
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ],
    checkpoint_sha256s: &[
        "75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6",
        "0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1",
        "7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d",
    ],
    tokenizer_json_sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    config_json_sha256: "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
    assistant_tail: AssistantTail::BareGenerationPrompt,
    hidden_size: 2560,
    intermediate_size: 9728,
    num_hidden_layers: 36,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    vocab_size: 151_936,
    rope_theta: 5_000_000.0,
    cpu_backend_id: "decoder-logit-qwen3-4b/cpu-fp32",
    release_date: "2026-10-01",
    description: "Pinned raw Qwen3-4B-Instruct-2507 direct-logit decision control (Apache-2.0, \
                  temperature 1.0, single read); untrained control readout, research status",
};

/// Profiles of this family, in catalog order.
pub const PROFILES: &[&Qwen3LogitProfile] = &[&QWEN3_06B, &QWEN3_17B, &QWEN3_4B];

/// The profile with the given loader ID, if this build pins it.
pub fn profile_by_loader_id(loader_id: &str) -> Option<&'static Qwen3LogitProfile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.loader_id == loader_id)
}

/// Pinned `config.json` values the loader enforces before trusting weights,
/// derived from the profile's pinned geometry.
pub(crate) fn pinned_config(profile: &Qwen3LogitProfile) -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("architectures[0]", json!("Qwen3ForCausalLM")),
        ("model_type", json!("qwen3")),
        ("hidden_size", json!(profile.hidden_size)),
        ("intermediate_size", json!(profile.intermediate_size)),
        ("num_hidden_layers", json!(profile.num_hidden_layers)),
        ("num_attention_heads", json!(profile.num_attention_heads)),
        ("num_key_value_heads", json!(profile.num_key_value_heads)),
        ("head_dim", json!(profile.head_dim)),
        ("vocab_size", json!(profile.vocab_size)),
        ("tie_word_embeddings", json!(true)),
        ("rms_norm_eps", json!(1e-06)),
        // HF writes integral rope thetas as JSON integers; compare as u64.
        ("rope_theta", json!(profile.rope_theta as u64)),
    ]
}

/// Errors raised while loading or evaluating a pinned profile.
#[derive(Debug, Error)]
pub enum DecoderLogitQwen3Error {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for one pinned profile of this family.
#[derive(Debug, Clone)]
pub struct DecoderLogitQwen3EngineConfig {
    /// The pinned profile to load.
    pub profile: &'static Qwen3LogitProfile,
    /// Model root containing the pinned shards, `config.json`, and
    /// `tokenizer.json` at the profile's
    /// [`Qwen3LogitProfile::backbone_revision`]. Artifacts are verified in
    /// place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DecoderLogitQwen3EngineConfig {
    /// Fill admission defaults around the given profile and model root.
    pub fn new(profile: &'static Qwen3LogitProfile, model_root: impl Into<PathBuf>) -> Self {
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
    fn shard_digest_counts_match_the_shard_lists() {
        for profile in PROFILES {
            assert_eq!(
                profile.checkpoint_shards.len(),
                profile.checkpoint_sha256s.len(),
                "{}",
                profile.loader_id
            );
            assert!(!profile.checkpoint_shards.is_empty());
        }
    }

    #[test]
    fn raw_controls_serve_no_calibration_and_no_knockout() {
        for profile in PROFILES {
            assert_eq!(profile.calibration(), Calibration::Uniform(1.0));
        }
    }

    #[test]
    fn assistant_tails_match_the_reference_templates() {
        assert_eq!(
            AssistantTail::EmptyThinkBlock.as_str(),
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            AssistantTail::BareGenerationPrompt.as_str(),
            "<|im_start|>assistant\n"
        );
    }

    #[test]
    fn lookup_finds_every_profile() {
        for profile in PROFILES {
            assert_eq!(profile_by_loader_id(profile.loader_id), Some(*profile));
        }
        assert_eq!(profile_by_loader_id("missing"), None);
    }
}
