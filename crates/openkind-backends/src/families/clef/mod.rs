//! Family: `clef`.
//!
//! Cloudflare's Clef decision models — `Cloudflare/clef` (27B) and
//! `Cloudflare/clef-flash` (9B) — are post-trained Qwen3.5 hybrid backbones
//! with a joint schema head that reads the backbone's final-norm hidden
//! states and scores every allowed option of every question in a single
//! forward pass. No token is ever sampled; the answer is the softmax over
//! the head's per-option logits. The reference serving layer is the model
//! card's `joint_schema_model.py` (Apache-2.0), whose record encoding,
//! joint-head math, and SystemOne answer mapping this family reproduces.
//!
//! Profiles:
//!
//! | Profile | Checkpoint | Execution |
//! |---|---|---|
//! | `clef-flash` | `Cloudflare/clef-flash` BF16 safetensors | candle CPU, BF16 weights with FP32 compute |
//! | `clef-flash-gguf` | `bartowski/Cloudflare_clef-flash-GGUF` Q4_K_M + official joint head | candle CPU quantized runner |
//! | `clef-flash-mlx-4bit` | `mlx-community/clef-flash-4bit` | MLX/Metal quantized |
//! | `clef-27b-gguf` | `bartowski/Cloudflare_clef-GGUF` Q4_K_M + official joint head | candle CPU quantized runner |
//!
//! Profile semantics:
//! - Probability space:
//!   [`ProbabilitySpace::ConditionalOnOfferedOptions`] — the head scores the
//!   options the request offers; the distribution over offered options sums
//!   to one and no semantic-none mass exists. An offered `__none__` key is
//!   scored as an ordinary option.
//! - Continuation state: none — every request is an independent
//!   full-sequence forward and nothing is retained across questions.
//! - Text generation: none. Vision: none — image and video inputs are
//!   outside the text decision surface this family exposes.
//!
//! Declared renderer differences from the reference `encode_record`: the
//! Jev wire question map is unordered, so fields render in sorted-key order
//! (the Python reference preserves JSON insertion order), and noul option
//! descriptions come from the caller's wire criteria rather than the
//! Clef-internal default strings. See `renderer.rs`.

mod engine;
mod gguf;
mod head;
mod model;
mod renderer;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx;

use openkind_engine::ProbabilitySpace;

pub use self::engine::ClefEngine;
pub use self::head::LexicalLookup;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "clef";

/// Declared probability space of every profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Provisional application-policy threshold recorded with every profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this family. The threshold exists so telemetry and evidence artifacts
/// can record acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Joint-head geometry beyond the backbone width. The two Clef releases
/// share every field except the backbone width (4096 flash, 5120 27B).
const fn joint_head_config(hidden_size: usize) -> head::JointHeadConfig {
    head::JointHeadConfig {
        hidden_size,
        width: 1_024,
        routing_layers: 2,
        layers: 4,
        heads: 16,
        feedforward: 4_096,
    }
}

/// Execution backend of one profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClefExecution {
    /// Candle BF16 safetensors with FP32 compute (the correctness oracle).
    CandleBf16,
    /// Candle quantized runner over a llama.cpp GGUF backbone.
    CandleGguf,
    /// MLX/Metal quantized execution (feature `mlx`).
    Mlx4Bit,
}

/// Pinned artifacts of one profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClefProfile {
    /// Loader identifier (`<loader-id>:<profile-id>` on the wire).
    pub loader_id: &'static str,
    /// HuggingFace repository carrying the backbone checkpoint.
    pub backbone_id: &'static str,
    /// Frozen repository revision of the backbone checkpoint.
    pub backbone_revision: &'static str,
    /// Derived 20-hex profile ID.
    pub profile_id: &'static str,
    /// Pinned checkpoint artifact paths (relative to the model root) with
    /// their SHA-256 digests and byte lengths.
    pub checkpoint_shards: &'static [(&'static str, &'static str, u64)],
    /// Pinned joint-head checkpoint.
    pub joint_head: (&'static str, &'static str, u64),
    /// Pinned tokenizer JSON.
    pub tokenizer_json_sha256: &'static str,
    /// Pinned config JSON (digest-verified contract enforcement).
    pub config_json_sha256: &'static str,
    /// Tokenizer JSON artifact path.
    pub tokenizer_path: &'static str,
    /// Config JSON artifact path.
    pub config_path: &'static str,
    /// Backbone geometry of this profile.
    pub geometry: crate::qwen35::Qwen35Geometry,
    /// Joint-head configuration of this profile.
    pub joint_head_config: head::JointHeadConfig,
    /// Backend id of the candle CPU execution path.
    pub cpu_backend_id: &'static str,
    /// Execution backend this profile loads.
    pub execution: ClefExecution,
    /// Public release date of the pinned revision (RFC 3339 date).
    pub release_date: &'static str,
    /// One-line description surfaced through the registry.
    pub description: &'static str,
}

impl ClefProfile {
    /// Checkpoint artifact names in pinned order.
    pub fn shard_names(&self) -> impl Iterator<Item = &'static str> {
        self.checkpoint_shards.iter().map(|(name, _, _)| *name)
    }
}

/// `Cloudflare/clef-flash` BF16 profile (candle CPU correctness oracle).
pub static CLEF_FLASH: ClefProfile = ClefProfile {
    loader_id: "clef-flash",
    backbone_id: "Cloudflare/clef-flash",
    backbone_revision: "17f0b0ad64efb65d273590632833508766b2aae6",
    profile_id: "dfe12a21a5c9dd5b2fb1",
    checkpoint_shards: &[
        (
            "model-00001-of-00004.safetensors",
            "8b45a8e968141cdcc58fb71c9adfc258e2c77b5f062bc636c1fd5bc5d916b565",
            4_942_706_120,
        ),
        (
            "model-00002-of-00004.safetensors",
            "7590856c713eed844a2dcf48e6c43c4de165b788bc3f80e328311183cdbc7db8",
            4_987_757_928,
        ),
        (
            "model-00003-of-00004.safetensors",
            "e6eac2467952c33361ed7dcb3c7959d1086bbe57201cd3749c3d769fdc17fe63",
            4_954_810_240,
        ),
        (
            "model-00004-of-00004.safetensors",
            "9fcecc6556b39171238373a465f409794b7f821fb4cd1e6459e3a9c0fe317af7",
            3_934_446_832,
        ),
    ],
    joint_head: (
        "joint_head.safetensors",
        "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
        243_538_016,
    ),
    tokenizer_json_sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    config_json_sha256: "66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238",
    tokenizer_path: "tokenizer.json",
    config_path: "config.json",
    geometry: crate::qwen35::Qwen35Geometry::CLEF_FLASH,
    joint_head_config: joint_head_config(4_096),
    cpu_backend_id: "clef-flash/cpu-bf16w-fp32c",
    execution: ClefExecution::CandleBf16,
    release_date: "2026-10-01",
    description: "Cloudflare Clef-Flash 9B joint-schema decision model (BF16 oracle).",
};

/// `bartowski/Cloudflare_clef-flash-GGUF` Q4_K_M profile plus the official
/// BF16 joint head.
pub static CLEF_FLASH_GGUF: ClefProfile = ClefProfile {
    loader_id: "clef-flash-gguf",
    backbone_id: "bartowski/Cloudflare_clef-flash-GGUF",
    backbone_revision: "d7f376ea88c05e7bb1014dd5351a93df9dd8029e",
    profile_id: "c330d9ee7e9cc658ad45",
    checkpoint_shards: &[(
        "Cloudflare_clef-flash-Q4_K_M.gguf",
        "45f803cbcb6144784653bc31cde957e0d184d963a5198d423dc589a79e178d45",
        5_841_052_992,
    )],
    joint_head: CLEF_FLASH.joint_head,
    tokenizer_json_sha256: CLEF_FLASH.tokenizer_json_sha256,
    config_json_sha256: CLEF_FLASH.config_json_sha256,
    tokenizer_path: CLEF_FLASH.tokenizer_path,
    config_path: CLEF_FLASH.config_path,
    geometry: crate::qwen35::Qwen35Geometry::CLEF_FLASH,
    joint_head_config: joint_head_config(4_096),
    cpu_backend_id: "clef-flash-gguf/cpu-q4km",
    execution: ClefExecution::CandleGguf,
    release_date: CLEF_FLASH.release_date,
    description: "Clef-Flash 9B Q4_K_M GGUF backbone with the official joint head.",
};

/// `mlx-community/clef-flash-4bit` profile (MLX/Metal quantized execution).
pub static CLEF_FLASH_MLX_4BIT: ClefProfile = ClefProfile {
    loader_id: "clef-flash-mlx-4bit",
    backbone_id: "mlx-community/clef-flash-4bit",
    backbone_revision: "7cec35c6ddd2cc6a18f1644399c361024ec7b863",
    profile_id: "0fb395e836f1c22a3fe3",
    checkpoint_shards: &[
        (
            "model-00001-of-00002.safetensors",
            "39894f085d844004c8e5c2394e45b5ab66bce4637ecdf23d92bf7c85cb66238f",
            5_349_769_710,
        ),
        (
            "model-00002-of-00002.safetensors",
            "40258d3d4f7225a0acbb45f4d34693465b98c8e2c0a4b0a512e536750379fcf3",
            600_449_850,
        ),
    ],
    joint_head: CLEF_FLASH.joint_head,
    tokenizer_json_sha256: "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
    config_json_sha256: "c323f36eb35fb9349b2bde60b3e16ee3d86316e20a5166c91f4a0585286f0ec1",
    tokenizer_path: "tokenizer.json",
    config_path: "config.json",
    geometry: crate::qwen35::Qwen35Geometry::CLEF_FLASH,
    joint_head_config: joint_head_config(4_096),
    cpu_backend_id: "clef-flash-mlx-4bit/mlx-q4",
    execution: ClefExecution::Mlx4Bit,
    release_date: CLEF_FLASH.release_date,
    description: "Clef-Flash 9B 4-bit MLX quantization with the official joint head.",
};

/// `bartowski/Cloudflare_clef-GGUF` Q4_K_M profile (27B) plus the official
/// BF16 joint head.
pub static CLEF_27B_GGUF: ClefProfile = ClefProfile {
    loader_id: "clef-27b-gguf",
    backbone_id: "bartowski/Cloudflare_clef-GGUF",
    backbone_revision: "e306f00c6c85da175dfb8de952ebb872087426a7",
    profile_id: "48cb5634b4a258de5a6b",
    checkpoint_shards: &[(
        "Cloudflare_clef-Q4_K_M.gguf",
        "6a03997c1fe1b22580d15b540535f61b3da79e261766cac7104febc8e4651849",
        17_203_416_256,
    )],
    joint_head: (
        "joint_head.safetensors",
        "a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953",
        256_125_024,
    ),
    tokenizer_json_sha256: CLEF_FLASH.tokenizer_json_sha256,
    config_json_sha256: "c42e88892bd3fd84e8276b2ad90df58c1c3b797676ea161035006a72ad468c58",
    tokenizer_path: CLEF_FLASH.tokenizer_path,
    config_path: CLEF_FLASH.config_path,
    geometry: crate::qwen35::Qwen35Geometry::CLEF,
    joint_head_config: joint_head_config(5_120),
    cpu_backend_id: "clef-27b-gguf/cpu-q4km",
    execution: ClefExecution::CandleGguf,
    release_date: CLEF_FLASH.release_date,
    description: "Cloudflare Clef 27B Q4_K_M GGUF backbone with the official joint head.",
};

/// Every loadable profile of this family.
///
/// `CLEF_FLASH_MLX_4BIT` is intentionally absent: its MLX execution path
/// runs end-to-end but has not established joint-head parity with the CPU
/// oracle (the quantized-input DeltaNet state diverges; see the family
/// page's open items). It is promoted here once parity fixtures pass.
pub static PROFILES: &[&ClefProfile] = &[&CLEF_FLASH, &CLEF_FLASH_GGUF, &CLEF_27B_GGUF];

/// Resolve a profile by its loader identifier.
pub fn profile_by_loader_id(loader_id: &str) -> Option<&'static ClefProfile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.loader_id == loader_id)
}

/// Error type of the Clef family.
#[derive(Debug, thiserror::Error)]
pub enum ClefError {
    /// A pinned contract field did not match the loaded artifact.
    #[error("reference contract mismatch for `{field}`: expected `{expected}`, found `{actual}`")]
    ContractMismatch {
        /// Name of the mismatched field.
        field: &'static str,
        /// Required pinned value.
        expected: String,
        /// Observed artifact value.
        actual: String,
    },
}

/// Default admission settings for a Clef engine.
pub(crate) fn default_limits() -> FamilyLimits {
    FamilyLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 8,
        retry_after_ms: 250,
        evaluation_timeout: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_ids_match_the_derivation() {
        for profile in PROFILES {
            assert_eq!(
                profile.profile_id,
                crate::families::support::derive_profile_id(
                    FAMILY_SLUG,
                    profile.backbone_id,
                    profile.backbone_revision,
                ),
                "profile {} id derivation",
                profile.loader_id,
            );
        }
    }

    #[test]
    fn loader_ids_are_unique() {
        let mut ids: Vec<_> = PROFILES.iter().map(|p| p.loader_id).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count);
    }
}
