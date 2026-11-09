//! Family: `drex`.
//!
//! Nace.AI's open Drex decision models (`github.com/nace-ai/drex-decision-models`,
//! report arXiv:2611.07854). This module pins Drex v1.5: a merged,
//! full-parameter fine-tune of a Qwen3.5 9B hybrid text backbone
//! (`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`) with the Kev pointer head.
//! Drex v2.0 (MoE), Drex DLM (bidirectional encoder), and Drex Max (27B
//! code readout) are surveyed only; see `docs/families/drex.md`.
//!
//! Architecture (per the bundled reference `code/kev/`): a prefill-only
//! causal backbone that never generates text. Hybrid backbones run the
//! reference's row form, so every question is one causal row of the state
//! document followed by that question's branch, and the reference itself
//! refuses `state_bidir` and option isolation on hybrid backbones. The prompt
//! layout is the kev family's (`<|fim_prefix|>` state, `<|fim_middle|>`
//! instruction, `<|box_start|>`/`<|box_end|>` options, `<|fim_suffix|>`
//! decide), reused through the kev renderer under Drex's own token limits.
//! The pointer head scores `k(h_opt) · q(h_decide) / sqrt(256)` with biased
//! FP32 projections and no normalization, and the option softmax is the
//! answer distribution.
//!
//! Pinned artifacts (all verified in place, nothing downloaded at load):
//! `nace-ai/drex-v1.5` at [`BACKBONE_REVISION`]. The head ships only as the
//! torch pickle `head.pt`. It is read after its SHA-256 is verified, through
//! candle's pickle reader, which interprets the opcode stream without
//! importing or calling anything. No converted head is required, so no Drex
//! file has to be mirrored.
//!
//! License: the weights are under the modified "Nace AI Open RAIL-M"
//! license, verified by digest at load. Project policy
//! (`docs/RESEARCH.md`, "License policy") is inference-only: no mirroring,
//! a license notice at pull, and no distillation or teacher use of Drex
//! output. [`TEACHER_USE_PERMITTED`] records the last rule for callers.
//!
//! Declared determinism difference from the reference: Choice options render
//! in the wire's sorted label order (the reference keeps caller order).

mod engine;
mod model;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub use self::mlx::{
    DrexMlxEngine, DrexMlxEngineConfig, MLX_BACKEND_ID, MLX_EXECUTION_ARITHMETIC_ID,
};

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::DrexEngine;
use crate::families::support::FamilyLimits;
use crate::qwen35::Qwen35Geometry;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "drex";
/// Loader name of the pinned profile.
pub const LOADER_ID: &str = "drex-v15";
/// Pinned checkpoint repository (merged backbone, head, tokenizer).
pub const BACKBONE_ID: &str = "nace-ai/drex-v1.5";
/// Pinned immutable checkpoint revision.
pub const BACKBONE_REVISION: &str = "70a8a22d9b36bc91977efe296c3f59c9f756e6c2";
/// Base model the checkpoint fine-tunes, per its model card and `config.json`.
pub const BASE_MODEL_ID: &str = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "2d495e07ea056d6c4317";
/// Arithmetic/device identity of the family execution path: BF16 weights
/// widened per use, FP32 compute.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-bf16w-fp32c-qwen35-text";
/// Backend id reported by the CPU engine.
pub const CPU_BACKEND_ID: &str = "drex-v15/cpu-bf16w-fp32c";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// License name shown to operators.
pub const LICENSE_NAME: &str = "Nace AI Open RAIL-M License (Modified), version 1.0";
/// License text at the pinned revision.
pub const LICENSE_URL: &str = "https://huggingface.co/nace-ai/drex-v1.5/blob/70a8a22d9b36bc91977efe296c3f59c9f756e6c2/LICENSE";
/// SHA-256 of the pinned `LICENSE`.
pub const LICENSE_SHA256: &str = "3ee4dcae9750be4be007654b3ce3ea551743b1fb39b9a4bae280db94e1b2e3e9";
/// Whether Drex output may train another model (distillation, teacher use).
///
/// `false`: the license counts models trained on Drex output as
/// derivatives and forbids using output to create them. Any component that
/// selects a teacher engine must refuse this family.
pub const TEACHER_USE_PERMITTED: bool = false;

/// Whether a resolved model name identifies a Drex model, whose output must
/// never become training data.
///
/// The proxy cache only sees the upstream teacher's reported model name, so
/// this matches the family slug in it, case-insensitively (`drex-v1.5-native`,
/// `drex-v15:<profile>`, `nace-ai/drex-v2.0`). An operator who serves Drex
/// under an alias without `drex` in it defeats the check; the license
/// obligation stays with that operator.
pub fn forbids_teacher_use(model_name: &str) -> bool {
    !TEACHER_USE_PERMITTED && model_name.to_ascii_lowercase().contains(FAMILY_SLUG)
}

/// Pinned checkpoint shards: `(file name, SHA-256, size in bytes)`.
pub const CHECKPOINT_SHARDS: [(&str, &str, u64); 4] = [
    (
        "model-00001-of-00004.safetensors",
        "1d2a5ffedb5923c930489f58d14197cfa054b403b3589ea87961f0a48fec502c",
        4_942_706_120,
    ),
    (
        "model-00002-of-00004.safetensors",
        "1121fb2144c38ed6612b6f7cbfa183fe31efec33dca8cb5a69fc88bfeae8982f",
        4_987_757_928,
    ),
    (
        "model-00003-of-00004.safetensors",
        "b144d0c842fb783557ca0e819850ee302772f75bfdc9e24d955643996a65d3c6",
        4_954_810_240,
    ),
    (
        "model-00004-of-00004.safetensors",
        "bd7b4f1d63756ac0fae48fcc054b76ec32d66e893b71dfde6220a95f4e509e6c",
        3_022_388_120,
    ),
];
/// SHA-256 of the pinned `config.json`.
pub const CONFIG_SHA256: &str = "4a9248516d271480884c80271eb8023757bd313932b07facd4cd6a820737ae83";
/// SHA-256 of the pinned `model.safetensors.index.json`.
pub const INDEX_SHA256: &str = "e8ef36c3e0c658f657e282fb6d68609770f039ef2886b89f8ca6a1a293422408";
/// SHA-256 of the pinned `tokenizer.json`.
pub const TOKENIZER_JSON_SHA256: &str =
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";
/// SHA-256 of the pinned `head.pt` (torch pickle; read without execution).
pub const HEAD_PT_SHA256: &str = "9c9fe6c3b81f997e1813d4adc0f94b720afd317ecc8ce2f690edf56614755881";
/// Size of the pinned `head.pt` in bytes.
pub const HEAD_PT_BYTES: u64 = 8_393_151;

/// Frozen maximum state length in tokens, including the state delimiter
/// (the reference serving default `KEV_CONTEXT` of 16,384). Longer states
/// fail closed; truncation is forbidden.
pub const MAX_STATE_TOKENS: usize = 16_384;
/// Frozen maximum row length (state plus one question branch), the same
/// reference serving default.
pub const MAX_ROW_TOKENS: usize = 16_384;

/// Calibration temperature applied to the pointer logits: the `temperature`
/// recorded in the pinned `head.pt` (raw logits).
pub const CALIBRATION_TEMPERATURE: f64 = 1.0;

/// Text-backbone geometry: the Qwen3.5 9B hybrid dimensions the clef-flash
/// oracle already executes (32 layers, hidden 4096, 3:1 DeltaNet table).
pub(crate) const GEOMETRY: Qwen35Geometry = Qwen35Geometry::CLEF_FLASH;
/// Pointer dimension of the pinned head.
pub(crate) const POINTER_DIM: usize = 256;
/// Vocabulary of the embedding table.
pub(crate) const VOCAB_SIZE: usize = 248_320;

/// Pinned `config.json` values the loader enforces before trusting weights.
/// The checkpoint is a flat text-only `Qwen3_5ForCausalLM` config.
pub(crate) fn pinned_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("architectures", json!(["Qwen3_5ForCausalLM"])),
        ("model_type", json!("qwen3_5_text")),
        ("hidden_size", json!(4_096)),
        ("intermediate_size", json!(12_288)),
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
        ("rms_norm_eps", json!(1e-06)),
        ("attn_output_gate", json!(true)),
        ("attention_bias", json!(false)),
        ("partial_rotary_factor", json!(0.25)),
        ("dtype", json!("bfloat16")),
    ]
}

/// Errors raised while loading or evaluating the pinned Drex profile.
#[derive(Debug, Error)]
pub enum DrexError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
impl From<crate::qwen35::mlx::MlxError> for DrexError {
    fn from(error: crate::qwen35::mlx::MlxError) -> Self {
        Self::Family(crate::families::support::FamilyError::from(error))
    }
}

/// Filesystem configuration for the pinned Drex profile.
#[derive(Debug, Clone)]
pub struct DrexEngineConfig {
    /// Model root holding the pinned checkpoint at [`BACKBONE_REVISION`]:
    /// the four shards, `model.safetensors.index.json`, `config.json`,
    /// `tokenizer.json`, `head.pt`, and `LICENSE`. Verified in place;
    /// nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl DrexEngineConfig {
    /// Fill admission defaults around the given root. A 9B CPU row takes
    /// seconds, so the queue is short and the deadline generous.
    pub fn new(model_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 2,
                retry_after_ms: 2_000,
                evaluation_timeout: Some(Duration::from_secs(1_800)),
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
    fn teacher_use_is_refused() {
        // The license forbids training on Drex output; see the module docs.
        const { assert!(!TEACHER_USE_PERMITTED) };
    }

    #[test]
    fn drex_model_names_are_refused_as_teachers() {
        for name in [
            "drex-v1.5-native",
            "drex-v15:2d495e07ea056d6c4317",
            "nace-ai/Drex-Max-v1",
        ] {
            assert!(forbids_teacher_use(name), "{name}");
        }
        for name in ["jev-1.14.0", "kev:39d88c11faeb4ac165fa", "d3-flash"] {
            assert!(!forbids_teacher_use(name), "{name}");
        }
    }

    #[test]
    fn geometry_matches_the_pinned_config() {
        let config: std::collections::HashMap<_, _> = pinned_config().into_iter().collect();
        assert_eq!(config["hidden_size"], GEOMETRY.hidden_size);
        assert_eq!(config["intermediate_size"], GEOMETRY.intermediate_size);
        assert_eq!(config["num_hidden_layers"], GEOMETRY.layer_count);
        assert_eq!(config["num_attention_heads"], GEOMETRY.attention_heads);
        assert_eq!(config["num_key_value_heads"], GEOMETRY.kv_heads);
        assert_eq!(config["linear_num_value_heads"], GEOMETRY.value_heads);
        assert_eq!(config["vocab_size"], VOCAB_SIZE);
        // The reference refuses bidirectional state on hybrid backends.
        const { assert!(GEOMETRY.causal) };
    }

    #[test]
    fn shard_sizes_sum_to_the_index_total() {
        // `model.safetensors.index.json` metadata: total_size 17,907,606,528
        // bytes of tensor data; shard files add their headers on top.
        let on_disk: u64 = CHECKPOINT_SHARDS.iter().map(|(_, _, bytes)| bytes).sum();
        assert!(on_disk > 17_907_606_528);
        assert!(on_disk < 17_907_606_528 + 64 * 1024 * 1024);
    }
}
