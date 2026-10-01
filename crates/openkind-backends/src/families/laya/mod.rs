//! Family: `laya`.
//!
//! The Laya decision-encoder family (Convai Innovations, Apache-2.0): a
//! ModernBERT-shaped encoder followed by a typed decision head, trained with
//! RLCD proper-scoring-rule rewards. One forward pass scores every option of
//! a question as a `[MASK]`-marked span ahead of the serialized state; the
//! head is shared across question types and only the type embedding, option
//! rendering, and temperature bucket differ. Nothing generates text.
//!
//! Three pinned profiles share this module; they differ only in pinned
//! artifacts, sequence budgets, temperature tables, and encoder dimensions:
//!
//! | Profile | Backbone | Encoder | max_len / head_max_len |
//! |---|---|---|---|
//! | `laya-english` | `convaiinnovations/laya` | ModernBERT-large | 512 / 192 |
//! | `laya-multilingual` | `convaiinnovations/laya-multilingual` | mmBERT-base | 1024 / 256 |
//! | `laya-typed-decisions` | `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 1024 / 256 |
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the marker scorer
//!   returns one logit per offered option span and the reported distribution
//!   is a temperature-scaled softmax over those options only. The reserved
//!   `__none__` key is an ordinary offered candidate when present; the family
//!   models no semantic-none mass of its own, matching the reference.
//! - Wire mapping: `Choice` sorts labels lexicographically for a
//!   deterministic marker order (the reference's readout is order-sensitive)
//!   and renders `label: description` (or the bare label when no
//!   description). `Score` renders `level i: <name>` in level order.
//!   `Noul` renders the fixed `[false, true]` pair with the reference
//!   criteria defaults. Score answers report the expected level index and
//!   Noul answers report P(true).
//! - Calibration: the shipped per-type and per-option-count temperatures
//!   from `rl_agent_config.json` are pinned as-loaded; decode applies them
//!   through the reference's `[0.5, 5.0]` clamp, under which the shipped
//!   `choice:11+` sharpening bucket resolves to 0.5.
//! - Sequence rendering follows the reference exactly: the
//!   `[CLS] <type> question: <instructions> [SEP] ([MASK] + option)… [SEP]
//!   state [SEP]` template, the 48-token option-span cap, the
//!   `head_max_len` budget with per-option cuts, and keep-first state
//!   truncation (keep-newest for array states). Over-long heads that drop
//!   markers fail closed.
//! - Object and array states serialize as compact JSON with Python
//!   `json.dumps` separators (`", "`, `": "`) and sorted keys; the wire
//!   parser's map order is not preserved.
//! - Continuation state: none; every question is one full forward pass.
//! - Excluded: the reference's `act_head` act/escalate head (upstream
//!   documents its output as carrying no usable signal) and the language
//!   router; both are out of scope for the pinned readout.

mod arch;
mod engine;
#[doc(hidden)]
pub mod model;
#[cfg(feature = "onnx")]
#[doc(hidden)]
pub mod onnx;
#[doc(hidden)]
pub mod renderer;

/// MLX/Metal execution backend for the pinned profiles (feature `mlx`).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub use self::mlx::{LayaMlxEngine, LayaMlxEngineConfig, MLX_EXECUTION_ARITHMETIC_ID};

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::LayaEngine;
use crate::families::modernbert::ModernBertConfig;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "laya";

/// Stable derived profile ID for `laya-english`.
pub const ENGLISH_PROFILE_ID: &str = "c8ea29bf1e33a343c4b7";
/// Stable derived profile ID for `laya-multilingual`.
pub const MULTILINGUAL_PROFILE_ID: &str = "f4064eb56fb7f7d325e1";
/// Stable derived profile ID for `laya-typed-decisions`.
pub const TYPED_DECISIONS_PROFILE_ID: &str = "9d28cfa9567902801ed1";

/// Arithmetic/device identity of the family execution path: fp16-stored
/// shards upcast to FP32 on CPU at load.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-laya";

/// Declared probability space of every laya profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Maximum options per question, matching the reference server's per-question
/// choice guard; heads that overflow `head_max_len` fail closed regardless.
pub const MAX_CANDIDATES: usize = 100;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this family. The upstream base checkpoints are near chance zero-shot
/// on typed decisions; the threshold exists so telemetry and evidence
/// artifacts can record acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Question-type ids in the reference's embedding table.
pub(crate) const QTYPE_CHOICE: usize = 0;
/// `score` occupies the second type-embedding row.
pub(crate) const QTYPE_SCORE: usize = 1;
/// `noul` occupies the third type-embedding row.
pub(crate) const QTYPE_NOUL: usize = 2;

/// Special-token ids and the mask token string of one pinned checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct LayaSpecials {
    /// Sequence-start token id (`[CLS]` / `<bos>`).
    pub cls: u32,
    /// Boundary token id (`[SEP]` / `<eos>`).
    pub sep: u32,
    /// Padding token id; unused on the unpadded single-row path.
    pub pad: u32,
    /// Option-marker token id.
    pub mask: u32,
    /// The mask token's surface string, replaced with a space in user text.
    pub mask_token: &'static str,
}

/// Pinned encoder dimensions of one profile.
#[derive(Debug, Clone, Copy)]
pub struct LayaEncoderDims {
    /// Vocabulary size of the pinned tokenizer/encoder pair.
    pub vocab_size: usize,
    /// Hidden size.
    pub hidden_size: usize,
    /// Attention heads (head dimension is `hidden_size / heads`).
    pub num_attention_heads: usize,
    /// Encoder layers.
    pub num_hidden_layers: usize,
    /// GEGLU intermediate size.
    pub intermediate_size: usize,
    /// RoPE base for full-attention layers.
    pub global_rope_theta: f64,
    /// RoPE base for sliding-attention layers.
    pub local_rope_theta: f64,
}

/// Everything pinned about one laya profile.
#[derive(Debug, Clone)]
pub struct LayaProfile {
    /// Loader/profile name, e.g. `laya-english`.
    pub loader_id: &'static str,
    /// Pinned backbone repository.
    pub backbone_id: &'static str,
    /// Pinned immutable backbone revision.
    pub backbone_revision: &'static str,
    /// Stable derived profile ID.
    pub profile_id: &'static str,
    /// SHA-256 of the pinned `model.safetensors`.
    pub checkpoint_sha256: &'static str,
    /// SHA-256 of the pinned `encoder/config.json`.
    pub encoder_config_sha256: &'static str,
    /// SHA-256 of the pinned `tokenizer/tokenizer.json`.
    pub tokenizer_json_sha256: &'static str,
    /// SHA-256 of the pinned `tokenizer/tokenizer_config.json`.
    pub tokenizer_config_sha256: &'static str,
    /// SHA-256 of the pinned `rl_agent_config.json`.
    pub agent_config_sha256: &'static str,
    /// Frozen maximum sequence length (`max_len`).
    pub max_sequence_tokens: usize,
    /// Frozen head budget (`head_max_len`).
    pub head_max_len: usize,
    /// Raw shipped per-type temperatures `[choice, score, noul]`.
    pub temperature: [f64; 3],
    /// Raw shipped per-option-count bucket temperatures, keys like
    /// `"choice:3-5"`.
    pub temperature_by_options: &'static [(&'static str, f64)],
    /// Pinned encoder dimensions.
    pub encoder: LayaEncoderDims,
    /// Pinned special-token ids.
    pub specials: LayaSpecials,
}

/// Pinned `convaiinnovations/laya` (English, ModernBERT-large) at
/// `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (Apache-2.0).
pub const LAYA_ENGLISH: LayaProfile = LayaProfile {
    loader_id: "laya-english",
    backbone_id: "convaiinnovations/laya",
    backbone_revision: "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    profile_id: ENGLISH_PROFILE_ID,
    checkpoint_sha256: "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
    encoder_config_sha256: "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979",
    tokenizer_json_sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    tokenizer_config_sha256: "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12",
    agent_config_sha256: "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
    max_sequence_tokens: 512,
    head_max_len: 192,
    temperature: [
        1.636_903_047_561_645_5,
        1.251_430_034_637_451_2,
        1.983_399_510_383_606,
    ],
    temperature_by_options: &[
        ("choice:2", 1.906_356_334_686_279_3),
        ("choice:3-5", 1.760_151_863_098_144_5),
        ("choice:6-10", 1.000_015_854_835_510_3),
        ("choice:11+", 0.100_582_808_256_149_29),
        ("score:3-5", 1.251_430_034_637_451_2),
        ("noul:2", 1.983_399_510_383_606),
    ],
    encoder: LayaEncoderDims {
        vocab_size: 50_368,
        hidden_size: 1_024,
        num_attention_heads: 16,
        num_hidden_layers: 28,
        intermediate_size: 2_624,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
    },
    specials: LayaSpecials {
        cls: 50_281,
        sep: 50_282,
        pad: 50_283,
        mask: 50_284,
        mask_token: "[MASK]",
    },
};

/// Pinned `convaiinnovations/laya-multilingual` (mmBERT-base) at
/// `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` (Apache-2.0; encoder derives
/// from `jhu-clsp/mmBERT-base`, MIT).
pub const LAYA_MULTILINGUAL: LayaProfile = LayaProfile {
    loader_id: "laya-multilingual",
    backbone_id: "convaiinnovations/laya-multilingual",
    backbone_revision: "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    profile_id: MULTILINGUAL_PROFILE_ID,
    checkpoint_sha256: "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
    encoder_config_sha256: "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4",
    tokenizer_json_sha256: "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
    tokenizer_config_sha256: "424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7",
    agent_config_sha256: "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6",
    max_sequence_tokens: 1_024,
    head_max_len: 256,
    temperature: [1.0, 1.0, 1.0],
    temperature_by_options: &[],
    encoder: LayaEncoderDims {
        vocab_size: 256_000,
        hidden_size: 768,
        num_attention_heads: 12,
        num_hidden_layers: 22,
        intermediate_size: 1_152,
        global_rope_theta: 160_000.0,
        // mmBERT pins both RoPE bases to 160000; the mismatch with the
        // ModernBERT sliding-window default is why the reference carries an
        // explicit rope-config fixup.
        local_rope_theta: 160_000.0,
    },
    specials: LayaSpecials {
        cls: 2,
        sep: 1,
        pad: 0,
        mask: 4,
        mask_token: "<mask>",
    },
};

/// Pinned `convaiinnovations/laya-typed-decisions` (fine-tuned
/// ModernBERT-large) at `1a793eb568e6718f15941d08f85432581df534e3`
/// (Apache-2.0).
pub const LAYA_TYPED_DECISIONS: LayaProfile = LayaProfile {
    loader_id: "laya-typed-decisions",
    backbone_id: "convaiinnovations/laya-typed-decisions",
    backbone_revision: "1a793eb568e6718f15941d08f85432581df534e3",
    profile_id: TYPED_DECISIONS_PROFILE_ID,
    checkpoint_sha256: "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
    encoder_config_sha256: "5268d24ad3b77c8151de5dcb0762ba4391619aad9ab0bda33e36fb083cfeae6d",
    tokenizer_json_sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    tokenizer_config_sha256: "08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1",
    agent_config_sha256: "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8",
    max_sequence_tokens: 1_024,
    head_max_len: 256,
    temperature: [
        1.014_802_455_902_099_6,
        1.037_425_994_873_046_9,
        1.057_512_521_743_774_4,
    ],
    temperature_by_options: &[
        ("choice:2", 1.906_356_334_686_279_3),
        ("choice:3-5", 1.760_151_863_098_144_5),
        ("choice:6-10", 1.000_015_854_835_510_3),
        ("choice:11+", 0.100_582_808_256_149_29),
        ("score:3-5", 1.251_430_034_637_451_2),
        ("noul:2", 1.983_399_510_383_606),
    ],
    encoder: LayaEncoderDims {
        vocab_size: 50_368,
        hidden_size: 1_024,
        num_attention_heads: 16,
        num_hidden_layers: 28,
        intermediate_size: 2_624,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
    },
    specials: LayaSpecials {
        cls: 50_281,
        sep: 50_282,
        pad: 50_283,
        mask: 50_284,
        mask_token: "[MASK]",
    },
};

impl LayaProfile {
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

    /// Pinned `encoder/config.json` values the loader enforces before
    /// trusting weights.
    pub(crate) fn pinned_encoder_config(&self) -> Vec<(&'static str, serde_json::Value)> {
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
            ("tie_word_embeddings", json!(true)),
            ("max_position_embeddings", json!(8192)),
            ("hidden_activation", json!("gelu")),
            // `cls_token_id` is deliberately not pinned: the multilingual
            // checkpoint ships a stale value there (the eos id) and the
            // reference builds sequences from the tokenizer's ids, which
            // `LayaRenderer::load` verifies against the pinned specials.
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

    /// Pinned `rl_agent_config.json` values the loader enforces before
    /// trusting weights.
    pub(crate) fn pinned_agent_config(&self) -> Vec<(&'static str, serde_json::Value)> {
        use serde_json::json;
        let mut fields = vec![
            ("head_layers", json!(2)),
            ("max_len", json!(self.max_sequence_tokens)),
            ("head_max_len", json!(self.head_max_len)),
            ("temperature", json!(self.temperature.map(f64::from))),
        ];
        if self.temperature_by_options.is_empty() {
            fields.push(("temperature_by_options", json!({})));
        } else {
            let mut table = serde_json::Map::new();
            for (bucket, temperature) in self.temperature_by_options {
                table.insert((*bucket).to_owned(), json!(temperature));
            }
            fields.push(("temperature_by_options", serde_json::Value::Object(table)));
        }
        fields
    }
}

/// Errors raised while loading or evaluating a pinned laya profile.
#[derive(Debug, Error)]
pub enum LayaError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
impl From<crate::qwen35::mlx::MlxError> for LayaError {
    fn from(error: crate::qwen35::mlx::MlxError) -> Self {
        Self::Family(crate::families::support::FamilyError::from(error))
    }
}

/// Filesystem configuration for one pinned laya profile.
#[derive(Debug, Clone)]
pub struct LayaEngineConfig {
    /// The pinned profile to load.
    pub profile: &'static LayaProfile,
    /// Model root containing the pinned `model.safetensors`,
    /// `encoder/config.json`, `tokenizer/`, and `rl_agent_config.json` at
    /// the profile's [`LayaProfile::backbone_revision`]. Artifacts are
    /// verified in place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl LayaEngineConfig {
    /// Fill admission defaults around the given model root.
    pub fn new(profile: &'static LayaProfile, model_root: impl Into<PathBuf>) -> Self {
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
        for profile in [&LAYA_ENGLISH, &LAYA_MULTILINGUAL, &LAYA_TYPED_DECISIONS] {
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
    fn candidate_bounds_match_the_reference_guard() {
        assert_eq!(MAX_CANDIDATES, 100);
        for profile in [&LAYA_ENGLISH, &LAYA_MULTILINGUAL, &LAYA_TYPED_DECISIONS] {
            assert!(MAX_CANDIDATES <= profile.max_sequence_tokens);
        }
    }

    #[test]
    fn temperatures_clamp_into_the_reference_range() {
        // Every pinned raw temperature resolves to its clamped value exactly
        // as the reference applies it at load.
        for profile in [&LAYA_ENGLISH, &LAYA_MULTILINGUAL, &LAYA_TYPED_DECISIONS] {
            for temperature in profile.temperature {
                let clamped = engine::clamp_temperature(temperature);
                assert!((0.5..=5.0).contains(&clamped));
            }
            for (_, temperature) in profile.temperature_by_options {
                let clamped = engine::clamp_temperature(*temperature);
                assert!((0.5..=5.0).contains(&clamped));
            }
        }
        // The shipped sharpening bucket resolves to the clamp floor.
        assert_eq!(engine::clamp_temperature(0.100_582_808_256_149_29), 0.5);
    }
}
