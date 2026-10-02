//! Family: `strands-decider`.
//!
//! The published Strands Decider 2B line (Hobson v19,
//! `github.com/strands-labs/strands-decider`, Apache-2.0): a rank-16 LoRA
//! adapter plus a separate pointer head on the `Qwen/Qwen3.5-2B-Base` hybrid
//! text backbone, trained against the public Jev decision contract. This
//! family is distinct from the `decider` family — that one reads tied
//! output-embedding slot logits off a merged checkpoint, while this readout
//! scores each option from its own hidden state — so per
//! [`NEW_FAMILY`](https://docs.families/NEW_FAMILY.md) it is its own family,
//! not a second `decider` profile.
//!
//! Architecture (per the published reference): a prefill-only causal
//! backbone that never generates text. One question is one causal row — the
//! `<state>` document first, then the `<question>` block with options
//! numbered from 1 and a trailing `<answer>` marker. The pointer head
//! LayerNorms hidden states (torch default epsilon `1e-5`), projects the
//! `<answer>` position as the query and each option's last-token position as
//! keys, and scores `q(h_answer) · k(h_option) / sqrt(256)` in FP32. The
//! option softmax is the answer distribution.
//!
//! Pinned profile artifacts (all verified in place, nothing downloaded at
//! load):
//! - adapter + head + tokenizer + serving config:
//!   `StrandsAgents/strands-decider-2B-hobson-v19` at
//!   `bb282d786bc251fd4e3068de3ada9ddbb38127cd` (Apache-2.0). The head ships
//!   as `head.safetensors`, so no conversion is needed.
//! - base: `Qwen/Qwen3.5-2B-Base` at
//!   `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` (Apache-2.0). The release's
//!   `provenance.json` labels this base revision "inferred"; it is the
//!   repository's current (and only) revision, and this profile pins the
//!   exact bytes it verified.
//!
//! The LoRA adapter is merged into the base weights at load time
//! (`W += (B @ A) · alpha/r`) through a backend view over the memory-mapped
//! checkpoint, so the shared Qwen3.5 layer kernels run untouched and no
//! weight file is ever written.
//!
//! Wire mapping (per the reference `strands_decider` serving code): `Noul`
//! scores the two options `false`/`true` (with the caller's criteria when
//! supplied, the reference defaults otherwise) and reports `p(true)`.
//! `Choice` scores one option per offered key and reports probabilities by
//! key. `Score` scores one option per ordered level (the line shows both the
//! 1-based position and the 0-based level label, exactly as the reference
//! renders it) and reports the expected level. The reference JSON state
//! serialization (`json.dumps(..., indent=2, ensure_ascii=False)`) is
//! reproduced.
//!
//! Declared determinism differences from the reference: Choice options
//! render in the wire's sorted label order (our criteria map is a hash map),
//! and the window budget takes the longest question across the whole request
//! (the reference batches requests in chunks of 32 and re-fits per chunk).

mod engine;
mod model;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::StrandsDeciderEngine;
use crate::families::decoder_logit_qwen35::Calibration;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "strands-decider";
/// Pinned checkpoint repository (adapter, head, tokenizer, serving config).
pub const BACKBONE_ID: &str = "StrandsAgents/strands-decider-2B-hobson-v19";
/// Pinned immutable checkpoint revision.
pub const BACKBONE_REVISION: &str = "bb282d786bc251fd4e3068de3ada9ddbb38127cd";
/// Pinned base-model repository (the checkpoint's `hobson_config.json`
/// `base_model` and `provenance.json` base revision).
pub const BASE_MODEL_ID: &str = "Qwen/Qwen3.5-2B-Base";
/// Pinned immutable base-model revision.
pub const BASE_MODEL_REVISION: &str = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "6a02bb0d1c6b25cae74b";
/// SHA-256 of the pinned base checkpoint
/// (`model.safetensors-00001-of-00001.safetensors`).
pub const BASE_CHECKPOINT_SHA256: &str =
    "928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d";
/// SHA-256 of the pinned base `config.json`.
pub const BASE_CONFIG_SHA256: &str =
    "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4";
/// SHA-256 of the pinned LoRA adapter (`lora/adapter_model.safetensors`).
pub const ADAPTER_SHA256: &str = "701bdb895887097f7954ec7eb06f7937d3b790035b5462195eb26abd80a4aebc";
/// SHA-256 of the pinned adapter config (`lora/adapter_config.json`).
pub const ADAPTER_CONFIG_SHA256: &str =
    "eb48e4ff81569664c4dd2a504b53da598eeaa390c265c2269ce4fe7526eab38a";
/// SHA-256 of the pinned pointer head (`head.safetensors`).
pub const HEAD_SHA256: &str = "daad0727152b6185447cee36f78230c144312feb7b19f02749b19645238b5287";
/// SHA-256 of the pinned tokenizer (`tokenizer.json`).
pub const TOKENIZER_JSON_SHA256: &str =
    "a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86";
/// SHA-256 of the pinned serving config (`hobson_config.json`).
pub const HOBSON_CONFIG_SHA256: &str =
    "2ae86f2ed56975f68e8f2f368104df8ce0ff4b7d9d146dcaf8eec5626d62449d";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen35-text";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this family. The threshold exists so telemetry and evidence artifacts
/// can record acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Frozen total row window (`hobson_config.json`: `"max_length": 4096`).
/// Longer encoded rows fail closed; truncation beyond the fitted policy is
/// forbidden.
pub const MAX_ROW_TOKENS: usize = 4_096;
/// Frozen question reserve: the largest token count a question can claim
/// before the state is squeezed (reference `max_question_fraction` 0.75 of
/// the 4096 window). Questions beyond it are truncated from the front,
/// keeping the tail (options and the `<answer>` marker survive).
pub const MAX_QUESTION_TOKENS: usize = 3_072;
/// LoRA rank of the pinned adapter.
pub(crate) const LORA_RANK: usize = 16;
/// LoRA scale (`lora_alpha / r` = 32 / 16).
pub(crate) const LORA_SCALE: f64 = 2.0;
/// Pointer dimension of the pinned head.
pub(crate) const POINTER_DIM: usize = 256;
/// Width of the 2B residual stream.
pub(crate) const HIDDEN_SIZE: usize = 2_048;
/// Vocabulary of the tied embedding table.
pub(crate) const VOCAB_SIZE: usize = 248_320;
/// LayerNorm epsilon of the pointer head (torch `nn.LayerNorm` default).
pub(crate) const LAYER_NORM_EPSILON: f32 = 1e-5;

/// Per-type calibration temperatures from `hobson_config.json`
/// (`temperature_by_kind`, fitted by the reference `calibrate` command).
pub const CALIBRATION: Calibration = Calibration::ByType {
    choice: 0.734_189_596_436_441,
    score: 1.327_809_421_423_48,
    noul: 0.910_713_699_846_042_8,
};

/// Reference default Noul criteria, rendered when the wire question carries
/// none (`prompting.py`: `NOUL_DEFAULT_CRITERIA`).
pub(crate) const NOUL_DEFAULT_CRITERIA: [&str; 2] = [
    "the statement does not hold for this state",
    "the statement holds for this state",
];

/// Pinned base `config.json` values the loader enforces before trusting
/// weights (dotted paths through the multimodal container's
/// `text_config`).
pub(crate) fn pinned_base_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("architectures[0]", json!("Qwen3_5ForConditionalGeneration")),
        ("text_config.model_type", json!("qwen3_5_text")),
        ("text_config.hidden_size", json!(2_048)),
        ("text_config.intermediate_size", json!(6_144)),
        ("text_config.num_hidden_layers", json!(24)),
        ("text_config.full_attention_interval", json!(4)),
        ("text_config.num_attention_heads", json!(8)),
        ("text_config.num_key_value_heads", json!(2)),
        ("text_config.head_dim", json!(256)),
        ("text_config.linear_num_key_heads", json!(16)),
        ("text_config.linear_num_value_heads", json!(16)),
        ("text_config.linear_key_head_dim", json!(128)),
        ("text_config.linear_value_head_dim", json!(128)),
        ("text_config.linear_conv_kernel_dim", json!(4)),
        ("text_config.vocab_size", json!(248_320)),
        ("text_config.tie_word_embeddings", json!(true)),
        ("text_config.rms_norm_eps", json!(1e-06)),
        ("text_config.attn_output_gate", json!(true)),
        ("text_config.attention_bias", json!(false)),
        ("text_config.dtype", json!("bfloat16")),
        ("text_config.rope_parameters.rope_theta", json!(10_000_000)),
    ]
}

/// Pinned `hobson_config.json` values the loader enforces before trusting
/// weights. These freeze the serving semantics the engine reproduces: a
/// config that changes the head geometry, the adapter rank, the window, or
/// the fitted temperatures fails the load instead of silently changing
/// answers.
pub(crate) fn pinned_hobson_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("base_model", json!("Qwen/Qwen3.5-2B-Base")),
        ("head_type", json!("pointer")),
        ("head_hidden", json!(0)),
        ("pointer_dim", json!(256)),
        ("max_length", json!(4_096)),
        ("torch_dtype", json!("bfloat16")),
        ("use_lora", json!(true)),
        ("lora_r", json!(16)),
        ("lora_alpha", json!(32)),
        ("temperature", json!(0.962_772_160_767_736_2)),
        ("temperature_by_kind.noul", json!(0.910_713_699_846_042_8)),
        ("temperature_by_kind.choice", json!(0.734_189_596_436_441)),
        ("temperature_by_kind.score", json!(1.327_809_421_423_48)),
    ]
}

/// Pinned `lora/adapter_config.json` values the loader enforces before
/// merging adapter weights into the base checkpoint.
pub(crate) fn pinned_adapter_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("peft_type", json!("LORA")),
        ("r", json!(16)),
        ("lora_alpha", json!(32)),
        ("bias", json!("none")),
        ("fan_in_fan_out", json!(false)),
        ("base_model_name_or_path", json!("Qwen/Qwen3.5-2B-Base")),
    ]
}

/// Errors raised while loading or evaluating the pinned strands-decider
/// profile.
#[derive(Debug, Error)]
pub enum StrandsDeciderError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned strands-decider profile.
#[derive(Debug, Clone)]
pub struct StrandsDeciderEngineConfig {
    /// Model root containing the pinned checkpoint artifacts
    /// (`adapter_model.safetensors`, `adapter_config.json`,
    /// `head.safetensors`, `tokenizer.json`, and `hobson_config.json`) at
    /// [`BACKBONE_REVISION`]. Verified in place; nothing is copied or
    /// downloaded.
    pub model_root: PathBuf,
    /// Base-model root containing the pinned `Qwen3.5-2B-Base`
    /// `model.safetensors-00001-of-00001.safetensors` and `config.json` at
    /// [`BASE_MODEL_REVISION`].
    pub base_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl StrandsDeciderEngineConfig {
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

    #[test]
    fn calibration_temperatures_are_finite_and_positive() {
        let temperatures = match CALIBRATION {
            Calibration::ByType {
                choice,
                score,
                noul,
            } => [choice, score, noul],
            other => panic!("unexpected uniform calibration: {other:?}"),
        };
        for temperature in temperatures {
            assert!(temperature.is_finite() && temperature > 0.0);
        }
    }
}
