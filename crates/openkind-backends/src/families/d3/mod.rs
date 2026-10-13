//! Family: `d3`.
//!
//! The Decision 3.0 family (`vllm-sr/d3` and its five siblings) prompts every
//! question as a chat-template decision block whose options each carry a
//! single-token answer code, runs one **noncausal** forward pass (the causal
//! mask is lifted in the full-attention layers; the gated-DeltaNet layers
//! stay causal by construction), and reads a separate 255-row float32
//! readout matrix at the final prompt position — the `d3-code-readout-v1`
//! contract shipped as `decision_config.json` beside every checkpoint. No
//! output token is ever sampled: the answer is the temperature-calibrated
//! distribution over code logits, restricted to the offered options' codes.
//! `Score` questions render their ordered levels as the option list (code
//! order is level order), exactly as the reference runtime does.
//!
//! Five pinned profiles, smallest first, each reproducing the shipped
//! `d3_runtime.py` (`d3-code-readout-v1`) at scalar temperature 1.0:
//!
//! | Profile | Checkpoint | Backbone |
//! |---|---|---|
//! | `d3-edge` (v3.1.0 release) | `vllm-sr/d3-edge` | pruned 12-layer Qwen3.5-0.8B |
//! | `d3-lite` (v3.1.0 release) | `vllm-sr/d3-lite` | 24-layer Qwen3.5-0.8B |
//! | `d3-nano` (v3.1.0 release) | `vllm-sr/d3-nano` | 24-layer Qwen3.5-2B |
//! | `d3-mini` (v3.1.0 release) | `vllm-sr/d3-mini` | 32-layer Qwen3.5-4B |
//! | `d3-flash` (v3.1.0 release) | `vllm-sr/d3-flash` | 32-layer Qwen3.5-9B |
//!
//! Work limits: every question is one full forward, so each profile caps a
//! question's prompt at [`D3Profile::max_row_tokens`] and the sum over a
//! request's questions at [`D3Profile::max_request_tokens`]. Both are
//! checked while planning, before any forward runs.
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the softmax over
//!   the offered options' code logits sums to one and no semantic-none mass
//!   exists. An offered `__none__` key is scored as an ordinary option.
//! - Continuation state: none — every row is an independent full-sequence
//!   forward and nothing is retained across questions or requests. The
//!   noncausal mask also means a question can never reuse another row's
//!   full-attention KV: `repeated_full` is the only legal strategy.
//! - Text generation: none.
//!
//! Reference contract: `d3_format.py` and `d3_runtime.py` shipped next to
//! the checkpoint (Apache-2.0), whose system prompt, state/question/option
//! rendering, chat-template bytes, code-table derivation, last-position
//! readout, and temperature softmax this family reproduces. Our wire
//! criteria map is a hash map, so Choice options render in the wire's
//! sorted label order — a declared determinism difference, as in the other
//! surveyed families. The vision tower that ships inside the checkpoint is
//! unused: this family serves text questions only.

mod engine;
#[cfg(test)]
mod load_order_tests;
mod model;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::{plan_prompt_ids, D3Engine};
pub use self::renderer::{
    derive_answer_codes, instruction_text, render_prompt, state_text, D3Renderer,
};
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "d3";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen35-text-noncausal";
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
pub struct D3Profile {
    /// Loader/profile name, e.g. `d3-edge`.
    pub loader_id: &'static str,
    /// Pinned checkpoint repository.
    pub backbone_id: &'static str,
    /// Pinned immutable checkpoint revision (the evaluated `v3.1.0` release
    /// commit; the Hub tag of the same name points elsewhere).
    pub backbone_revision: &'static str,
    /// Stable derived profile ID.
    pub profile_id: &'static str,
    /// Frozen backbone geometry (noncausal full attention).
    pub geometry: crate::qwen35::Qwen35Geometry,
    /// Checkpoint shards as `(path, bytes, sha256)` under the model root.
    pub shards: &'static [(&'static str, u64, &'static str)],
    /// SHA-256 of the pinned separate `readout.safetensors` (`weight`:
    /// `[255, hidden]` float32).
    pub readout_sha256: &'static str,
    /// Readout byte size.
    pub readout_bytes: u64,
    /// SHA-256 of the pinned `config.json`.
    pub config_json_sha256: &'static str,
    /// SHA-256 of the pinned `decision_config.json`.
    pub decision_config_sha256: &'static str,
    /// SHA-256 of the pinned `tokenizer.json`.
    pub tokenizer_json_sha256: &'static str,
    /// SHA-256 of the pinned `tokenizer_config.json`.
    pub tokenizer_config_json_sha256: &'static str,
    /// SHA-256 of the pinned `chat_template.jinja`. The renderer
    /// reimplements the fixed `[system, user] + generation-prompt` rendering
    /// in code; this digest pins which template the bytes reproduce.
    pub chat_template_sha256: &'static str,
    /// Scalar temperature from `decision_config.json`.
    pub temperature: f64,
    /// `decision_config.json` `base_model` the loader pins.
    pub declared_base_id: &'static str,
    /// `decision_config.json` base `revision` the loader pins.
    pub declared_base_revision: &'static str,
    /// Width of the code readout (`d3_format.MAX_OPTIONS`).
    pub max_candidates: usize,
    /// Frozen maximum Score levels per question (reference `SCORE_LEVELS`).
    pub max_score_levels: usize,
    /// Frozen maximum rendered row length in tokens. Every question renders
    /// exactly one row, so this is the per-question prompt cap. It is
    /// stricter than the reference's 262,144-token model context (it matches
    /// the decider family's cap) because the scalar CPU attention kernel is
    /// quadratic in it; longer questions fail closed because the reference
    /// never truncates.
    pub max_row_tokens: usize,
    /// Frozen maximum sum of every question's prompt tokens in one request.
    /// Each question is a separate full forward, so this bounds the total
    /// work one request can claim; over-budget requests fail closed during
    /// planning, before any forward runs.
    pub max_request_tokens: usize,
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

/// Pinned `vllm-sr/d3-edge` (the `d3-edge v3.1.0` release evaluated by the
/// Decision Index board and the vLLM semantic-router parity record,
/// Apache-2.0) at `18d61df78684675463af9ae732872478845c5a79`: a pruned
/// 12-layer Qwen3.5-0.8B hybrid fine-tune (noncausal full attention) plus a
/// separate `[255, 1024]` float32 code readout.
pub const D3_EDGE: D3Profile = D3Profile {
    loader_id: "d3-edge",
    backbone_id: "vllm-sr/d3-edge",
    backbone_revision: "18d61df78684675463af9ae732872478845c5a79",
    profile_id: "664f9ead734372935286",
    // The pinned `text_config.layer_types` of the pruned checkpoint: the
    // uniform 3:1 interval is declared but the 12-layer prune keeps an
    // irregular subset.
    geometry: crate::qwen35::Qwen35Geometry {
        full_attention_layers: [
            false, false, true, false, false, true, false, false, true, true, true, true, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false,
        ],
        ..crate::qwen35::Qwen35Geometry::BASE_08B
            .with_causal(false)
            .with_layer_count(12)
    },
    shards: &[(
        "model-00001-of-00001.safetensors",
        1_188_677_968,
        "91bb8618450a7f628f1432bbced76bc19ee7346c93d191a228fdd33c25a57a9d",
    )],
    readout_sha256: "de33d5dc795bbd77168e40bfadc340cca9b9f163b88744c10e1a788f4e166d3d",
    readout_bytes: 1_044_560,
    config_json_sha256: "5e529d9aea2f8d387750acf5b02cc32a47d7e87dc65d6d008fba98e5d15acc4d",
    decision_config_sha256: "30b6095b945d5a93697868afdf9be9123ea892085fbaff3078112c48da20dc6b",
    tokenizer_json_sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    tokenizer_config_json_sha256:
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    chat_template_sha256: "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    // decision_config.json: "temperature": 1.0.
    temperature: 1.0,
    declared_base_id: "Qwen/Qwen3.5-0.8B",
    declared_base_revision: "2fc06364715b967f1860aea9cf38778875588b17",
    max_candidates: 255,
    max_score_levels: 10,
    max_row_tokens: 65_536,
    max_request_tokens: 262_144,
    cpu_backend_id: "d3-edge/cpu-fp32",
    cuda_backend_id: "d3-edge/cuda-fp32",
    release_date: "2026-10-12",
    description: "Pinned d3-edge noncausal 12-layer Qwen3.5-0.8B code-readout decision engine \
                  (Apache-2.0, d3-code-readout-v1); prototype readout, research status",
};

/// Pinned `vllm-sr/d3-lite` (the `d3-lite v3.1.0` release commit pinned by
/// the semantic-router parity record, Apache-2.0) at
/// `a8d25a91ef75552803f7009d25b7bd435991abd9`: the full 24-layer
/// Qwen3.5-0.8B hybrid fine-tune (noncausal full attention on the uniform
/// 3:1 table) plus a separate `[255, 1024]` float32 code readout.
pub const D3_LITE: D3Profile = D3Profile {
    loader_id: "d3-lite",
    backbone_id: "vllm-sr/d3-lite",
    backbone_revision: "a8d25a91ef75552803f7009d25b7bd435991abd9",
    profile_id: "e0dd32dbc16b27546692",
    geometry: crate::qwen35::Qwen35Geometry::BASE_08B.with_causal(false),
    shards: &[(
        "model-00001-of-00001.safetensors",
        1_706_027_688,
        "ca6222dbf2b72899ae45580ed84d26d1f0e69a8de212edda295072000e34a4c2",
    )],
    readout_sha256: "fe11cfac5c79b6a83e35b707581b1260214279bb22b003c215c62719186ef629",
    readout_bytes: 1_044_560,
    config_json_sha256: "a9071195384e1fc2c68f9fe0fae007a22182ab77eb14fe8228a1a24b37d3cee9",
    decision_config_sha256: "30b6095b945d5a93697868afdf9be9123ea892085fbaff3078112c48da20dc6b",
    tokenizer_json_sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    tokenizer_config_json_sha256:
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    chat_template_sha256: "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    temperature: 1.0,
    declared_base_id: "Qwen/Qwen3.5-0.8B",
    declared_base_revision: "2fc06364715b967f1860aea9cf38778875588b17",
    max_candidates: 255,
    max_score_levels: 10,
    max_row_tokens: 65_536,
    max_request_tokens: 262_144,
    cpu_backend_id: "d3-lite/cpu-fp32",
    cuda_backend_id: "d3-lite/cuda-fp32",
    release_date: "2026-10-12",
    description: "Pinned d3-lite noncausal 24-layer Qwen3.5-0.8B code-readout decision engine \
                  (Apache-2.0, d3-code-readout-v1); prototype readout, research status",
};

/// Pinned `vllm-sr/d3-nano` (the `d3-nano v3.1.0` release commit pinned by
/// the semantic-router parity record, Apache-2.0) at
/// `3aeb1c22a5632aac72353c36b3d0111c40e7dbf4`: the full 24-layer
/// Qwen3.5-2B hybrid fine-tune (noncausal full attention on the uniform
/// 3:1 table) plus a separate `[255, 2048]` float32 code readout.
pub const D3_NANO: D3Profile = D3Profile {
    loader_id: "d3-nano",
    backbone_id: "vllm-sr/d3-nano",
    backbone_revision: "3aeb1c22a5632aac72353c36b3d0111c40e7dbf4",
    profile_id: "0198af5b5c8e8835605e",
    geometry: crate::qwen35::Qwen35Geometry::BASE_2B.with_causal(false),
    shards: &[(
        "model.safetensors",
        4_426_555_160,
        "7269b3260f1106f3443bfdb028a76143be2e679db3f29a31f4dfd1d595f03155",
    )],
    readout_sha256: "662e8a26d8016b2d479f1ca66bf64944603316c2b5e074ab7ac3304b8ae045e9",
    readout_bytes: 2_089_040,
    config_json_sha256: "9caa6ab8956c859c3852328dda6322893a18b30826d3916db91d938d6c74ac4f",
    decision_config_sha256: "9175a0f42d5886def9c8429fdf8b0be58a3c97d106b94520e2ac38fe326302bf",
    tokenizer_json_sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    tokenizer_config_json_sha256:
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    chat_template_sha256: "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    temperature: 1.0,
    declared_base_id: "Qwen/Qwen3.5-2B",
    declared_base_revision: "15852e8c16360a2fea060d615a32b45270f8a8fc",
    max_candidates: 255,
    max_score_levels: 10,
    max_row_tokens: 65_536,
    max_request_tokens: 262_144,
    cpu_backend_id: "d3-nano/cpu-fp32",
    cuda_backend_id: "d3-nano/cuda-fp32",
    release_date: "2026-10-12",
    description: "Pinned d3-nano noncausal 24-layer Qwen3.5-2B code-readout decision engine \
                  (Apache-2.0, d3-code-readout-v1); prototype readout, research status",
};

/// Pinned `vllm-sr/d3-mini` (the `d3-mini v3.1.0` release commit pinned by
/// the semantic-router parity record, Apache-2.0) at
/// `a475367de0d8133a3c1a2a0dc12303d89cfa2ab4`: the full 32-layer
/// Qwen3.5-4B hybrid fine-tune (the dims the pinned qwen3.5 geometry
/// serves, noncausal full attention) plus a separate `[255, 2560]`
/// float32 code readout. Its chat template inverts the thinking
/// conditional relative to the smaller siblings, which renders the same
/// empty `<think>` block for the reference's `enable_thinking=False`
/// calls; the per-profile template digest pins which file the bytes
/// reproduce.
pub const D3_MINI: D3Profile = D3Profile {
    loader_id: "d3-mini",
    backbone_id: "vllm-sr/d3-mini",
    backbone_revision: "a475367de0d8133a3c1a2a0dc12303d89cfa2ab4",
    profile_id: "434dbc50d3b581663e52",
    geometry: crate::qwen35::Qwen35Geometry::PINNED.with_causal(false),
    shards: &[
        (
            "model-00001-of-00002.safetensors",
            4_973_332_144,
            "60621cb8c3f90e634b98f0ffaf609159ee2c450630386f09dd5f7fd7938d7598",
        ),
        (
            "model-00002-of-00002.safetensors",
            4_105_283_936,
            "6189569fbe9245030d7ffb0b60a52268b7386e7a85a57aee1274a5178c1cc53d",
        ),
    ],
    readout_sha256: "d570665c1da72cb9fca7ad7b23a25720eb691362694dd69d4c80c17d7656ed55",
    readout_bytes: 2_611_280,
    config_json_sha256: "8f5f404a71ecc53c4ff26fc42cb4c778ffacfb4c9623f5e2d51245166d6b5057",
    decision_config_sha256: "588bc0ac7644479b9adde0498d80b6fb7c576388992564ac67599c05d29add07",
    tokenizer_json_sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    tokenizer_config_json_sha256:
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
    chat_template_sha256: "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    temperature: 1.0,
    declared_base_id: "Qwen/Qwen3.5-4B",
    declared_base_revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    max_candidates: 255,
    max_score_levels: 10,
    max_row_tokens: 65_536,
    max_request_tokens: 262_144,
    cpu_backend_id: "d3-mini/cpu-fp32",
    cuda_backend_id: "d3-mini/cuda-fp32",
    release_date: "2026-10-12",
    description: "Pinned d3-mini noncausal 32-layer Qwen3.5-4B code-readout decision engine \
                  (Apache-2.0, d3-code-readout-v1); prototype readout, research status",
};

/// Pinned `vllm-sr/d3-flash` (the `d3-flash v3.1.0` release commit pinned
/// by the semantic-router parity record, Apache-2.0) at
/// `521525effa79469d899a570726547dd7f00da3af`: the full 32-layer
/// Qwen3.5-9B hybrid fine-tune (the clef-flash dims, noncausal full
/// attention) plus a separate `[255, 4096]` float32 code readout. It
/// shares d3-mini's thinking-inverted chat template.
pub const D3_FLASH: D3Profile = D3Profile {
    loader_id: "d3-flash",
    backbone_id: "vllm-sr/d3-flash",
    backbone_revision: "521525effa79469d899a570726547dd7f00da3af",
    profile_id: "b2a4750f8bf4a1fa58cb",
    geometry: crate::qwen35::Qwen35Geometry::CLEF_FLASH.with_causal(false),
    shards: &[
        (
            "model-00001-of-00004.safetensors",
            4_911_663_824,
            "2c322016b91b1232c0d7628560164560d06d477a3edc3362714635f0e2f5949c",
        ),
        (
            "model-00002-of-00004.safetensors",
            4_987_839_568,
            "f769650fb747d85132fb321335af7c71fcf3ba95fd1b1e806447e891cb4b0546",
        ),
        (
            "model-00003-of-00004.safetensors",
            4_954_726_768,
            "c85a0f2f8df221b7864c74e778d802dc22fe1237fe2fa6ea8b81d7b612800a33",
        ),
        (
            "model-00004-of-00004.safetensors",
            1_931_248_816,
            "d686ef3f6cdeeaad5eedde9d516c29ed1c90a1a322dd002765732869c11fa269",
        ),
    ],
    readout_sha256: "8bbc9a9df4e1b7df85a80a1091977092b938d8d2d2869dfaeb8fd36063cf322f",
    readout_bytes: 4_178_000,
    config_json_sha256: "9c3e6eea217964096c7b032285bab6757720a847fbb9d0a04e551d2b1606c0eb",
    decision_config_sha256: "0269c72ac055d8149a7a778c477780af361f4c1cde3fd0e5ba9af791453ece18",
    tokenizer_json_sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    tokenizer_config_json_sha256:
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
    chat_template_sha256: "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    temperature: 1.0,
    declared_base_id: "Qwen/Qwen3.5-9B",
    declared_base_revision: "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    max_candidates: 255,
    max_score_levels: 10,
    max_row_tokens: 65_536,
    max_request_tokens: 262_144,
    cpu_backend_id: "d3-flash/cpu-fp32",
    cuda_backend_id: "d3-flash/cuda-fp32",
    release_date: "2026-10-12",
    description: "Pinned d3-flash noncausal 32-layer Qwen3.5-9B code-readout decision engine \
                  (Apache-2.0, d3-code-readout-v1); prototype readout, research status",
};

/// Profiles of this family, in catalog order.
pub const PROFILES: &[&D3Profile] = &[&D3_EDGE, &D3_LITE, &D3_NANO, &D3_MINI, &D3_FLASH];

/// Resolve a CLI short name (`edge`, `lite`, ...) to its profile. The short
/// names are the `--d3-aliases` / `--d3-model-roots` entry keys.
pub fn profile_by_short_name(short: &str) -> Option<&'static D3Profile> {
    match short {
        "edge" => Some(&D3_EDGE),
        "lite" => Some(&D3_LITE),
        "nano" => Some(&D3_NANO),
        "mini" => Some(&D3_MINI),
        "flash" => Some(&D3_FLASH),
        _ => None,
    }
}

/// The profile with the given loader ID, if this build pins it.
pub fn profile_by_loader_id(loader_id: &str) -> Option<&'static D3Profile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.loader_id == loader_id)
}

/// The profile with the given derived profile ID, if this build pins it.
pub fn profile_by_id(profile_id: &str) -> Option<&'static D3Profile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.profile_id == profile_id)
}

/// Width of the code readout and of the derived answer-code table
/// (`d3_format.MAX_OPTIONS`): `A..Z` plus the first single-token two-letter
/// uppercase strings, 255 entries.
pub const CODE_TABLE_WIDTH: usize = 255;

/// Vocab size every pinned d3 checkpoint's embedding table must have.
pub(crate) const VOCAB_SIZE: usize = 248_320;

/// Pinned `config.json` values the loader enforces before trusting weights.
pub(crate) fn pinned_config(profile: &D3Profile) -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    let geometry = profile.geometry;
    let layer_types: Vec<serde_json::Value> = geometry.full_attention_layers
        [..geometry.layer_count]
        .iter()
        .map(|full| {
            if *full {
                json!("full_attention")
            } else {
                json!("linear_attention")
            }
        })
        .collect();
    vec![
        ("model_type", json!("qwen3_5")),
        ("architectures", json!(["Qwen3_5Model"])),
        ("text_config.hidden_size", json!(geometry.hidden_size)),
        (
            "text_config.intermediate_size",
            json!(geometry.intermediate_size),
        ),
        ("text_config.num_hidden_layers", json!(geometry.layer_count)),
        (
            "text_config.full_attention_interval",
            json!(geometry.full_attention_interval),
        ),
        (
            "text_config.linear_num_key_heads",
            json!(geometry.key_heads),
        ),
        (
            "text_config.linear_num_value_heads",
            json!(geometry.value_heads),
        ),
        ("text_config.linear_key_head_dim", json!(geometry.head_dim)),
        (
            "text_config.linear_value_head_dim",
            json!(geometry.head_dim),
        ),
        (
            "text_config.linear_conv_kernel_dim",
            json!(geometry.conv_kernel),
        ),
        (
            "text_config.num_attention_heads",
            json!(geometry.attention_heads),
        ),
        ("text_config.num_key_value_heads", json!(geometry.kv_heads)),
        ("text_config.head_dim", json!(geometry.attention_head_dim)),
        ("text_config.rms_norm_eps", json!(geometry.rms_epsilon)),
        ("text_config.vocab_size", json!(VOCAB_SIZE)),
        ("text_config.layer_types", json!(layer_types)),
    ]
}

/// Pinned `decision_config.json` values the loader enforces before trusting
/// weights. These freeze the serving semantics the engine reproduces: a
/// config that changes the format, re-enables causal attention, moves the
/// pooling position, or retunes the temperature fails the load instead of
/// silently changing answers.
pub(crate) fn pinned_decision_config(
    profile: &D3Profile,
) -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("format_id", json!("d3-code-readout-v1")),
        ("format_version", json!(1)),
        ("prompt", json!("d3")),
        ("attention_mode", json!("noncausal_full_attention")),
        ("pooling", json!("last")),
        ("readout_dtype", json!("float32")),
        ("temperature", json!(profile.temperature)),
        ("base_model", json!(profile.declared_base_id)),
        ("revision", json!(profile.declared_base_revision)),
    ]
}

/// Errors raised while loading or evaluating a pinned profile.
#[derive(Debug, Error)]
pub enum D3Error {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for one pinned profile of this family.
#[derive(Debug, Clone)]
pub struct D3EngineConfig {
    /// The pinned profile to load.
    pub profile: &'static D3Profile,
    /// Model root containing the pinned checkpoint shards, `readout.safetensors`,
    /// `config.json`, `decision_config.json`, `tokenizer.json`,
    /// `tokenizer_config.json`, and `chat_template.jinja` at the profile's
    /// [`D3Profile::backbone_revision`]. Artifacts are verified in place;
    /// nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl D3EngineConfig {
    /// Fill admission defaults around the given profile and model root.
    pub fn new(profile: &'static D3Profile, model_root: impl Into<PathBuf>) -> Self {
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
    fn every_profile_is_noncausal_at_the_pinned_geometry() {
        for profile in PROFILES {
            assert!(
                !profile.geometry.causal,
                "{} serves a noncausal checkpoint",
                profile.loader_id
            );
            assert_eq!(profile.geometry.full_attention_interval, 4);
            assert_eq!(profile.temperature, 1.0);
            assert_eq!(profile.max_candidates, CODE_TABLE_WIDTH);
            assert!(profile.max_score_levels <= CODE_TABLE_WIDTH);
        }
    }

    #[test]
    fn limits_stay_inside_the_reference_contract() {
        // The reference serves a 262,144-token model context and never
        // truncates, so a stricter cap only rejects input; it can never
        // change an answer the reference would give.
        const REFERENCE_CONTEXT_TOKENS: usize = 262_144;
        for profile in PROFILES {
            assert_eq!(
                profile.max_row_tokens, 65_536,
                "{}: the per-question cap is the decided 65,536 tokens",
                profile.loader_id
            );
            assert!(
                profile.max_row_tokens < REFERENCE_CONTEXT_TOKENS,
                "{}: the per-question cap must stay stricter than the reference context",
                profile.loader_id
            );
            assert_eq!(
                profile.max_request_tokens, 262_144,
                "{}: the per-request budget is the decided 262,144 tokens",
                profile.loader_id
            );
            assert!(
                profile.max_row_tokens <= profile.max_request_tokens,
                "{}: one maximal question must fit the request budget",
                profile.loader_id
            );
        }
    }

    #[test]
    fn descriptions_match_the_catalog_text() {
        // The catalog and manifest descriptions follow one template; a lost
        // `\` line continuation once left a run of spaces in the string.
        for profile in PROFILES {
            let base = profile
                .declared_base_id
                .strip_prefix("Qwen/")
                .expect("pinned base ids live under Qwen/");
            let expected = format!(
                "Pinned {} noncausal {}-layer {base} code-readout decision engine \
                 (Apache-2.0, d3-code-readout-v1); prototype readout, research status",
                profile.loader_id, profile.geometry.layer_count
            );
            assert_eq!(profile.description, expected, "{}", profile.loader_id);
            assert!(
                !profile.description.contains("  "),
                "{}: description has a run of spaces",
                profile.loader_id
            );
        }
    }
}
