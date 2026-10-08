//! JEV-protocol readout shared by the JEV (Qwen3.5) and GEV (Gemma 4) decision
//! models.
//!
//! Both checkpoints answer a typed question in one forward pass over a
//! `[kind] … [state] … [question] … [options] … [decision]:` prompt and
//! never generate text. This module is the hardware-neutral part of the
//! `mlx-vlm` reference (`mlx_vlm/models/jev/jev.py` on `feat/jev`,
//! `mlx_vlm/models/gev/gev.py` on `feat/gev`): `decision_config` validation,
//! prompt rendering, choice-label token derivation, slot readout with bias
//! and per-kind temperature, and the GEV tournament for more than 16 options.
//!
//! It contains no backbone. A backbone adapter must supply the final-position
//! vocabulary logits (JEV) or head outputs (GEV) and is responsible for the
//! 8-bit affine loading and the Qwen3.5 or Gemma 4 forward. The pinned MLX
//! conversions already have the System 1 LoRA merged (no `base_model.model.*`
//! tensors remain), so the reference's load-time merge is needed only for raw
//! adapter checkpoints; [`DecisionConfig::lora_scale`] serves that case.
//! Nothing here is a `DecisionEngine`, and its presence does not make either
//! profile Rust-loadable.

mod config;
mod engine;
mod layout;
mod pins;
mod prompt;
mod readout;

/// MLX loader for the pinned JEV-27B-VL 8-bit profile.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx_engine;
/// Affine-quantized Qwen3.5 backbone on MLX (feature `mlx`, macOS arm64).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod qwen35_quantized;
#[cfg(all(test, feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod qwen35_quantized_tests;

#[cfg(test)]
mod engine_tests;
#[cfg(test)]
mod tests;

pub use config::{DecisionConfig, DecisionKind, Protocol, MAX_CHOICE_OPTIONS};
pub use engine::{JevEngine, JevForward, BACKEND_ID, MAX_PROMPT_TOKENS};
pub use layout::{
    expected_tensors, ExpectedTensor, JevConfig, QuantParams, Storage, DECODER_PREFIX, LM_HEAD,
    VISION_PREFIX,
};
pub use prompt::{choice_option_text, render_prompt, ChoiceLabels, GEV_BOS, GEV_GROUP_LABELS};
pub use readout::{gev_choice_tournament, GEV_MAX_GROUP};
