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
mod prompt;
mod readout;

#[cfg(test)]
mod tests;

pub use config::{DecisionConfig, DecisionKind, Protocol, MAX_CHOICE_OPTIONS};
pub use prompt::{choice_option_text, render_prompt, ChoiceLabels, GEV_BOS, GEV_GROUP_LABELS};
pub use readout::{gev_choice_tournament, GEV_MAX_GROUP};
