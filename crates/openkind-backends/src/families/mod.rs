//! Surveyed-family model loaders, readouts, and engine adapters.
//!
//! Each submodule implements one family from
//! [`docs/families`](../../../docs/families/README.md): a pinned offline
//! checkpoint loader, a family readout, and a `DecisionEngine` adapter built
//! on the shared bounded scaffold. Families here declare
//! `ConditionalOnOfferedOptions` probability semantics unless their module
//! documents otherwise, never generate text autoregressively, and never
//! download artifacts.

/// Temperature calibration datasets and Platt scaling utilities.
pub mod calibration;
/// Cloudflare Clef joint-schema decision family adapter.
pub mod clef;
/// Decider slot-logit family adapter (plain state-first layout).
pub mod decider;
/// Gemma 4 backbone letter-logit decision family adapter.
pub mod gemma4;
/// Decoder-only letter-token logit readout family adapter.
pub mod decoder_logit_letter;
/// Decoder-only LLM logit readout family adapter.
pub mod decoder_logit_llm;
/// Raw dense-Qwen3 letter-logit control family adapter.
pub mod decoder_logit_qwen3;
/// Qwen 3.5 decoder-only logit readout family adapter.
pub mod decoder_logit_qwen35;
/// ModernBERT encoder instruct-label readout family adapter.
pub mod encoder_instruct_label;
/// Encoder natural language inference (NLI) readout family adapter.
pub mod encoder_nli;
/// KEV family model loader and decision engine adapter.
pub mod kev;
/// Laya encoder decision readout family adapter.
pub mod laya;
pub(crate) mod letter_renderer;
pub(crate) mod modernbert;
/// Qwen3Guard safety and classification family adapter.
pub mod qwen3guard;
/// Router-script dynamic decision family adapter.
pub mod router_script;
/// Schema scorer decision family adapter.
pub mod schema_scorer;
/// Shared bounded family engine scaffold, limits, controls, and error types.
pub mod support;
/// Von option-marker encoder decision readout family adapter.
pub mod von;
/// Winnow candidate-filtering decision family adapter.
pub mod winnow;
/// Jev wire conversion utilities and response assembly helpers.
pub mod wire;

/// Shared MLX ModernBERT encoder body (feature `mlx`, macOS arm64).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod mlx_modernbert;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
impl From<crate::qwen35::mlx::MlxError> for FamilyError {
    fn from(error: crate::qwen35::mlx::MlxError) -> Self {
        FamilyError::Mlx(error.to_string())
    }
}

pub use support::{
    derive_profile_id, temperature_softmax, verify_digest, BoundedFamilyEngine, FamilyControl,
    FamilyError, FamilyEvaluator, FamilyLimits,
};
pub use wire::{answer_from_probabilities, unpack_question, RESERVED_NONE_OPTION};
