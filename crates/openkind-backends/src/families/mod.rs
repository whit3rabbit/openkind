//! Surveyed-family model loaders, readouts, and engine adapters.
//!
//! Each submodule implements one family from
//! [`docs/families`](../../../docs/families/README.md): a pinned offline
//! checkpoint loader, a family readout, and a `DecisionEngine` adapter built
//! on the shared bounded scaffold. Families here declare
//! `ConditionalOnOfferedOptions` probability semantics unless their module
//! documents otherwise, never generate text autoregressively, and never
//! download artifacts.

pub mod calibration;
pub mod decoder_logit_letter;
pub mod decoder_logit_llm;
pub mod decoder_logit_qwen35;
pub mod encoder_instruct_label;
pub mod encoder_nli;
pub mod kev;
pub mod laya;
pub(crate) mod letter_renderer;
pub(crate) mod modernbert;
pub mod qwen3guard;
pub mod router_script;
pub mod schema_scorer;
pub mod support;
pub mod winnow;
pub mod wire;

pub use support::{
    derive_profile_id, temperature_softmax, verify_digest, BoundedFamilyEngine, FamilyControl,
    FamilyError, FamilyEvaluator, FamilyLimits,
};
pub use wire::{answer_from_probabilities, unpack_question, RESERVED_NONE_OPTION};
