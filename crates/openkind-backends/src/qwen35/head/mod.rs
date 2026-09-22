//! Fitted head readout evaluation for the Qwen 3.5 backbone.

mod evaluation;
mod math;
mod tensors;
mod types;

#[cfg(test)]
mod tests;

pub use evaluation::ScoreSummaryHead;
pub use types::{HeadEvaluation, PolicyAction, PrimitiveKind, FEATURE_WIDTH};
