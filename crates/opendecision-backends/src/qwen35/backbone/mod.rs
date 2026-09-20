//! Phase 3B reference contracts, native CPU backbone, and continuation state.

mod branch;
mod contract;
mod embedding;
mod layer0;
mod model;
mod reference;
mod types;

#[cfg(test)]
mod branch_tests;

pub use branch::Qwen35BranchBatch;
pub use contract::LayerKind;
pub use embedding::{EmbeddingOutput, Qwen35Embedding};
pub use layer0::{Layer0Output, Qwen35Layer0};
pub use model::{BackboneOutput, BackboneState, Qwen35Backbone};
pub use reference::BackboneReference;
pub use types::{FullSequenceRecord, StageComparison, TraceStage};
