//! Phase 3B reference contracts, native CPU backbone, and continuation state.

mod batched;
mod branch;
mod contract;
mod embedding;
mod layer0;
mod model;
mod nested;
mod persistence;
mod reference;
mod strategy;
mod types;

#[cfg(test)]
mod batched_tests;
#[cfg(test)]
mod branch_tests;
#[cfg(test)]
mod embedding_tests;
#[cfg(test)]
mod layer0_tests;
#[cfg(test)]
mod nested_tests;
#[cfg(test)]
mod persistence_tests;
#[cfg(test)]
mod strategy_tests;
#[cfg(test)]
mod test_support;

pub use batched::{
    run_batched_candidates, run_batched_nested, run_batched_questions, BatchedCandidateResult,
    BatchedCandidates, BatchedNestedRun, BatchedQuestionResult, BatchedQuestions,
};
pub use branch::Qwen35BranchBatch;
pub use contract::LayerKind;
pub use embedding::{EmbeddingOutput, Qwen35Embedding};
// MLX parity-backend weight loading reuses the Candle oracle's verified
// shard path helper.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) use embedding::verify_decoder_shard;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) use embedding::{CONFIG_SHA256, MODEL_INDEX_SHA256};
pub use layer0::{Layer0Output, Qwen35Layer0};
pub use model::{BackboneOutput, BackboneState, Qwen35Backbone};
pub use nested::{
    run_sequential_nested, NestedCandidateResult, NestedQuestion, NestedQuestionResult, NestedRun,
    SequentialNestedExecutor,
};
pub use reference::BackboneReference;
pub use strategy::{
    choose_strategy, run_repeated_full, run_strategy, run_with_scheduler, CountingExecutor,
    ExecutionStrategy, ProcessMemoryEnvelope, RetentionEstimates, SchedulerConfig,
    StrategyDecision, StrategyEstimates, StrategyOutput, StrategyRequest,
};
pub use types::{FullSequenceRecord, StageComparison, TraceStage};
