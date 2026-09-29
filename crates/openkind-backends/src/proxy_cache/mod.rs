//! Proxy-cache subsystem: a distilling cache in front of a remote Jev API.
//!
//! The server's proxy option forwards `/v1/systemone` traffic to an upstream
//! Jev-compatible API and records (request embedding, teacher distribution)
//! rows. Per task — one (tenant, model, instructions, criteria) tuple — a
//! multinomial logistic-regression student learns the teacher's soft
//! targets over frozen request embeddings. Requests are answered locally
//! only when the student's confidence passes a threshold calibrated with a
//! Clopper–Pearson upper bound on the disagreement rate and a kNN
//! out-of-distribution gate passes; everything else goes upstream.
//!
//! Module map:
//! - [`task`]: task identity, configuration knobs, routing reasons
//! - [`text`]: canonical request text for embedding and hashing
//! - [`encoder`]: the `TextEmbedder` contract and the dependency-free hash embedder
//! - [`bert_encoder`]: candle CPU embedder for BERT-family checkpoints (bge)
//! - [`student`]: linear student, full-batch Adam fit with early stopping
//! - [`ood`]: kNN cosine-distance gate with leave-one-out threshold
//! - [`calibrate`]: threshold grid, Clopper–Pearson bound, policy fitting
//! - [`store`]: per-task SQLite sample and event store
//! - [`registry`]: immutable `student-vN` version directories on disk
//! - [`engine`]: the per-task lifecycle loop
//! - [`manager`]: task registry, admission, and the background training worker

pub mod calibrate;
pub mod encoder;
pub mod engine;
pub mod manager;
pub mod ood;
pub mod registry;
pub mod store;
pub mod student;
pub mod task;
pub mod text;

/// candle CPU embedder for BERT-family sentence-transformer checkpoints.
pub mod bert_encoder;

/// MLX embedder for BERT-family checkpoints (feature `mlx`, macOS arm64).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx_bert_encoder;

use thiserror::Error;

/// Failures of the proxy-cache subsystem.
#[derive(Debug, Error)]
pub enum ProxyCacheError {
    /// The encoder failed to produce an embedding.
    #[error("proxy-cache encoder failure: {0}")]
    Encoder(String),
    /// The sample store failed.
    #[error("proxy-cache store failure: {0}")]
    Store(String),
    /// A version artifact could not be written or read.
    #[error("proxy-cache version artifact failure: {0}")]
    Version(String),
    /// A task or request violated the subsystem's contract.
    #[error("proxy-cache contract violation: {0}")]
    Contract(String),
    /// The student, gate, or policy hit a numerical failure.
    #[error("proxy-cache numerical failure: {0}")]
    Numerical(String),
}

impl From<rusqlite::Error> for ProxyCacheError {
    fn from(error: rusqlite::Error) -> Self {
        ProxyCacheError::Store(error.to_string())
    }
}

pub use encoder::{HashEmbedder, TextEmbedder};
pub use manager::{ProxyCacheManager, ProxyCacheManagerConfig};
pub use task::{
    Channel, LabelTarget, RareClasses, RoutingReason, TaskConfig, TaskMode, TaskSpec,
    TeacherChangePolicy, MAX_CLASSES,
};
