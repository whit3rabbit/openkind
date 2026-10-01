//! Pinned, verified local evaluation-dataset installations for OpenKind
//! benchmarking.
//!
//! This crate downloads labeled public datasets from the Hugging Face Hub at
//! exact pinned revisions into a per-user cache outside the repository, and
//! verifies every file digest. Dataset bytes are never committed to this
//! repository and never redistributed; the checked-in registry pins only
//! repository identities, revision hashes, sizes, and digests. Downloads
//! happen only under the explicit `openkind-bench dataset pull` and
//! `dataset pin` commands; construction, local reads, builds, and tests never
//! touch the network.

pub mod auth;
pub mod definitions;
pub mod registry;
pub mod rows;
pub mod store;

pub use auth::{resolve_token, TokenSource};
pub use definitions::{DatasetDefinition, DATASET_DEFINITIONS};
pub use registry::{DatasetEntry, DatasetFile, DatasetRegistry, Gated, TemplateRef};
pub use store::{default_datasets_dir, DatasetStore, InstalledDataset};

/// The checked-in registry bytes pinned by `registry/v1/datasets.json`.
pub const DATASETS_JSON: &str = include_str!("../registry/v1/datasets.json");

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("dataset {0} is not in the registry")]
    NotCurated(String),
    #[error("dataset {0} is not installed; run `openkind-bench dataset pull {0}`")]
    NotInstalled(String),
    #[error("digest mismatch: {0}")]
    DigestMismatch(String),
    #[error("{0} is busy")]
    Busy(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("parquet: {0}")]
    Parquet(String),
}

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;
