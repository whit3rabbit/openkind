//! Curated model manifests, verified downloads, and offline installations.
//! Inference loaders remain in `openkind-backends`; this crate never executes
//! model-supplied code or downloads during construction or local reads.

mod manifest;
mod store;

pub use manifest::{Artifact, Catalog, CatalogEntry, Manifest, Source};
pub use store::{default_models_dir, InstalledModel, ModelStore};

use thiserror::Error;

/// The production catalog is a static file in this repository. Catalog entries
/// point to immutable profile manifests whose bytes are checked by SHA-256.
pub const CATALOG_URL: &str = "https://raw.githubusercontent.com/whit3rabbit/openkind-model-registry/main/registry/v1/catalog.json";
pub const QWEN35_STATE_FIRST_MODEL_NAME: &str = "qwen35-state-first:a047d6802c3f06f085b8";
/// Pinned laya English decision-encoder model name.
pub const LAYA_ENGLISH_MODEL_NAME: &str = "laya-english:c8ea29bf1e33a343c4b7";
/// Pinned laya multilingual decision-encoder model name.
pub const LAYA_MULTILINGUAL_MODEL_NAME: &str = "laya-multilingual:f4064eb56fb7f7d325e1";
/// Pinned laya typed-decisions model name.
pub const LAYA_TYPED_DECISIONS_MODEL_NAME: &str = "laya-typed-decisions:9d28cfa9567902801ed1";

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid model metadata: {0}")]
    Invalid(String),
    #[error("model `{0}` is not installed")]
    NotInstalled(String),
    #[error("model `{0}` is in use or being changed")]
    Busy(String),
    #[error("model `{0}` is not in the curated catalog")]
    NotCurated(String),
    #[error("artifact digest mismatch for {0}")]
    DigestMismatch(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
