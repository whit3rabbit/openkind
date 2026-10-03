//! Curated model manifests, verified downloads, and offline installations.
//! Inference loaders remain in `openkind-backends`; this crate never executes
//! model-supplied code or downloads during construction or local reads.

mod manifest;
mod store;

pub use manifest::{Artifact, Catalog, CatalogEntry, Manifest, Source};
pub use store::{default_models_dir, InstalledModel, ModelStore};

use thiserror::Error;

/// The production catalog mirrors the static file in this repository. Its
/// digest is pinned below so the mutable mirror cannot authorize new content.
pub const CATALOG_URL: &str = "https://raw.githubusercontent.com/whit3rabbit/openkind-model-registry/main/registry/v1/catalog.json";
pub const CATALOG_SHA256: &str = "0c8b3b7bdbda3a16520460e659d46a92986a3c326e96d29f1550f6675ba79455";
pub const QWEN35_STATE_FIRST_MODEL_NAME: &str = "qwen35-state-first:a047d6802c3f06f085b8";
/// Pinned laya English decision-encoder model name.
pub const LAYA_ENGLISH_MODEL_NAME: &str = "laya-english:c8ea29bf1e33a343c4b7";
/// Pinned laya multilingual decision-encoder model name.
pub const LAYA_MULTILINGUAL_MODEL_NAME: &str = "laya-multilingual:f4064eb56fb7f7d325e1";
/// Pinned laya typed-decisions model name.
pub const LAYA_TYPED_DECISIONS_MODEL_NAME: &str = "laya-typed-decisions:9d28cfa9567902801ed1";
/// Pinned decoder-logit-letter model name.
pub const DECODER_LOGIT_LETTER_MODEL_NAME: &str = "decoder-logit-letter:5492c97dfcdaf3fe9439";
/// Pinned encoder-nli model name.
pub const ENCODER_NLI_MODEL_NAME: &str = "encoder-nli:1041a4c362338a61b820";
/// Pinned encoder-instruct-label model name.
pub const ENCODER_INSTRUCT_LABEL_MODEL_NAME: &str = "encoder-instruct-label:9fd68313a5606eca42f2";
/// Pinned decoder-logit-llm model name.
pub const DECODER_LOGIT_LLM_MODEL_NAME: &str = "decoder-logit-llm:465963d705b6f35d6208";
/// Pinned schema-scorer model name.
pub const SCHEMA_SCORER_MODEL_NAME: &str = "schema-scorer:5a7350af556f0ee66566";
/// Pinned qwen3guard (Stream) model name.
pub const QWEN3GUARD_MODEL_NAME: &str = "qwen3guard:0fcf416cab16d94f933d";
/// Pinned kev model name.
pub const KEV_MODEL_NAME: &str = "kev:39d88c11faeb4ac165fa";
/// Pinned decoder-logit-qwen35 model name.
pub const DECODER_LOGIT_QWEN35_MODEL_NAME: &str = "decoder-logit-qwen35:415bcf4a064e6dadcf85";
/// Pinned Cloudflare Clef-Flash BF16 model name.
pub const CLEF_FLASH_MODEL_NAME: &str = "clef-flash:dfe12a21a5c9dd5b2fb1";
/// Pinned Cloudflare Clef-Flash GGUF Q4_K_M model name.
pub const CLEF_FLASH_GGUF_MODEL_NAME: &str = "clef-flash-gguf:c330d9ee7e9cc658ad45";
/// Pinned Cloudflare Clef 27B GGUF Q4_K_M model name.
pub const CLEF_27B_GGUF_MODEL_NAME: &str = "clef-27b-gguf:48cb5634b4a258de5a6b";
/// Pinned plumb-4b model name.
pub const PLUMB_4B_MODEL_NAME: &str = "plumb-4b:c1f080794d38e94a0bc2";
/// Pinned decider-4b model name.
pub const DECIDER_4B_MODEL_NAME: &str = "decider-4b:0529bf6f2bed84641701";
/// Pinned raw decoder-logit-qwen3 control model names.
pub const DECODER_LOGIT_QWEN3_06B_MODEL_NAME: &str = "decoder-logit-qwen3-06b:d900f4af57509fe02e62";
pub const DECODER_LOGIT_QWEN3_17B_MODEL_NAME: &str = "decoder-logit-qwen3-17b:8119b9271f8d011e7d03";
pub const DECODER_LOGIT_QWEN3_4B_MODEL_NAME: &str = "decoder-logit-qwen3-4b:9dfaf11792a8d061b6b8";
/// Pinned von model name.
pub const VON_MODEL_NAME: &str = "von:69219703407bd39cca0c";
/// Pinned winnow model name.
pub const WINNOW_MODEL_NAME: &str = "winnow:4dff8c5b03cfbf680db6";
/// Pinned winnow-e4b (Gemma 4 backbone) model name.
pub const WINNOW_E4B_MODEL_NAME: &str = "winnow-e4b:656ac636ce450cf79c7d";
/// Pinned Strands Decider 2B (Hobson v19) model name.
pub const STRANDS_DECIDER_2B_MODEL_NAME: &str = "strands-decider-2b:6a02bb0d1c6b25cae74b";
/// Pinned BGE-small sentence-embedding encoder for the proxy-cache student.
pub const ENCODER_EMBEDDING_MODEL_NAME: &str = "encoder-embedding:8d9498269ef05d95d93c";

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
