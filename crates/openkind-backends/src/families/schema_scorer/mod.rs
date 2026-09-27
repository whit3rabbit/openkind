//! Family: `schema-scorer`.
//!
//! A cross-encoder with a single scalar logit scores one
//! `(state + question schema, candidate)` pair per candidate; the per-question
//! softmax over candidate logits is the answer. The pinned profile uses
//! `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0, BERT-architecture,
//! 6-layer/384-hidden cross-encoder trained with a scalar relevance head)
//! through the candle `bert` implementation, FP32 on CPU.
//!
//! This is the open-weights realization of the surveyed schema-scorer shape:
//! the upstream TypeSafe contract is not owned by openkind, so the profile
//! pins the same *architecture* (single-logit cross-encoder specialized at
//! the Jev question schema) with an open relevance checkpoint and the Jev
//! schema rendering defined by this module.
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — softmax over the
//!   offered candidates; no semantic-none mass of its own.
//! - Query rendering: `state` and the question instruction form the query
//!   side; each candidate criterion forms the passage side.
//! - Continuation state: none — one independent forward pass per candidate.
//! - Text generation: none.

mod engine;
mod model;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::SchemaScorerEngine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "schema-scorer";
/// Pinned backbone repository (MS MARCO cross-encoder).
pub const BACKBONE_ID: &str = "cross-encoder/ms-marco-MiniLM-L-6-v2";
/// Pinned immutable backbone revision.
pub const BACKBONE_REVISION: &str = "233902d25c440f23af6f7d6e94d2946bac0bee0a";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "5a7350af556f0ee66566";
/// SHA-256 of the pinned `model.safetensors` checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae";
/// SHA-256 of the pinned `config.json`.
pub const CONFIG_JSON_SHA256: &str =
    "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc";
/// SHA-256 of the pinned `tokenizer.json`.
pub const TOKENIZER_JSON_SHA256: &str =
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-bert";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Frozen maximum encoded query–passage pair length. Longer pairs fail
/// closed; truncation is forbidden.
pub const MAX_SEQUENCE_TOKENS: usize = 512;
/// Maximum candidates per question. Cost scales linearly per candidate.
pub const MAX_CANDIDATES: usize = 32;
/// Minimum candidates per question, mirroring the Jev wire contract.
pub const MIN_CANDIDATES: usize = 2;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this profile.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Calibration temperature fitted offline on the pinned calibration workload.
///
/// Fitted at 0.08008611758881097 by minimizing mean NLL over the shared
/// 23-case letter-family calibration workload; the scalar-logit head
/// produces small relevance differences, so the fitted temperature sharpens
/// strongly. See `docs/BENCHMARKS.md` for the recorded fit. Loaders reject
/// any other temperature.
pub const CALIBRATION_TEMPERATURE: f64 = 0.080_086_117_588_810_97;

/// Pinned `config.json` values the loader enforces before trusting weights.
pub(crate) fn pinned_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("architectures[0]", json!("BertForSequenceClassification")),
        ("model_type", json!("bert")),
        ("hidden_size", json!(384)),
        ("num_hidden_layers", json!(6)),
        ("num_attention_heads", json!(12)),
        ("intermediate_size", json!(1536)),
        ("vocab_size", json!(30522)),
        ("max_position_embeddings", json!(512)),
        ("type_vocab_size", json!(2)),
        ("id2label.0", json!("LABEL_0")),
    ]
}

/// Errors raised while loading or evaluating the pinned schema-scorer profile.
#[derive(Debug, Error)]
pub enum SchemaScorerError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned schema-scorer profile.
#[derive(Debug, Clone)]
pub struct SchemaScorerEngineConfig {
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// and `tokenizer.json` at [`BACKBONE_REVISION`]. Artifacts are verified
    /// in place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl SchemaScorerEngineConfig {
    /// Fill admission defaults around the given model root.
    pub fn new(model_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 2,
                retry_after_ms: 1_000,
                evaluation_timeout: Some(Duration::from_secs(600)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_id_matches_the_derivation_rule() {
        assert_eq!(
            PROFILE_ID,
            crate::families::support::derive_profile_id(
                FAMILY_SLUG,
                BACKBONE_ID,
                BACKBONE_REVISION
            )
        );
    }

    #[test]
    fn declared_probability_space_is_conditional() {
        assert_eq!(
            DECLARED_PROBABILITY_SPACE.as_str(),
            ProbabilitySpace::ConditionalOnOfferedOptions.as_str()
        );
    }
}
