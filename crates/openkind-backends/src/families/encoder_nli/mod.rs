//! Family: `encoder-nli`.
//!
//! An NLI encoder performs one premise–hypothesis forward pass per
//! candidate: the state is the premise, each candidate criterion is the
//! hypothesis, and the per-candidate entailment probability is the decision
//! score. The pinned profile uses `typeform/distilbert-base-uncased-mnli`
//! (Apache-2.0, MNLI fine-tune) through the candle `distilbert`
//! implementation, FP32 on CPU.
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the distribution
//!   over offered options (normalized entailment probabilities) sums to one;
//!   the readout has no semantic-none mass of its own.
//! - `Noul` maps to the entailment probability of the `true` criterion
//!   hypothesis directly; the `false` criterion is not scored.
//! - Continuation state: none — every candidate is an independent full
//!   forward pass; nothing is retained between questions or requests.
//! - Text generation: none.

mod engine;
#[doc(hidden)]
pub mod model;
#[doc(hidden)]
pub mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::EncoderNliEngine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "encoder-nli";
/// Pinned backbone repository (MNLI fine-tune of DistilBERT base uncased).
pub const BACKBONE_ID: &str = "typeform/distilbert-base-uncased-mnli";
/// Pinned immutable backbone revision.
pub const BACKBONE_REVISION: &str = "cfa538a0fddbbd978fefe8966c1aeff7ad409c90";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "1041a4c362338a61b820";
/// SHA-256 of the pinned `model.safetensors` checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "16d47e5948c7076ecfe8b9d343c0d1474cd600f18405c40cfcf183605787af41";
/// SHA-256 of the pinned `config.json`.
pub const CONFIG_JSON_SHA256: &str =
    "d6d658b44d7260410d8aa3f6cd585016656dfe57dd57855c070f86ddcb257385";
/// SHA-256 of the pinned `vocab.txt` WordPiece vocabulary.
pub const VOCAB_TXT_SHA256: &str =
    "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-distilbert";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Frozen maximum encoded premise–hypothesis pair length. Longer pairs fail
/// closed; truncation is forbidden.
pub const MAX_SEQUENCE_TOKENS: usize = 512;
/// Maximum candidates per question. Cost scales linearly per candidate.
pub const MAX_CANDIDATES: usize = 32;
/// Minimum candidates per question, mirroring the Jev wire contract.
pub const MIN_CANDIDATES: usize = 2;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this profile. The threshold exists so telemetry and evidence artifacts
/// can record acceptance against a declared operating point.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Calibration temperature fitted offline on the pinned calibration workload.
///
/// The temperature calibrates the decision distribution over candidates,
/// applied to the log-entailment logits; the label-level NLI softmax is not
/// rescaled. Fitted at 1.65588139232157294 by minimizing mean NLL over 15
/// deterministic synthetic premise-hypothesis decision cases whose correct
/// option is entailed by the state document (8 `Choice`, 4 `Noul`, 3
/// `Score`); see `docs/BENCHMARKS.md` for the recorded fit. Loaders reject
/// any other temperature.
pub const CALIBRATION_TEMPERATURE: f64 = 1.655_881_392_321_573;

/// NLI label order of the pinned checkpoint (`config.json` id2label).
pub(crate) const LABEL_ENTAILMENT: usize = 0;
pub(crate) const LABEL_NEUTRAL: usize = 1;
pub(crate) const LABEL_CONTRADICTION: usize = 2;

/// Pinned `config.json` values the loader enforces before trusting weights.
pub(crate) fn pinned_config() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        (
            "architectures[0]",
            json!("DistilBertForSequenceClassification"),
        ),
        ("model_type", json!("distilbert")),
        ("dim", json!(768)),
        ("n_layers", json!(6)),
        ("n_heads", json!(12)),
        ("hidden_dim", json!(3072)),
        ("vocab_size", json!(30522)),
        ("max_position_embeddings", json!(512)),
        ("id2label.0", json!("ENTAILMENT")),
        ("id2label.1", json!("NEUTRAL")),
        ("id2label.2", json!("CONTRADICTION")),
    ]
}

/// Errors raised while loading or evaluating the pinned encoder-NLI profile.
#[derive(Debug, Error)]
pub enum EncoderNliError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned encoder-NLI profile.
#[derive(Debug, Clone)]
pub struct EncoderNliEngineConfig {
    /// Model root containing the pinned `model.safetensors`, `config.json`,
    /// and `vocab.txt` at [`BACKBONE_REVISION`]. Artifacts are verified in
    /// place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl EncoderNliEngineConfig {
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
