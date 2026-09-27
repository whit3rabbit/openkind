//! Family: `winnow`.
//!
//! A learned router: a decoder backbone with a trained LoRA adapter reads
//! the request state through the letter-logit readout and routes the request
//! to a sibling engine. The pinned profile fine-tunes
//! `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`
//! (Apache-2.0) with a rank-8 LoRA over the last 8 layers, trained with MLX
//! (`mlx_lm.lora`, 300 iterations, Adam, lr 1e-4, scale 20) on a synthetic
//! 800-example English/multilingual routing corpus owned by this repository.
//! The adapter is vendored at
//! [`tests/fixtures/winnow_adapter/adapters.safetensors`](../../tests/fixtures/winnow_adapter/adapters.safetensors).
//!
//! Unlike [`router-script`](crate::families::router_script), routing is a
//! model decision over learned signals rather than a script rule table. The
//! router runs one forward pass per request and never generates text.
//!
//! Profile semantics:
//! - Probability space: inherited from the routed sibling; the routing
//!   distribution itself is telemetry, not a wire answer.
//! - Routing contract: the frozen prompt labels are `A` (english) and
//!   `B` (multilingual); the daemon maps them to served sibling aliases.
//!   No `__none__` label exists in the pinned adapter.
//! - Continuation state: KV cache only, cleared after every forward.

mod engine;
mod model;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;

pub use self::engine::WinnowEngine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "winnow";
/// Pinned base backbone repository.
pub const BACKBONE_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";
/// Pinned immutable base revision.
pub const BACKBONE_REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";
/// Stable derived profile ID for this pinned profile.
pub const PROFILE_ID: &str = "4dff8c5b03cfbf680db6";
/// SHA-256 of the pinned LoRA adapter artifact.
pub const ADAPTER_SHA256: &str = "f4dbf4dae0974b0afae79ceb73487d24a994520fd53365fcb20a28e157b49d83";
/// SHA-256 of the pinned `tokenizer.json`.
pub const TOKENIZER_JSON_SHA256: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
/// Arithmetic/device identity of the family execution path.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-fp32-qwen2-lora";
/// LoRA rank and scale of the pinned adapter.
pub const LORA_RANK: usize = 8;
/// LoRA scale of the pinned adapter.
pub const LORA_SCALE: f64 = 20.0;
/// Frozen maximum rendered prompt length. Longer requests fail closed.
pub const MAX_SEQUENCE_TOKENS: usize = 8_192;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this profile.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Calibration temperature applied to the routing letter logits. Pinned at
/// `1.0`: the adapter converged to near-lossless routing on its validation
/// split (val loss 0.030), and no NLL fit could improve the operating point.
pub const CALIBRATION_TEMPERATURE: f64 = 1.0;

/// Errors raised while loading or evaluating the pinned winnow profile.
#[derive(Debug, Error)]
pub enum WinnowError {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned winnow profile.
#[derive(Debug, Clone)]
pub struct WinnowEngineConfig {
    /// Model root with the pinned base checkpoint (`model.safetensors` +
    /// `tokenizer.json`) at [`BACKBONE_REVISION`]. Verified in place.
    pub model_root: PathBuf,
    /// Path to the pinned LoRA adapter (`adapters.safetensors`).
    pub adapter_path: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl WinnowEngineConfig {
    /// Fill admission defaults around the given roots.
    pub fn new(model_root: impl Into<PathBuf>, adapter_path: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            adapter_path: adapter_path.into(),
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
}

#[cfg(test)]
mod routing_contract_tests {
    use super::*;

    #[test]
    fn pinned_adapter_digest_is_recorded() {
        // The vendored adapter is a repository artifact; the pinned digest
        // must always match the vendored file.
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/winnow_adapter/adapters.safetensors");
        let actual = crate::families::support::sha256_file(&fixture).expect("digest");
        assert_eq!(actual, ADAPTER_SHA256);
    }

    #[test]
    fn lora_constants_match_the_training_run() {
        assert_eq!(LORA_RANK, 8);
        assert_eq!(LORA_SCALE, 20.0);
    }
}
