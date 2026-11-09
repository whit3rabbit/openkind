//! MLX/Metal execution of the pinned raw-control checkpoints (feature
//! `mlx`, macOS arm64).
//!
//! The forward reuses the shared dense-Qwen MLX backbone
//! ([`DenseQwenMlxModel`]): the digest-verified BF16 shards stream into FP32
//! arrays at the profile's pinned geometry and prefill the rendered letter
//! pass as one causal sequence. The readout mirrors `super::model` op for
//! op: the final prompt position's FP32 hidden state (the final-normed
//! state, which the CPU oracle reads through the same `last_hidden`
//! contract) dotted against the tied embedding rows of the answer letters,
//! read host-side with the same exact widening. Nothing is sampled and no
//! continuation state exists.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! the process-wide [`MlxRuntime`] execution lock, per the workspace MLX
//! discipline.

use std::sync::Arc;

use crate::families::qwen_dense_mlx::{DenseQwenConfig, DenseQwenLoadSpec, DenseQwenMlxModel};
use crate::families::support::{FamilyControl, FamilyError};
use crate::qwen35::mlx::MlxRuntime;

use super::engine::LetterLogits;
use super::model::VerifiedArtifacts;
use super::{Qwen3LogitProfile, MAX_SEQUENCE_TOKENS};

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// BF16 checkpoints widened to FP32 arrays computed on the Metal GPU through
/// the shared dense Qwen MLX backbone.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-decoder-logit-qwen3";

/// The pinned family model on the MLX backbone: FP32 decoder arrays plus the
/// host-resident tied embedding used by the letter readout.
pub(super) struct Qwen3ControlMlxModel {
    backbone: DenseQwenMlxModel,
    hidden_size: usize,
}

impl Qwen3ControlMlxModel {
    /// Load the digest-verified shards into FP32 MLX arrays at the profile's
    /// pinned geometry.
    pub(super) fn load(
        artifacts: &VerifiedArtifacts,
        profile: &Qwen3LogitProfile,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        // HF sharding keeps the tied embedding table in the first shard; a
        // layout that moved it fails closed on the named read below.
        let embedding_shard = artifacts.checkpoints.first().ok_or_else(|| {
            FamilyError::InvalidInput("the profile pins no checkpoint shards".to_owned())
        })?;
        let config = DenseQwenConfig {
            vocab_size: profile.vocab_size,
            hidden_size: profile.hidden_size,
            head_dim: profile.head_dim,
            num_attention_heads: profile.num_attention_heads,
            num_key_value_heads: profile.num_key_value_heads,
            num_hidden_layers: profile.num_hidden_layers,
            intermediate_size: profile.intermediate_size,
            rope_theta: profile.rope_theta,
            rms_norm_eps: 1e-6,
            qk_norm: true,
            max_sequence_tokens: MAX_SEQUENCE_TOKENS,
        };
        let spec = DenseQwenLoadSpec {
            shards: &artifacts.checkpoints,
            embedding_shard,
            adapter: None,
            config,
        };
        // The loader reads tensors through its own host-side safetensors
        // reader and allocates MLX arrays, so it must run under the
        // process-wide execution lock like every other MLX operation (the
        // lock is not reentrant).
        let backbone = runtime.execute(|| DenseQwenMlxModel::load(spec, runtime))??;
        Ok(Self {
            backbone,
            hidden_size: profile.hidden_size,
        })
    }

    /// Final-normed FP32 hidden row at the final prompt position, the CPU
    /// oracle's `last_hidden` contract.
    fn last_hidden(&self, prompt_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let hidden = self
            .backbone
            .runtime()
            .execute(|| self.backbone.forward_hidden(prompt_ids))??;
        if hidden.len() != prompt_ids.len() * self.hidden_size {
            return Err(FamilyError::Numerical(format!(
                "backbone output width does not match the pinned geometry: {} values for \
                 {} tokens x {} wide",
                hidden.len(),
                prompt_ids.len(),
                self.hidden_size
            )));
        }
        let start = (prompt_ids.len() - 1) * self.hidden_size;
        Ok(hidden[start..].to_vec())
    }
}

impl LetterLogits for Qwen3ControlMlxModel {
    fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        control.check()?;
        // The forward takes the process-wide execution lock for its whole
        // duration; no outer `runtime.execute` wraps it (the lock is not
        // reentrant).
        let hidden = self.last_hidden(prompt_ids)?;
        control.check()?;
        let rows = self.backbone.embed_rows(letter_ids)?;
        if rows.len() != letter_ids.len() * self.hidden_size {
            return Err(FamilyError::Numerical(format!(
                "letter readout geometry mismatch: {} values for {} letters x {} wide",
                rows.len(),
                letter_ids.len(),
                self.hidden_size
            )));
        }
        let logits: Vec<f64> = rows
            .chunks_exact(self.hidden_size)
            .map(|row| {
                row.iter()
                    .zip(&hidden)
                    .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                    .sum::<f64>()
            })
            .collect();
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(FamilyError::Numerical(
                "letter-logit readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }
}
