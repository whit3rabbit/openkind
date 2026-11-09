//! MLX/Metal execution of the pinned decoder-letter checkpoint.
//!
//! The forward reuses the shared dense Qwen MLX body
//! ([`DenseQwenMlxModel`]) at the pinned Qwen2 geometry (no q/k norm): the
//! digest-verified checkpoint streams into FP32 MLX arrays in place and the
//! family verifies nothing else. The readout mirrors
//! [`super::super::model`] op for op: the final prompt position's FP32
//! hidden row is dotted against the tied output-embedding rows of the
//! option letters, computed host-side in f64. Because the pinned
//! `tie_word_embeddings` checkpoint has no separate LM head, the full-vocab
//! logits row the CPU oracle indexes is exactly `hidden · embedding.T`, so
//! the per-letter dots reproduce the indexed values. Nothing is sampled and
//! no continuation state exists.

use std::sync::Arc;

use crate::families::qwen_dense_mlx::{DenseQwenConfig, DenseQwenLoadSpec, DenseQwenMlxModel};
use crate::families::support::FamilyError;
use crate::qwen35::mlx::MlxRuntime;

use super::super::model::VerifiedArtifacts;

/// The pinned family model on the shared dense Qwen MLX body.
pub(super) struct DecoderLetterMlxModel {
    backbone: DenseQwenMlxModel,
    /// Pinned vocabulary and hidden width, retained for the readout's
    /// fail-closed geometry checks (the shared body keeps them private).
    vocab_size: usize,
    hidden_size: usize,
}

/// Derive the dense Qwen MLX geometry from the pinned (digest-verified)
/// `config.json`. The loader already rejected any drift from the pinned
/// values, so these fields are the checkpoint's contract.
fn dense_config(config: &candle_transformers::models::qwen2::Config) -> DenseQwenConfig {
    DenseQwenConfig {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        head_dim: config.hidden_size / config.num_attention_heads,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        num_hidden_layers: config.num_hidden_layers,
        intermediate_size: config.intermediate_size,
        rope_theta: config.rope_theta,
        rms_norm_eps: config.rms_norm_eps,
        // Qwen2 has no per-head q/k RMSNorm.
        qk_norm: false,
        max_sequence_tokens: super::super::MAX_SEQUENCE_TOKENS,
    }
}

impl DecoderLetterMlxModel {
    /// Load the digest-verified checkpoint into FP32 MLX arrays on the
    /// shared dense body.
    pub(super) fn load(
        artifacts: &VerifiedArtifacts,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        let config = dense_config(&artifacts.config);
        let shards = vec![artifacts.checkpoint.clone()];
        let spec = DenseQwenLoadSpec {
            shards: &shards,
            embedding_shard: &artifacts.checkpoint,
            // The pinned profile carries no adapter.
            adapter: None,
            config,
        };
        // The safetensors reader and every array construction run under the
        // process-wide MLX execution lock; the lock is not reentrant, so the
        // engine load must not already hold it.
        let backbone = runtime.execute(|| DenseQwenMlxModel::load(spec, runtime))??;
        Ok(Self {
            backbone,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
        })
    }

    /// Forward the prompt once and return the next-token logits restricted
    /// to `letter_ids`. The answer slot is the final prompt position;
    /// nothing is sampled and no continuation tokens exist.
    ///
    /// The tied-embedding identity: the CPU oracle reads the full-vocab
    /// logits row produced by `hidden @ embedding.T` and indexes it at each
    /// letter id; a dot of the same hidden row against that id's embedding
    /// row reproduces the same value, so only the letter rows leave the
    /// embedding table.
    pub(super) fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<Vec<f64>, FamilyError> {
        if prompt_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "prompt token ids are empty".to_owned(),
            ));
        }
        // Fail closed on out-of-vocabulary letters exactly like the CPU
        // `letter_token_contract` check.
        for token in letter_ids {
            let in_range = usize::try_from(*token)
                .map(|index| index < self.vocab_size)
                .unwrap_or(false);
            if !in_range {
                return Err(FamilyError::ContractMismatch {
                    field: "letter_token_contract",
                    expected: "letter token inside the model vocabulary".to_owned(),
                    actual: format!("letter token id {token} out of range"),
                });
            }
        }
        let rows = self.backbone.embed_rows(letter_ids)?;
        if rows.len() != letter_ids.len() * self.hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "letter embedding rows have unexpected geometry: {} values for {} letters",
                rows.len(),
                letter_ids.len()
            )));
        }
        let hidden = self
            .backbone
            .runtime()
            .execute(|| self.backbone.forward_hidden(prompt_ids))??;
        let token_count = prompt_ids.len();
        if hidden.len() != token_count * self.hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "backbone output width does not match the pinned geometry: {} values for \
                 {token_count} tokens",
                hidden.len()
            )));
        }
        // Last prompt position: the answer slot.
        let slot = &hidden[(token_count - 1) * self.hidden_size..];
        let logits: Vec<f64> = rows
            .chunks_exact(self.hidden_size)
            .map(|row| {
                row.iter()
                    .zip(slot)
                    .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                    .sum::<f64>()
            })
            .collect();
        if logits.len() != letter_ids.len() {
            return Err(FamilyError::InvalidInput(format!(
                "letter readout produced {} logits for {} letters",
                logits.len(),
                letter_ids.len()
            )));
        }
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(FamilyError::Numerical(
                "letter readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }
}
