//! MLX/Metal execution of the pinned JevK5 checkpoint.
//!
//! The forward reuses the parity-verified MLX Qwen3.5 backbone — the same
//! `MlxQwen35Backbone` implementation that serves the pinned state-first
//! profile — through the survey-checkpoint descriptor: the family verifies
//! its own pinned digests, then the backbone streams the single-file BF16
//! checkpoint in place, widens it to FP32 exactly, and prefills the rendered
//! prompt. The readout mirrors `super::model` op for op: the final prompt
//! position's FP32 feature vector dotted against the tied output-embedding
//! rows of the answer letters, read host-side with the same exact widening.
//! Nothing is sampled and no continuation state exists.

use std::sync::Arc;

use crate::qwen35::mlx::{
    MlxCheckpointFormat, MlxError, MlxPrecision, MlxQwen35Backbone, MlxRuntime, MlxSurveyCheckpoint,
};
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding};

use crate::families::decoder_logit_qwen35::model::{VerifiedArtifacts, HIDDEN_SIZE, VOCAB_SIZE};
use crate::families::decoder_logit_qwen35::{Qwen35LogitProfile, RENDERER_ID};

/// The pinned family model on the MLX backbone plus the host-resident tied
/// embedding reader used by the letter readout.
pub(super) struct Jevk5MlxModel {
    backbone: MlxQwen35Backbone,
    embedding: Qwen35Embedding,
}

impl Jevk5MlxModel {
    /// Load the digest-verified checkpoint of `profile` into FP32 MLX arrays.
    ///
    /// The embedding layout is read from the safetensors header after the
    /// whole-file digest passed, so the reader can seek rows in place; the
    /// backbone then streams every required decoder tensor through the
    /// survey descriptor. Must run inside [`MlxRuntime::execute`].
    pub(super) fn load(
        profile: &Qwen35LogitProfile,
        artifacts: &VerifiedArtifacts,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, MlxError> {
        let layout = EmbeddingLayout::read(&artifacts.checkpoint, VOCAB_SIZE, HIDDEN_SIZE)
            .map_err(|error| MlxError::InvalidState(error.to_string()))?;
        let embedding = Qwen35Embedding::from_layout(artifacts.checkpoint.clone(), layout);
        let survey = MlxSurveyCheckpoint {
            // The variant names the pinned single-file HF tensor layout this
            // family's checkpoints share (JevK5 introduced it; Plumb-4B and
            // later letter-logit profiles reuse the identical layout).
            format: MlxCheckpointFormat::PinnedJevk5,
            backbone_id: profile.backbone_id,
            backbone_revision: profile.backbone_revision,
            tokenizer_digest: profile.tokenizer_json_sha256,
            shard_path: artifacts.checkpoint.clone(),
            embedding: embedding.clone(),
        };
        let backbone = MlxQwen35Backbone::load_survey(
            &survey,
            runtime.clone(),
            MlxPrecision::Fp32,
            profile.profile_id,
            RENDERER_ID,
        )?;
        Ok(Self {
            backbone,
            embedding,
        })
    }

    /// Forward the prompt once on the GPU and return the next-token logits
    /// restricted to `letter_ids`.
    ///
    /// The answer slot is the final prompt position; the tied
    /// output-embedding rows of the letter tokens are the readout
    /// projection, computed host-side in f64 exactly like the CPU path.
    pub(super) fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<Vec<f64>, MlxError> {
        let (output, _state) = self.backbone.prefill(prompt_ids)?;
        let feature = output.feature();
        let rows = self
            .embedding
            .embed(letter_ids)
            .map_err(|error| MlxError::InvalidState(error.to_string()))?;
        if rows.token_count() != letter_ids.len() || rows.hidden_size() != HIDDEN_SIZE {
            return Err(MlxError::InvalidState(format!(
                "letter embedding rows have unexpected geometry {}x{}",
                rows.token_count(),
                rows.hidden_size()
            )));
        }
        let logits: Vec<f64> = rows
            .values()
            .as_chunks::<HIDDEN_SIZE>()
            .0
            .iter()
            .map(|row| {
                row.iter()
                    .zip(feature)
                    .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                    .sum::<f64>()
            })
            .collect();
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(MlxError::InvalidState(
                "letter-logit readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }
}
