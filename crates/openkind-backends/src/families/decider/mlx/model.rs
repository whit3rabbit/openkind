//! MLX/Metal execution of the pinned decider-4b checkpoint.
//!
//! The forward reuses the parity-verified MLX Qwen3.5 backbone — the same
//! `MlxQwen35Backbone` implementation that serves the pinned state-first
//! profile — through the survey-checkpoint descriptor: the family verifies
//! its own pinned digests, then the backbone streams the single-file BF16
//! checkpoint in place, widens it to FP32 exactly, and prefills the rendered
//! row. The readout mirrors `super::super::model` op for op: the final
//! prompt position's FP32 feature vector (the `Answer: (` slot) dotted
//! against the tied output-embedding rows of the answer labels, read
//! host-side with the same exact widening. Nothing is sampled and no
//! continuation state exists.

use std::sync::Arc;

use crate::families::qwen35_survey_mlx::prefill_row;
use crate::families::support::{FamilyControl, FamilyError};

use crate::families::decider::model::{VerifiedArtifacts, HIDDEN_SIZE, VOCAB_SIZE};
use crate::families::decider::{DeciderProfile, RENDERER_ID};
use crate::qwen35::mlx::{
    MlxCheckpointFormat, MlxError, MlxPrecision, MlxQwen35Backbone, MlxRuntime, MlxSurveyCheckpoint,
};
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding};

/// The pinned family model on the MLX backbone plus the host-resident tied
/// embedding reader used by the slot readout.
pub(super) struct DeciderMlxModel {
    backbone: MlxQwen35Backbone,
    embedding: Qwen35Embedding,
}

impl DeciderMlxModel {
    /// Load the digest-verified checkpoint of `profile` into FP32 MLX arrays.
    ///
    /// The embedding layout is read from the safetensors header after the
    /// whole-file digest passed, so the reader can seek rows in place; the
    /// backbone then streams every required decoder tensor through the
    /// survey descriptor.
    pub(super) fn load(
        profile: &DeciderProfile,
        artifacts: &VerifiedArtifacts,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, MlxError> {
        let layout = EmbeddingLayout::read(&artifacts.checkpoint, VOCAB_SIZE, HIDDEN_SIZE)
            .map_err(|error| MlxError::InvalidState(error.to_string()))?;
        let embedding = Qwen35Embedding::from_layout(artifacts.checkpoint.clone(), layout);
        let survey = MlxSurveyCheckpoint {
            // The variant names this family's pinned single-file merged
            // checkpoint. Its tensor layout is the one JevK5 introduced, but
            // the evidence identity names the decider repository.
            format: MlxCheckpointFormat::PinnedDecider,
            backbone_id: profile.backbone_id,
            backbone_revision: profile.backbone_revision,
            tokenizer_digest: profile.tokenizer_json_sha256,
            shards: vec![artifacts.checkpoint.clone()],
            embedding: embedding.clone(),
            lora: None,
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

    /// Forward one rendered row on the GPU and return the next-token logits
    /// restricted to `label_ids`.
    ///
    /// The answer slot is the final prompt position; the tied
    /// output-embedding rows of the label tokens are the readout
    /// projection, computed host-side in f64 exactly like the CPU path.
    pub(super) fn slot_logits(
        &self,
        prompt_ids: &[u32],
        label_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        let output = prefill_row(&self.backbone, prompt_ids, control)?;
        let feature = output.feature();
        let rows = self
            .embedding
            .embed(label_ids)
            .map_err(|error| MlxError::InvalidState(error.to_string()))?;
        if rows.token_count() != label_ids.len() || rows.hidden_size() != HIDDEN_SIZE {
            return Err(MlxError::InvalidState(format!(
                "label embedding rows have unexpected geometry {}x{}",
                rows.token_count(),
                rows.hidden_size()
            ))
            .into());
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
                "slot-logit readout produced a non-finite logit".to_owned(),
            )
            .into());
        }
        Ok(logits)
    }
}
