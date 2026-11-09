//! MLX/Metal execution of the pinned sieve-4b serving checkpoint.
//!
//! The forward reuses the parity-verified MLX Qwen3.5 backbone through the
//! survey-checkpoint descriptor: the family verifies its own pinned base
//! shards and adapter, then the backbone streams the base checkpoint in
//! place, widens it to FP32 exactly, and applies the PEFT LoRA merge
//! (`base + (B @ A) · alpha/r`) host-side before the values reach MLX — the
//! same merged weights the CPU oracle serves. The readout mirrors
//! `super::super::model` op for op: the final branch position's FP32 feature
//! vector dotted against the first `count` rows of the trained float32 head.
//! Nothing is sampled and no continuation state exists.

use std::sync::Arc;

use crate::families::qwen35_survey_mlx::prefill_row;
use crate::families::support::{FamilyControl, FamilyError};

use crate::families::sieve::model::{read_head, VerifiedArtifacts, EMBEDDING_TENSOR};
use crate::families::sieve::{SieveProfile, CODE_TABLE_WIDTH, RENDERER_ID};
use crate::qwen35::mlx::{
    MlxCheckpointFormat, MlxError, MlxPrecision, MlxQwen35Backbone, MlxRuntime,
    MlxSurveyCheckpoint, MlxSurveyLora,
};
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding};

/// Vocab size the base embedding table must have (mirrors
/// `super::super::model::VOCAB_SIZE`).
const VOCAB_SIZE: usize = 248_320;

/// The pinned family model on the MLX backbone plus the trained float32
/// readout head.
pub(super) struct SieveMlxModel {
    backbone: MlxQwen35Backbone,
    /// Row-major `[CODE_TABLE_WIDTH, hidden]` float32 head rows.
    readout: Vec<f32>,
    hidden_size: usize,
}

impl SieveMlxModel {
    /// Load the digest-verified base shards of `profile`, merge the verified
    /// adapter, and build the FP32 MLX backbone plus head.
    pub(super) fn load(
        profile: &SieveProfile,
        artifacts: &VerifiedArtifacts,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, MlxError> {
        let geometry = profile.geometry;
        let first_shard = artifacts
            .base_shards
            .first()
            .ok_or_else(|| MlxError::InvalidState("profile pins no base shard".to_owned()))?;
        let layout = EmbeddingLayout::read_named(
            first_shard,
            EMBEDDING_TENSOR,
            VOCAB_SIZE,
            geometry.hidden_size,
        )
        .map_err(|error| MlxError::InvalidState(error.to_string()))?;
        let embedding = Qwen35Embedding::from_layout(first_shard.clone(), layout);
        let tokenizer_digest = profile
            .checkpoint_files
            .iter()
            .find(|(name, _, _)| *name == "tokenizer.json")
            .map(|(_, _, sha256)| *sha256)
            .ok_or_else(|| MlxError::InvalidState("profile pins no tokenizer digest".to_owned()))?;
        let survey = MlxSurveyCheckpoint {
            format: MlxCheckpointFormat::PinnedSieve,
            backbone_id: profile.backbone_id,
            backbone_revision: profile.backbone_revision,
            tokenizer_digest,
            shards: artifacts.base_shards.clone(),
            embedding: embedding.clone(),
            lora: Some(MlxSurveyLora {
                adapter_path: artifacts.adapter.clone(),
                rank: profile.lora_rank,
                scale: profile.lora_scale,
            }),
        };
        let backbone = MlxQwen35Backbone::load_survey(
            &survey,
            runtime.clone(),
            MlxPrecision::Fp32,
            profile.profile_id,
            RENDERER_ID,
        )?;
        let readout = read_head(&artifacts.head, CODE_TABLE_WIDTH, geometry.hidden_size)
            .map_err(|error| MlxError::InvalidState(error.to_string()))?;
        Ok(Self {
            backbone,
            readout,
            hidden_size: geometry.hidden_size,
        })
    }

    /// Forward one branch on the GPU and return the code logits restricted
    /// to the first `count` head rows (the offered options' codes; the
    /// reference masks the remaining rows before the softmax, which
    /// excluding them reproduces exactly).
    pub(super) fn code_logits(
        &self,
        prompt_ids: &[u32],
        count: usize,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        if count == 0 || count > CODE_TABLE_WIDTH {
            return Err(MlxError::InvalidState(format!(
                "code readout serves 1..={CODE_TABLE_WIDTH} codes, found {count}"
            ))
            .into());
        }
        let output = prefill_row(&self.backbone, prompt_ids, control)?;
        let feature = output.feature();
        let mut logits = Vec::with_capacity(count);
        for row in self.readout.chunks_exact(self.hidden_size).take(count) {
            let logit: f64 = row
                .iter()
                .zip(feature)
                .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                .sum();
            if !logit.is_finite() {
                return Err(MlxError::InvalidState(
                    "code readout produced a non-finite logit".to_owned(),
                )
                .into());
            }
            logits.push(logit);
        }
        Ok(logits)
    }
}
