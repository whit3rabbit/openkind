//! `DecisionEngine` adapter for the pinned schema-scorer profile.

use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{SchemaScorerModel, VerifiedArtifacts};
use super::renderer::SchemaScorerRenderer;
use super::{
    SchemaScorerEngineConfig, SchemaScorerError, BACKBONE_ID, MAX_CANDIDATES, MIN_CANDIDATES,
    PROFILE_ID,
};

/// Loaded pinned schema-scorer engine.
pub struct SchemaScorerEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: SchemaScorerRenderer,
    model: SchemaScorerModel,
}

impl SchemaScorerEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    pub fn load(
        config: SchemaScorerEngineConfig,
    ) -> Result<BoundedFamilyEngine, SchemaScorerError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = SchemaScorerRenderer::load(&artifacts.tokenizer)?;
        let model = SchemaScorerModel::load(&artifacts)?;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for SchemaScorerEngine {
    fn backend_id(&self) -> &str {
        "schema-scorer/cpu-fp32"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} scalar cross-encoder schema scorer ({PROFILE_ID}, FP32 CPU)."
            ),
            release_date: "2026-09-26".into(),
        }
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        control.check()?;
        let state = match &request.state {
            State::Text(text) => text.clone(),
            State::Object(map) => serde_json::to_string_pretty(map).map_err(|error| {
                FamilyError::InvalidInput(format!("state serialization failed: {error}"))
            })?,
            State::Array(items) => serde_json::to_string_pretty(items).map_err(|error| {
                FamilyError::InvalidInput(format!("state serialization failed: {error}"))
            })?,
        };
        if state.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "state must contain non-whitespace text".to_owned(),
            ));
        }
        let mut question_ids: Vec<_> = request.questions.keys().cloned().collect();
        question_ids.sort();
        if question_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "request contains no questions".to_owned(),
            ));
        }

        let mut answers = std::collections::HashMap::<
            String,
            openkind_core::Answer,
            openkind_core::WireHashState,
        >::with_capacity_and_hasher(
            question_ids.len(), Default::default()
        );
        let mut input_tokens: u64 = 0;
        for id in &question_ids {
            control.check()?;
            let question = request
                .questions
                .get(id)
                .expect("question id came from the request map");
            let instruction = crate::families::wire::instruction_text(question)?;
            let unpacked = crate::families::wire::unpack_question(id, question)?;
            if unpacked.criteria.len() > MAX_CANDIDATES {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` offers {} candidates but the profile accepts at most {MAX_CANDIDATES}",
                    unpacked.criteria.len()
                )));
            }
            if unpacked.criteria.len() < MIN_CANDIDATES {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` needs at least {MIN_CANDIDATES} candidates"
                )));
            }
            // The Jev question schema renders into the query side: the state
            // document plus the question instruction. The candidate criterion
            // is the passage side.
            let query = format!("{state}\n\nQuestion: {instruction}");
            let mut logits = Vec::with_capacity(unpacked.criteria.len());
            for criterion in &unpacked.criteria {
                control.check()?;
                let pair_ids = self.inner.renderer.encode_pair(&query, criterion)?;
                let logit = self.inner.model.relevance_logit(&pair_ids)?;
                input_tokens = input_tokens.saturating_add(pair_ids.len() as u64);
                logits.push(logit);
            }
            let probabilities = temperature_softmax(&logits, super::CALIBRATION_TEMPERATURE)?;
            let answer =
                crate::families::wire::answer_from_probabilities(&unpacked, &probabilities)?;
            answers.insert(id.clone(), answer);
        }
        control.check()?;
        Ok(SystemResponse {
            model: request.model,
            answers,
            usage: Usage {
                input_tokens: u32::try_from(input_tokens).unwrap_or(u32::MAX),
                output_tokens: 0,
            },
        })
    }
}
