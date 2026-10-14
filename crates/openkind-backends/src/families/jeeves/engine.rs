//! Bounded native adapter, preserving Jeeves' offered-option probability space.
use super::{
    artifacts::VerifiedArtifacts, model::JeevesModel, renderer::JeevesRenderer, MAX_QUESTIONS,
    MODEL_ID, PROFILE_ID,
};
use crate::families::kev::engine::{question_instructions, question_options};
use crate::families::kev::renderer::render_json;
use crate::families::support::{
    BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator, FamilyLimits,
};
use crate::families::wire::{answer_from_probabilities, unpack_question, RESERVED_NONE_OPTION};
use openkind_core::{ModelInfo, Question, State, SystemRequest, SystemResponse, Usage};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

/// Filesystem and bounded admission configuration for native Jeeves.
#[derive(Debug, Clone)]
pub struct JeevesEngineConfig {
    /// Root of the pinned 4-bit package. No assets are fetched or copied.
    pub model_root: PathBuf,
    /// Cooperative cancellation, queue, and deadline limits.
    pub limits: FamilyLimits,
}

/// Native no-thinking Jeeves engine. It never runs a Python interpreter.
pub struct JeevesEngine {
    renderer: JeevesRenderer,
    model: Mutex<JeevesModel>,
}
impl JeevesEngine {
    /// Verify artifacts, initialize the model once, then wrap bounded admission.
    pub fn load(config: JeevesEngineConfig) -> Result<BoundedFamilyEngine, FamilyError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = JeevesRenderer::load(&artifacts.tokenizer)?;
        let model = JeevesModel::load(&artifacts)?;
        Ok(BoundedFamilyEngine::new(
            Arc::new(Self {
                renderer,
                model: Mutex::new(model),
            }),
            config.limits,
        ))
    }
}

impl FamilyEvaluator for JeevesEngine {
    fn backend_id(&self) -> &str {
        "jeeves/mlx-affine4-fp32"
    }
    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: "jeeves-native".into(),
            description: format!(
                "{MODEL_ID} native no-thinking pointer decisions ({PROFILE_ID}, affine4/FP32)."
            ),
            release_date: "2026-09-29".into(),
        }
    }
    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        control.check()?;
        if request.questions.is_empty() || request.questions.len() > MAX_QUESTIONS {
            return Err(FamilyError::InvalidInput(format!(
                "Jeeves needs 1..{MAX_QUESTIONS} questions"
            )));
        }
        let state = match &request.state {
            State::Text(text) => text.clone(),
            State::Object(map) => render_json(&serde_json::Value::Object(map.clone())),
            State::Array(items) => render_json(&serde_json::Value::Array(items.clone())),
        };
        let mut ids: Vec<_> = request.questions.keys().collect();
        ids.sort();
        let mut rows = Vec::with_capacity(ids.len());
        // Validate and render the complete request before expensive forward execution.
        for id in ids {
            let question = &request.questions[id];
            if let Question::Choice(choice) = question {
                if !choice
                    .criteria
                    .get(RESERVED_NONE_OPTION)
                    .and_then(|s| s.as_deref())
                    .is_some_and(|s| !s.trim().is_empty())
                {
                    return Err(FamilyError::InvalidInput("native Jeeves Choice requires a described __none__ option; it is an offered candidate, not a separate calibrated rejection head".into()));
                }
            }
            let unpacked = unpack_question(id, question)?;
            let explicit = matches!(question, Question::Noul(noul) if noul.criteria.is_some());
            let options = question_options(&unpacked, question, explicit);
            let instruction = render_json(question_instructions(question));
            let row = self.renderer.encode(&state, &instruction, &options)?;
            rows.push((id.clone(), unpacked, row));
        }
        let mut answers = Default::default();
        let mut input_tokens = 0_u64;
        for (id, unpacked, row) in rows {
            control.check()?;
            let result = self
                .model
                .lock()
                .map_err(|_| FamilyError::Numerical("Jeeves model lock poisoned".into()))?
                .score(&row, Some(control))?;
            input_tokens = input_tokens.saturating_add(row.ids.len() as u64);
            let answer = answer_from_probabilities(&unpacked, &result.probabilities)?;
            std::collections::HashMap::insert(&mut answers, id, answer);
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
