//! `DecisionEngine` adapter for the pinned decoder-letter profile.

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::letter_renderer::LetterRenderer;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{DecoderLetterModel, VerifiedArtifacts};

use super::{DecoderLetterEngineConfig, DecoderLetterError, BACKBONE_ID, PROFILE_ID};

/// Loaded pinned decoder-letter engine.
pub struct DecoderLetterEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: LetterRenderer,
    model: DecoderLetterModel,
}

impl DecoderLetterEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The returned adapter implements [`DecisionEngine`] and carries
    /// admission, queue, deadline, and cancellation control.
    pub fn load(
        config: DecoderLetterEngineConfig,
    ) -> Result<BoundedFamilyEngine, DecoderLetterError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = LetterRenderer::load(&artifacts.tokenizer, super::MAX_SEQUENCE_TOKENS)?;
        let model = DecoderLetterModel::load(&artifacts)?;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for DecoderLetterEngine {
    fn backend_id(&self) -> &str {
        "decoder-logit-letter/cpu-fp32"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} letter-logit decision engine ({PROFILE_ID}, FP32 CPU)."
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
        let mut question_ids: Vec<_> = request.questions.keys().cloned().collect();
        question_ids.sort();
        if question_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "request contains no questions".to_owned(),
            ));
        }

        let mut answers: HashMap<String, openkind_core::Answer, openkind_core::WireHashState> =
            HashMap::with_capacity_and_hasher(question_ids.len(), Default::default());
        let mut input_tokens: u64 = 0;
        for id in &question_ids {
            control.check()?;
            let question = request
                .questions
                .get(id)
                .expect("question id came from the request map");
            let instruction = crate::families::wire::instruction_text(question)?;
            let unpacked = crate::families::wire::unpack_question(id, question)?;
            let rendered = self
                .inner
                .renderer
                .render(&state, &instruction, &unpacked.criteria)?;
            input_tokens = input_tokens
                .saturating_add(u64::from(self.inner.renderer.prompt_token_count(&rendered)));
            let logits = self
                .inner
                .model
                .letter_logits(rendered.prompt_ids(), rendered.letter_ids())?;
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
