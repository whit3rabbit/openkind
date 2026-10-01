//! `DecisionEngine` adapter for the pinned decoder-logit-llm (GGUF) profile.

use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::letter_renderer::LetterRenderer;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{DecoderLlmModel, VerifiedArtifacts};
use super::{DecoderLlmEngineConfig, DecoderLlmError, BACKBONE_ID, GGUF_VARIANT, PROFILE_ID};

/// Loaded pinned decoder-logit-llm engine.
pub struct DecoderLlmEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: LetterRenderer,
    model: DecoderLlmModel,
    backend_id: String,
}

impl DecoderLlmEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    pub fn load(config: DecoderLlmEngineConfig) -> Result<BoundedFamilyEngine, DecoderLlmError> {
        Self::load_with_execution(config, crate::device::FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// The quantized GGUF readout has no ONNX counterpart; accelerated
    /// loads require the `cuda` feature.
    pub fn load_with_execution(
        config: DecoderLlmEngineConfig,
        execution: crate::device::FamilyExecution,
    ) -> Result<BoundedFamilyEngine, DecoderLlmError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = LetterRenderer::load(&artifacts.tokenizer, super::MAX_SEQUENCE_TOKENS)?;
        let backend_id = match execution {
            crate::device::FamilyExecution::Cpu => "decoder-logit-llm/cpu-q8_0".to_owned(),
            #[cfg(feature = "cuda")]
            crate::device::FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device()?;
                let model = DecoderLlmModel::load_with_device(&artifacts, device)?;
                let engine = Self {
                    inner: Arc::new(Inner {
                        renderer,
                        model,
                        backend_id: "decoder-logit-llm/cuda-q8_0".to_owned(),
                    }),
                };
                return Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits));
            }
            #[cfg(feature = "onnx")]
            crate::device::FamilyExecution::Onnx { .. } => {
                return Err(FamilyError::ExecutionUnavailable(
                    "the quantized GGUF readout has no ONNX export; select cpu or (with the \
                     `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
        };
        let model = DecoderLlmModel::load(&artifacts)?;
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                backend_id,
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for DecoderLlmEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} ({GGUF_VARIANT}) letter-logit decision engine ({PROFILE_ID})."
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
