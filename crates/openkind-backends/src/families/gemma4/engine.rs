//! `DecisionEngine` adapter for the pinned gemma4-decision profile.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::Device;
use openkind_core::{ModelInfo, Question, State, SystemRequest, SystemResponse, Usage};

use crate::device::FamilyExecution;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::backbone::{Gemma4TextConfig, Gemma4TextModel};
use super::gguf::{load_checkpoint, VerifiedArtifacts};
use super::renderer::WinnowRenderer;

use super::{Gemma4EngineConfig, Gemma4Error, BACKBONE_ID, PROFILE_ID};

/// Loaded pinned gemma4-decision engine.
pub struct Gemma4DecisionEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: WinnowRenderer,
    model: Gemma4TextModel,
    backend_id: String,
}

impl Gemma4DecisionEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The reference execution is candle's quantized CPU path: compute
    /// weights run as q8_0 `QMatMul` kernels and the giant embedding
    /// tables stream their rows from the verified GGUF, keeping the
    /// resident footprint near the on-disk artifact size. Every question is
    /// an independent full-sequence forward. See [`Self::load_with_execution`]
    /// for accelerated backends.
    pub fn load(config: Gemma4EngineConfig) -> Result<BoundedFamilyEngine, Gemma4Error> {
        Self::load_with_execution(config, FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// Artifact verification (digests, pinned GGUF metadata, letter-token
    /// contract) is identical on every backend; only the execution device
    /// differs. Loads fail closed when a backend is unavailable.
    pub fn load_with_execution(
        config: Gemma4EngineConfig,
        execution: FamilyExecution,
    ) -> Result<BoundedFamilyEngine, Gemma4Error> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let gguf_letters =
            super::gguf::gguf_letter_token_ids(&artifacts, super::renderer::LETTER_MAX_OPTIONS)?;
        let renderer = WinnowRenderer::load(
            &artifacts.tokenizer,
            super::MAX_SEQUENCE_TOKENS,
            Some(&gguf_letters),
        )?;
        let device = match execution {
            FamilyExecution::Cpu => Device::Cpu,
            #[cfg(feature = "cuda")]
            FamilyExecution::Cuda { .. } => execution.candle_device()?,
            #[cfg(feature = "onnx")]
            FamilyExecution::Onnx { .. } => {
                return Err(FamilyError::InvalidInput(
                    "gemma4-decision has no ONNX execution path".to_owned(),
                )
                .into());
            }
        };
        let cfg = Gemma4TextConfig::winnow_e4b();
        let checkpoint = load_checkpoint(&artifacts, &cfg, &device)?;
        let model = Gemma4TextModel::new(&cfg, &checkpoint).map_err(FamilyError::from)?;
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                backend_id: format!("gemma4-decision/{}", execution.id_fragment()),
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

/// Raw `instructions` payload of a wire question, rendered by the Winnow
/// protocol as JSON (a string instruction serializes with quotes).
fn instruction_value(question: &Question) -> &serde_json::Value {
    match question {
        Question::Noul(question) => &question.instructions,
        Question::Choice(question) => &question.instructions,
        Question::Score(question) => &question.instructions,
    }
}

impl FamilyEvaluator for Gemma4DecisionEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} Gemma 4 letter-logit decision engine ({PROFILE_ID}, {}).",
                self.inner.backend_id
            ),
            release_date: "2026-10-01".into(),
        }
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        control.check()?;
        let state = match &request.state {
            State::Text(text) => serde_json::Value::String(text.clone()),
            State::Object(map) => serde_json::Value::Object(map.clone()),
            State::Array(items) => serde_json::Value::Array(items.clone()),
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
            let unpacked = crate::families::wire::unpack_question(id, question)?;
            let rendered = self.inner.renderer.render(
                &state,
                instruction_value(question),
                &unpacked.labels,
                &unpacked.criteria,
                unpacked.ordered,
            )?;
            input_tokens = input_tokens
                .saturating_add(u64::from(self.inner.renderer.prompt_token_count(&rendered)));
            let prompt_ids = rendered.prompt_ids();
            let logits = self
                .inner
                .model
                .letter_logits(&prompt_ids, rendered.letter_ids())?;
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
