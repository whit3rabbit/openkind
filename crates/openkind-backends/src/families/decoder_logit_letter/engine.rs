//! `DecisionEngine` adapter for the pinned decoder-letter profile.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::Device;
use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::device::FamilyExecution;
use crate::families::letter_renderer::LetterRenderer;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{DecoderLetterModel, VerifiedArtifacts};

use super::{DecoderLetterEngineConfig, DecoderLetterError, BACKBONE_ID, PROFILE_ID};

/// Letter-logit source shared by the candle and ONNX models.
trait LetterLogits: Send + Sync {
    /// Next-token logits at the answer slot, restricted to `letter_ids`.
    fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<Vec<f64>, FamilyError>;
}

impl LetterLogits for DecoderLetterModel {
    fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<Vec<f64>, FamilyError> {
        DecoderLetterModel::letter_logits(self, prompt_ids, letter_ids)
    }
}

/// ONNX model readout bridging the shared logit source (feature `onnx`).
#[cfg(feature = "onnx")]
impl LetterLogits for super::onnx::DecoderLetterOnnxModel {
    fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<Vec<f64>, FamilyError> {
        super::onnx::DecoderLetterOnnxModel::letter_logits(self, prompt_ids, letter_ids)
    }
}

/// Loaded pinned decoder-letter engine.
pub struct DecoderLetterEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: LetterRenderer,
    model: Box<dyn LetterLogits>,
    backend_id: String,
}

impl DecoderLetterEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The returned adapter implements [`DecisionEngine`] and carries
    /// admission, queue, deadline, and cancellation control. The reference
    /// execution is FP32 CPU; see [`Self::load_with_execution`] for
    /// accelerated backends.
    pub fn load(
        config: DecoderLetterEngineConfig,
    ) -> Result<BoundedFamilyEngine, DecoderLetterError> {
        Self::load_with_execution(config, FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// Artifact verification (digests, pinned config) is identical on every
    /// backend; only the readout execution differs, and ONNX additionally
    /// requires `model.onnx` and its digest manifest in the model root.
    /// Loads fail closed when a backend is unavailable.
    pub fn load_with_execution(
        config: DecoderLetterEngineConfig,
        execution: FamilyExecution,
    ) -> Result<BoundedFamilyEngine, DecoderLetterError> {
        let artifacts = VerifiedArtifacts::verify_for_execution(&config.model_root, execution)?;
        let renderer = LetterRenderer::load(&artifacts.tokenizer, super::MAX_SEQUENCE_TOKENS)?;
        let model: Box<dyn LetterLogits> = match execution {
            FamilyExecution::Cpu => Box::new(DecoderLetterModel::load(&artifacts, Device::Cpu)?),
            #[cfg(feature = "cuda")]
            FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device().map_err(FamilyError::from)?;
                Box::new(DecoderLetterModel::load(&artifacts, device)?)
            }
            #[cfg(feature = "onnx")]
            FamilyExecution::Onnx { device_id } => {
                let acceleration = crate::onnx::OnnxAcceleration::from_onnx_execution(device_id)
                    .map_err(FamilyError::from)?;
                Box::new(super::onnx::DecoderLetterOnnxModel::load(
                    &config.model_root,
                    acceleration,
                )?)
            }
            #[cfg(feature = "onnx-rocm")]
            FamilyExecution::OnnxRocm { device_id } => {
                Box::new(super::onnx::DecoderLetterOnnxModel::load(
                    &config.model_root,
                    crate::onnx::OnnxAcceleration::Rocm { device_id },
                )?)
            }
        };
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                backend_id: format!("decoder-logit-letter/{}", execution.id_fragment()),
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for DecoderLetterEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} letter-logit decision engine ({PROFILE_ID}, {}).",
                self.inner.backend_id
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
