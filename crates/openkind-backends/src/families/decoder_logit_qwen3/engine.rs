//! `DecisionEngine` adapter for the pinned `decoder-logit-qwen3` controls.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::Device;
use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::device::FamilyExecution;
use crate::families::decoder_logit_qwen35::QuestionKind;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};
use crate::families::wire;

use super::model::{Qwen3ControlModel, VerifiedArtifacts};
use super::renderer::NOUL_DEFAULT_DESCRIPTIONS;
use super::{DecoderLogitQwen3EngineConfig, DecoderLogitQwen3Error, Qwen3LogitProfile};

/// Letter-logit source shared by the candle and ONNX models.
trait LetterLogits: Send + Sync {
    /// Next-token logits at the answer slot, restricted to `letter_ids`.
    fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError>;
}

impl LetterLogits for Qwen3ControlModel {
    fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        Qwen3ControlModel::letter_logits(self, prompt_ids, letter_ids, control)
    }
}

/// ONNX model readout bridging the shared logit source (feature `onnx`).
#[cfg(feature = "onnx")]
impl LetterLogits for super::onnx::Qwen3ControlOnnxModel {
    fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        super::onnx::Qwen3ControlOnnxModel::letter_logits(self, prompt_ids, letter_ids, control)
    }
}

/// Loaded pinned raw-control engine for one profile of the family.
pub struct DecoderLogitQwen3Engine {
    inner: Arc<Inner>,
}

struct Inner {
    profile: &'static Qwen3LogitProfile,
    renderer: super::renderer::Qwen3ControlRenderer,
    model: Box<dyn LetterLogits>,
    backend_id: String,
}

impl FamilyEvaluator for DecoderLogitQwen3Engine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} raw letter-logit control ({}, {}, temperature 1.0).",
                self.inner.profile.backbone_id,
                self.inner.profile.profile_id,
                self.inner.backend_id
            ),
            release_date: self.inner.profile.release_date.into(),
        }
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        let profile = self.inner.profile;
        control.check()?;
        let state: serde_json::Value = match &request.state {
            State::Text(text) => serde_json::Value::String(text.clone()),
            State::Object(map) => serde_json::to_value(map).map_err(|error| {
                FamilyError::InvalidInput(format!("state serialization failed: {error}"))
            })?,
            State::Array(items) => serde_json::to_value(items).map_err(|error| {
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
            let criterion = wire::instruction_text(question)?;
            // Noul keeps the protocol's default criterion text when the
            // caller supplied none; Choice and Score use the shared wire
            // unpacking.
            let unpacked = match question {
                openkind_core::Question::Noul(noul) => {
                    let (false_description, true_description) = noul
                        .criteria
                        .as_ref()
                        .map(|criteria| (criteria.r#false.clone(), criteria.r#true.clone()))
                        .unwrap_or_else(|| {
                            (
                                NOUL_DEFAULT_DESCRIPTIONS[0].to_owned(),
                                NOUL_DEFAULT_DESCRIPTIONS[1].to_owned(),
                            )
                        });
                    wire::UnpackedQuestion {
                        id: id.clone(),
                        primitive: wire::QuestionPrimitive::Noul,
                        labels: vec!["false".into(), "true".into()],
                        criteria: vec![false_description, true_description],
                        ordered: false,
                    }
                }
                other => wire::unpack_question(id, other)?,
            };
            let options: Vec<(String, String)> = unpacked
                .labels
                .iter()
                .cloned()
                .zip(unpacked.criteria.iter().cloned())
                .collect();
            if options.len() > super::MAX_OPTIONS_PER_PASS {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` offers {} candidates but the raw control serves at most {} \
                     in a single pass; split the question",
                    options.len(),
                    super::MAX_OPTIONS_PER_PASS
                )));
            }
            let token_count = std::cell::Cell::new(0_u64);
            // The protocol renders Noul options true-first; our wire
            // unpacking is false-first. Render in protocol order and map the
            // distribution back to the wire's candidate order.
            let mut options = options;
            if matches!(unpacked.primitive, wire::QuestionPrimitive::Noul) {
                options.reverse();
            }
            control.check()?;
            let pass = self
                .inner
                .renderer
                .render_pass(&state, &criterion, &options)?;
            token_count.set(u64::from(self.inner.renderer.pass_token_count(&pass)));
            let logits =
                self.inner
                    .model
                    .letter_logits(pass.prompt_ids(), pass.letter_ids(), control)?;
            let mut probabilities = temperature_softmax(
                &logits,
                profile.calibration().resolve(match unpacked.primitive {
                    wire::QuestionPrimitive::Choice => QuestionKind::Choice,
                    wire::QuestionPrimitive::Score => QuestionKind::Score,
                    wire::QuestionPrimitive::Noul => QuestionKind::Noul,
                }),
            )?;
            if matches!(unpacked.primitive, wire::QuestionPrimitive::Noul) {
                probabilities.reverse();
            }
            input_tokens = input_tokens.saturating_add(token_count.get());
            let answer = wire::answer_from_probabilities(&unpacked, &probabilities)?;
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

impl DecoderLogitQwen3Engine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The returned adapter implements [`DecisionEngine`] and carries
    /// admission, queue, deadline, and cancellation control. The reference
    /// execution is FP32 CPU; see [`Self::load_with_execution`] for
    /// accelerated backends.
    pub fn load(
        config: DecoderLogitQwen3EngineConfig,
    ) -> Result<BoundedFamilyEngine, DecoderLogitQwen3Error> {
        Self::load_with_execution(config, FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// Artifact verification (digests, pinned config) is identical on every
    /// backend; only the readout execution differs, and ONNX additionally
    /// requires `model.onnx` and its digest manifest in the model root.
    /// Loads fail closed when a backend is unavailable.
    pub fn load_with_execution(
        config: DecoderLogitQwen3EngineConfig,
        execution: FamilyExecution,
    ) -> Result<BoundedFamilyEngine, DecoderLogitQwen3Error> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root, config.profile)?;
        let renderer = super::renderer::Qwen3ControlRenderer::load(
            &artifacts.tokenizer,
            config.profile.assistant_tail,
        )?;
        let model: Box<dyn LetterLogits> = match execution {
            FamilyExecution::Cpu => Box::new(Qwen3ControlModel::load(
                &artifacts,
                config.profile,
                Device::Cpu,
            )?),
            #[cfg(feature = "cuda")]
            FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device()?;
                Box::new(Qwen3ControlModel::load(&artifacts, config.profile, device)?)
            }
            #[cfg(feature = "onnx")]
            FamilyExecution::Onnx { device_id } => {
                let acceleration = crate::onnx::OnnxAcceleration::from_onnx_execution(device_id)
                    .map_err(FamilyError::from)?;
                Box::new(super::onnx::Qwen3ControlOnnxModel::load(
                    &config.model_root,
                    acceleration,
                )?)
            }
        };
        // Reproduces `profile.cpu_backend_id` byte-for-byte on the CPU
        // reference execution.
        let backend_id = format!("{}/{}", config.profile.loader_id, execution.id_fragment());
        let engine = Self {
            inner: Arc::new(Inner {
                profile: config.profile,
                renderer,
                model,
                backend_id,
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}
