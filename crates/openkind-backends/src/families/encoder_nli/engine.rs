//! `DecisionEngine` adapter for the pinned encoder-NLI profile.

use std::sync::Arc;

use candle_core::Device;
use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::device::FamilyExecution;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{EncoderNliModel, VerifiedArtifacts};
use super::renderer::EncoderNliRenderer;
use super::{
    EncoderNliEngineConfig, EncoderNliError, BACKBONE_ID, MAX_CANDIDATES, MIN_CANDIDATES,
    PROFILE_ID,
};

/// True when the caller supplied explicit Noul criteria rather than relying
/// on the wire defaults.
fn question_criteria_are_explicit(question: &openkind_core::Question) -> bool {
    match question {
        openkind_core::Question::Noul(noul) => noul.criteria.is_some(),
        _ => false,
    }
}

/// Three-way NLI logit source shared by the candle and ONNX models.
trait NliLogits: Send + Sync {
    /// Three-way NLI logits in checkpoint label order.
    fn nli_logits(&self, pair_ids: &[u32]) -> Result<[f64; 3], FamilyError>;
}

impl NliLogits for EncoderNliModel {
    fn nli_logits(&self, pair_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        EncoderNliModel::nli_logits(self, pair_ids)
    }
}

/// ONNX model readout bridging the shared logit source (feature `onnx`).
#[cfg(feature = "onnx")]
impl NliLogits for super::onnx::EncoderNliOnnxModel {
    fn nli_logits(&self, pair_ids: &[u32]) -> Result<[f64; 3], FamilyError> {
        super::onnx::EncoderNliOnnxModel::nli_logits(self, pair_ids)
    }
}

/// Loaded pinned encoder-NLI engine.
pub struct EncoderNliEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: EncoderNliRenderer,
    model: Box<dyn NliLogits>,
    backend_id: String,
}

impl EncoderNliEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The reference execution is FP32 CPU; see [`Self::load_with_execution`]
    /// for accelerated backends.
    pub fn load(config: EncoderNliEngineConfig) -> Result<BoundedFamilyEngine, EncoderNliError> {
        Self::load_with_execution(config, FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// Artifact verification (digests, pinned config) is identical on every
    /// backend; only the readout execution differs, and ONNX additionally
    /// requires `model.onnx` and its digest manifest in the model root.
    /// Loads fail closed when a backend is unavailable.
    pub fn load_with_execution(
        config: EncoderNliEngineConfig,
        execution: FamilyExecution,
    ) -> Result<BoundedFamilyEngine, EncoderNliError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = EncoderNliRenderer::load(&artifacts.vocab)?;
        let model: Box<dyn NliLogits> = match execution {
            FamilyExecution::Cpu => Box::new(EncoderNliModel::load(&artifacts, Device::Cpu)?),
            #[cfg(feature = "cuda")]
            FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device()?;
                Box::new(EncoderNliModel::load(&artifacts, device)?)
            }
            #[cfg(feature = "onnx")]
            FamilyExecution::Onnx { device_id } => {
                let acceleration = crate::onnx::OnnxAcceleration::from_onnx_execution(device_id)
                    .map_err(FamilyError::from)?;
                Box::new(super::onnx::EncoderNliOnnxModel::load(
                    &config.model_root,
                    acceleration,
                )?)
            }
            #[cfg(feature = "onnx-rocm")]
            FamilyExecution::OnnxRocm { device_id } => {
                Box::new(super::onnx::EncoderNliOnnxModel::load(
                    &config.model_root,
                    crate::onnx::OnnxAcceleration::Rocm { device_id },
                )?)
            }
        };
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                backend_id: format!("encoder-nli/{}", execution.id_fragment()),
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl Inner {
    /// Three-way NLI probabilities for one premise–hypothesis pair, together
    /// with the encoded pair token count for usage accounting.
    fn nli_probabilities(
        &self,
        premise: &str,
        hypothesis: &str,
    ) -> Result<([f64; 3], usize), FamilyError> {
        let pair_ids = self.renderer.encode_pair(premise, hypothesis)?;
        let logits = self.model.nli_logits(&pair_ids)?;
        let probabilities = temperature_softmax(&logits, 1.0)?;
        Ok((
            [
                probabilities[super::LABEL_ENTAILMENT],
                probabilities[super::LABEL_NEUTRAL],
                probabilities[super::LABEL_CONTRADICTION],
            ],
            pair_ids.len(),
        ))
    }

    /// Noul readout: entailment mass over decided (entailment vs
    /// contradiction) mass for one hypothesis statement.
    ///
    /// The hypothesis is the caller's `true` criterion when explicit Noul
    /// criteria are supplied, and the question instruction itself otherwise.
    /// Interrogative hypotheses leave most mass in the neutral class, which
    /// the ratio treats as undecided; statement hypotheses resolve sharply.
    fn noul_probability(
        &self,
        premise: &str,
        hypothesis: &str,
    ) -> Result<(f64, usize), FamilyError> {
        let (probabilities, tokens) = self.nli_probabilities(premise, hypothesis)?;
        let entailment = probabilities[super::LABEL_ENTAILMENT];
        let contradiction = probabilities[super::LABEL_CONTRADICTION];
        let decided = entailment + contradiction;
        if !decided.is_finite() || decided <= f64::MIN_POSITIVE {
            return Err(FamilyError::Numerical(format!(
                "noul hypothesis produced no decidable entailment/contradiction mass: {decided}"
            )));
        }
        Ok((entailment / decided, tokens))
    }
}

impl FamilyEvaluator for EncoderNliEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} premise-hypothesis entailment engine ({PROFILE_ID}, {}).",
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
            let instruction = crate::families::wire::instruction_text(question)?;
            let answer = match unpacked.primitive {
                crate::families::wire::QuestionPrimitive::Noul => {
                    // The family contract maps Noul to the entailment mass of
                    // the true-criterion hypothesis over the decided mass;
                    // the false criterion is not scored.
                    let true_criterion = unpacked
                        .labels
                        .iter()
                        .position(|label| label == "true")
                        .and_then(|index| unpacked.criteria.get(index))
                        .expect("noul unpacking provides a true criterion");
                    let hypothesis = if question_criteria_are_explicit(question) {
                        true_criterion.clone()
                    } else {
                        instruction.clone()
                    };
                    let (probability, tokens) = self.inner.noul_probability(&state, &hypothesis)?;
                    input_tokens = input_tokens.saturating_add(tokens as u64);
                    openkind_core::Answer::Noul(openkind_core::NoulAnswer { noul: probability })
                }
                _primitive => {
                    let mut entailments = Vec::with_capacity(unpacked.criteria.len());
                    for criterion in &unpacked.criteria {
                        control.check()?;
                        let (probability, tokens) = self
                            .inner
                            .nli_probabilities(&state, criterion)
                            .map(|(probabilities, tokens)| {
                                (probabilities[super::LABEL_ENTAILMENT], tokens)
                            })?;
                        input_tokens = input_tokens.saturating_add(tokens as u64);
                        entailments.push(probability);
                    }
                    // The profile temperature calibrates the decision
                    // distribution over candidates, applied to the
                    // log-entailment logits. The label-level NLI softmax is
                    // not rescaled.
                    let logits: Vec<f64> = entailments
                        .iter()
                        .map(|probability| {
                            if *probability <= f64::MIN_POSITIVE {
                                f64::NEG_INFINITY
                            } else {
                                probability.ln()
                            }
                        })
                        .collect();
                    let probabilities =
                        temperature_softmax(&logits, super::CALIBRATION_TEMPERATURE)?;
                    crate::families::wire::answer_from_probabilities(&unpacked, &probabilities)?
                }
            };
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
