//! `DecisionEngine` adapter for the pinned strands-decider profile.
//!
//! Wire mapping mirrors the reference `strands_decider` serving code:
//! `Noul` scores the two options `false`/`true` (annotated with the caller's
//! criteria when supplied, the reference defaults otherwise) and reports
//! `p(true)`; `Choice` scores one option per offered key in the shared
//! sorted order; `Score` scores one option per ordered level and reports the
//! expected level. The option softmax is temperature-calibrated per
//! primitive and mapped back through the shared family wire helpers.

use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::decoder_logit_qwen35::QuestionKind;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{StrandsDeciderModel, VerifiedArtifacts};
use super::renderer::{question_layout, StrandsRenderer};
use super::{
    StrandsDeciderEngineConfig, StrandsDeciderError, BACKBONE_ID, BASE_MODEL_ID, CALIBRATION,
};

/// Loaded pinned strands-decider engine.
pub struct StrandsDeciderEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: StrandsRenderer,
    model: StrandsDeciderModel,
    backend_id: String,
}

/// The raw instruction payload of any question primitive.
fn question_instructions(question: &openkind_core::Question) -> &serde_json::Value {
    match question {
        openkind_core::Question::Noul(noul) => &noul.instructions,
        openkind_core::Question::Choice(choice) => &choice.instructions,
        openkind_core::Question::Score(score) => &score.instructions,
    }
}

/// True when the caller supplied explicit Noul criteria rather than relying
/// on the wire defaults.
fn noul_criteria_are_explicit(question: &openkind_core::Question) -> bool {
    match question {
        openkind_core::Question::Noul(noul) => noul.criteria.is_some(),
        _ => false,
    }
}

impl Inner {
    /// Calibrated option probabilities for one question, plus the encoded
    /// row's token count: the state document plus this question's block as
    /// one causal row.
    fn probabilities(
        &self,
        state: &serde_json::Value,
        unpacked: &crate::families::wire::UnpackedQuestion,
        question: &openkind_core::Question,
        control: &FamilyControl,
    ) -> Result<(Vec<f64>, usize), FamilyError> {
        control.check()?;
        let layout = question_layout(unpacked, noul_criteria_are_explicit(question));
        let row = self
            .renderer
            .encode_row(state, &layout, question_instructions(question))?;
        let option_count = layout.pairs.len();
        let logits = self.model.rows_logits(&row, control)?;
        if logits.len() != option_count {
            return Err(FamilyError::ContractMismatch {
                field: "option.count",
                expected: format!("{option_count} option logits"),
                actual: format!("{} logits", logits.len()),
            });
        }
        let temperature = CALIBRATION.resolve(match unpacked.primitive {
            crate::families::wire::QuestionPrimitive::Noul => QuestionKind::Noul,
            crate::families::wire::QuestionPrimitive::Choice => QuestionKind::Choice,
            crate::families::wire::QuestionPrimitive::Score => QuestionKind::Score,
        });
        let probabilities = temperature_softmax(&logits, temperature)?;
        Ok((probabilities, row.ids.len()))
    }
}

impl StrandsDeciderEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    pub fn load(
        config: StrandsDeciderEngineConfig,
    ) -> Result<BoundedFamilyEngine, StrandsDeciderError> {
        Self::load_with_execution(config, crate::device::FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// The pointer-head readout over per-row hidden states has no ONNX
    /// export; accelerated loads require the `cuda` feature.
    pub fn load_with_execution(
        config: StrandsDeciderEngineConfig,
        execution: crate::device::FamilyExecution,
    ) -> Result<BoundedFamilyEngine, StrandsDeciderError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root, &config.base_root)?;
        let renderer = StrandsRenderer::load(&artifacts.tokenizer)?;
        let backend_id = match execution {
            crate::device::FamilyExecution::Cpu => "strands-decider-2b/cpu-fp32".to_owned(),
            #[cfg(feature = "cuda")]
            crate::device::FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device().map_err(FamilyError::from)?;
                let model = StrandsDeciderModel::load_with_device(&artifacts, device)?;
                let engine = Self {
                    inner: Arc::new(Inner {
                        renderer,
                        model,
                        backend_id: "strands-decider-2b/cuda-fp32".to_owned(),
                    }),
                };
                return Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits));
            }
            #[cfg(feature = "onnx")]
            crate::device::FamilyExecution::Onnx { .. } => {
                return Err(FamilyError::ExecutionUnavailable(
                    "the strands-decider pointer-head readout over the Qwen3.5 hybrid backbone \
                     has no ONNX export; select cpu or (with the `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
            #[cfg(feature = "onnx-rocm")]
            crate::device::FamilyExecution::OnnxRocm { .. } => {
                return Err(FamilyError::ExecutionUnavailable(
                    "the strands-decider pointer-head readout over the Qwen3.5 hybrid backbone \
                     has no ONNX export and therefore no ROCm execution; select cpu or (with \
                     the `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
        };
        let model = StrandsDeciderModel::load(&artifacts)?;
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

impl FamilyEvaluator for StrandsDeciderEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} pointer decision engine on {BASE_MODEL_ID} \
                 ({}).",
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
            State::Text(text) => {
                if text.trim().is_empty() {
                    return Err(FamilyError::InvalidInput(
                        "state must contain non-whitespace text".to_owned(),
                    ));
                }
                serde_json::Value::String(text.clone())
            }
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
            let (probabilities, row_tokens) = self
                .inner
                .probabilities(&state, &unpacked, question, control)?;
            input_tokens = input_tokens.saturating_add(row_tokens as u64);
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
