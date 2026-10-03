//! Decision-engine adapter for the Clef family.
//!
//! Every request is one independent full-sequence forward: render the wire
//! request into the reference record encoding, run the backbone once, feed
//! the final-norm hidden states through the joint schema head, and map each
//! question's option distribution back onto the Jev wire answer. Nothing is
//! retained between requests and no token is sampled.

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{Answer, ModelInfo, State, SystemRequest, SystemResponse, WireHashState};

use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
    FamilyLimits,
};

use crate::families::wire::{answer_from_probabilities, unpack_question, QuestionPrimitive};

use super::gguf::GgufModel;
use super::model::{ClefExecutionModel, ClefModel, VerifiedArtifacts};
use super::renderer::{clef_render_value, ClefQuestionType, ClefRenderer};
use super::{ClefExecution, ClefProfile};

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use super::mlx::MlxClefModel;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

/// Blocking Clef evaluator over one digest-verified model.
pub struct ClefEngine {
    profile: &'static ClefProfile,
    model: Arc<dyn ClefExecutionModel>,
    renderer: ClefRenderer,
}

impl ClefEngine {
    /// Load the pinned profile from a digest-verified model root.
    pub fn load(
        model_root: impl AsRef<std::path::Path>,
        profile: &'static ClefProfile,
        limits: FamilyLimits,
    ) -> Result<BoundedFamilyEngine, FamilyError> {
        let model_root = model_root.as_ref();
        let model: Arc<dyn ClefExecutionModel> = match profile.execution {
            ClefExecution::CandleBf16 => {
                let artifacts = VerifiedArtifacts::verify(model_root, profile)?;
                Arc::new(ClefModel::load(&artifacts)?)
            }
            ClefExecution::CandleGguf => Arc::new(GgufModel::load(model_root, profile)?),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            ClefExecution::Mlx4Bit => {
                // The MLX loader performs its own artifact verification (the
                // quantized repo ships no safetensors index json).
                let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
                Arc::new(MlxClefModel::load(model_root, profile, runtime)?)
            }
            #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
            ClefExecution::Mlx4Bit => {
                return Err(FamilyError::ExecutionUnavailable(
                    "the clef-flash-mlx-4bit profile requires the `mlx` feature on macOS arm64"
                        .to_owned(),
                ))
            }
        };
        let renderer = ClefRenderer::load(
            &model_root.join(profile.tokenizer_path),
            profile.operational_context_tokens,
        )?;
        Ok(Self::from_shared(model, renderer, profile, limits))
    }

    /// Wrap an already-loaded model (fixture and test entry point).
    pub(crate) fn from_shared(
        model: Arc<dyn ClefExecutionModel>,
        renderer: ClefRenderer,
        profile: &'static ClefProfile,
        limits: FamilyLimits,
    ) -> BoundedFamilyEngine {
        let evaluator: Arc<dyn FamilyEvaluator> = Arc::new(Self {
            profile,
            model,
            renderer,
        });
        BoundedFamilyEngine::new(evaluator, limits)
    }
}

impl FamilyEvaluator for ClefEngine {
    fn backend_id(&self) -> &str {
        self.profile.cpu_backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: format!("{}:{}", self.profile.loader_id, self.profile.profile_id),
            description: self.profile.description.to_owned(),
            release_date: self.profile.release_date.to_owned(),
        }
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        let (input_tokens, answers) = self.evaluate_request(&request, control)?;
        Ok(SystemResponse {
            model: request.model,
            answers,
            usage: openkind_core::Usage {
                input_tokens: u32::try_from(input_tokens).unwrap_or(u32::MAX),
                output_tokens: 0,
            },
        })
    }
}

/// One rendered question with options in
/// evaluation order as `(option_id, description)`.
pub(crate) struct RenderQuestion {
    pub(crate) id: String,
    pub(crate) question_type: ClefQuestionType,
    pub(crate) instruction: String,
    pub(crate) options: Vec<(String, String)>,
}

impl ClefEngine {
    /// Full evaluation: render, forward, read out, and map answers.
    fn evaluate_request(
        &self,
        request: &SystemRequest,
        control: &FamilyControl,
    ) -> Result<(usize, HashMap<String, Answer, WireHashState>), FamilyError> {
        evaluate_with(&*self.model, &self.renderer, request, control)
    }
}

/// Render, forward, read out, and map answers; returns the encoded input
/// token count and one wire answer per question.
fn evaluate_with(
    model: &dyn ClefExecutionModel,
    renderer: &ClefRenderer,
    request: &SystemRequest,
    control: &FamilyControl,
) -> Result<(usize, HashMap<String, Answer, WireHashState>), FamilyError> {
    control.check()?;
    // Questions render in sorted-id order: the wire map is unordered, and a
    // deterministic order keeps the joint head's cross-question attention
    // reproducible.
    let mut sorted: Vec<(&String, &openkind_core::Question)> = request.questions.iter().collect();
    sorted.sort_by(|left, right| left.0.cmp(right.0));
    let mut unpacked: Vec<(String, crate::families::wire::UnpackedQuestion)> =
        Vec::with_capacity(sorted.len());
    // One rendered question: (id, type, instruction, options) with options
    // in evaluation order as (option_id, description).
    let mut render_questions: Vec<RenderQuestion> = Vec::with_capacity(sorted.len());
    for (id, question) in &sorted {
        let unpack = unpack_question(id, question)?;
        let clef_type = match unpack.primitive {
            QuestionPrimitive::Noul => ClefQuestionType::Noul,
            QuestionPrimitive::Choice => ClefQuestionType::Choice,
            QuestionPrimitive::Score => ClefQuestionType::Score,
        };
        let instruction = instruction_text(question);
        // Clef scores noul options in (true, false) order; choice options
        // sort lexicographically (the unpacked order); score options are
        // positional.
        let options: Vec<(String, String)> = match unpack.primitive {
            QuestionPrimitive::Noul => vec![
                ("true".into(), unpack.criteria[1].clone()),
                ("false".into(), unpack.criteria[0].clone()),
            ],
            QuestionPrimitive::Choice | QuestionPrimitive::Score => unpack
                .labels
                .iter()
                .zip(unpack.criteria.iter())
                .map(|(label, criterion)| (label.clone(), criterion.clone()))
                .collect(),
        };
        render_questions.push(RenderQuestion {
            id: unpack.id.clone(),
            question_type: clef_type,
            instruction,
            options,
        });
        unpacked.push(((*id).clone(), unpack));
    }
    let state_value = state_json(request);
    let encoded = renderer.encode(&state_value, &render_questions)?;
    let logits = model.evaluate_record(&encoded, control)?;
    if logits.len() != encoded.questions.len() {
        return Err(FamilyError::InvalidInput(
            "joint head returned a question count that does not match the encoded record".into(),
        ));
    }
    let mut answers =
        HashMap::with_capacity_and_hasher(request.questions.len(), Default::default());
    for ((question_id, unpack), (question, question_logits)) in
        unpacked.iter().zip(encoded.questions.iter().zip(&logits))
    {
        debug_assert_eq!(question_id, &question.question_id);
        // Softmax at temperature 1.0: the reference applies no calibration.
        let mut probabilities = temperature_softmax(question_logits, 1.0)?;
        // Align clef's evaluation order with the wire candidate order: noul
        // renders (true, false) but unpacks to [false, true]; choice and
        // score orders already match.
        if unpack.primitive == QuestionPrimitive::Noul {
            probabilities.reverse();
        }
        let answer = answer_from_probabilities(unpack, &probabilities)?;
        answers.insert(question_id.clone(), answer);
    }
    Ok((encoded.input_ids.len(), answers))
}

/// Instruction text exactly as the reference `encode_record` renders it:
/// absent or empty instructions fall back to the question id; strings pass
/// through; other payloads render as compact sorted JSON.
fn instruction_text(question: &openkind_core::Question) -> String {
    let instructions = match question {
        openkind_core::Question::Noul(question) => &question.instructions,
        openkind_core::Question::Choice(question) => &question.instructions,
        openkind_core::Question::Score(question) => &question.instructions,
    };
    match instructions {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(text) => text.clone(),
        other => clef_render_value(other).unwrap_or_default(),
    }
}

/// The wire state as the JSON value the renderer serializes: text states
/// pass through as strings, structured states as their JSON value.
fn state_json(request: &SystemRequest) -> serde_json::Value {
    match &request.state {
        State::Text(text) => serde_json::Value::String(text.clone()),
        State::Object(map) => serde_json::Value::Object(map.clone()),
        State::Array(items) => serde_json::Value::Array(items.clone()),
    }
}
