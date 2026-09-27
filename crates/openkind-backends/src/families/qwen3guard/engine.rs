//! `DecisionEngine` adapter for the pinned qwen3guard profile.

use std::sync::Arc;

use openkind_core::{Answer, ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{Qwen3GuardModel, VerifiedArtifacts};
use super::renderer::Qwen3GuardRenderer;
use super::{
    Qwen3GuardEngineConfig, Qwen3GuardError, BACKBONE_ID, CLASS_CONTROVERSIAL, CLASS_SAFE,
    CLASS_UNSAFE, PROFILE_ID,
};

/// Loaded pinned guard engine.
pub struct Qwen3GuardEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: Qwen3GuardRenderer,
    model: Qwen3GuardModel,
}

/// Map an offered Choice label onto a risk-level class logit index.
///
/// `__none__` maps to the `Controversial` class, which the checkpoint
/// trains as the failure-to-decide mass; offering both `controversial` and
/// `__none__` fails closed as a duplicate class. Any other label is outside
/// the fixed preset's schema and is rejected at the wire.
fn class_index(label: &str, none_used: &mut bool) -> Result<usize, FamilyError> {
    match label.to_ascii_lowercase().as_str() {
        "__none__" | "controversial" => {
            if *none_used {
                return Err(FamilyError::InvalidInput(
                    "guard question offers both `controversial` and `__none__`; \
                     they map to the same class"
                        .to_owned(),
                ));
            }
            *none_used = true;
            Ok(CLASS_CONTROVERSIAL)
        }
        "safe" => Ok(CLASS_SAFE),
        "unsafe" => Ok(CLASS_UNSAFE),
        other => Err(FamilyError::InvalidInput(format!(
            "guard option `{other}` is outside the fixed preset schema; allowed labels are              safe/unsafe/controversial (or `__none__` for the controversial class)"
        ))),
    }
}

impl Qwen3GuardEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    pub fn load(config: Qwen3GuardEngineConfig) -> Result<BoundedFamilyEngine, Qwen3GuardError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = Qwen3GuardRenderer::load(&artifacts.tokenizer)?;
        let model = Qwen3GuardModel::load(&artifacts)?;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for Qwen3GuardEngine {
    fn backend_id(&self) -> &str {
        "qwen3guard/cpu-fp32"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} stream guardrail decision engine ({PROFILE_ID}, FP32 CPU)."
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

        let mut answers =
            std::collections::HashMap::<String, Answer, openkind_core::WireHashState>::with_capacity_and_hasher(
                question_ids.len(),
                Default::default(),
            );
        let mut input_tokens: u64 = 0;
        for id in &question_ids {
            control.check()?;
            let question = request
                .questions
                .get(id)
                .expect("question id came from the request map");
            let unpacked = crate::families::wire::unpack_question(id, question)?;
            let prompt_ids = self.inner.renderer.render(&state)?;
            input_tokens = input_tokens.saturating_add(prompt_ids.len() as u64);
            let logits = self.inner.model.risk_logits(&prompt_ids)?;
            let answer = match unpacked.primitive {
                crate::families::wire::QuestionPrimitive::Noul => {
                    // Noul maps to the Unsafe class probability; the
                    // Controversial mass is not surfaced in the scalar
                    // answer (serve a Choice question for the full
                    // distribution).
                    let probabilities =
                        temperature_softmax(&logits, super::CALIBRATION_TEMPERATURE)?;
                    Answer::Noul(openkind_core::NoulAnswer {
                        noul: probabilities[CLASS_UNSAFE],
                    })
                }
                crate::families::wire::QuestionPrimitive::Score => {
                    return Err(FamilyError::InvalidInput(format!(
                        "question `{id}` is a Score question; the guard profile has no ordinal \
                         contract and rejects it instead of inventing levels"
                    )));
                }
                crate::families::wire::QuestionPrimitive::Choice => {
                    // Every offered option must map onto a guard class; the
                    // readout is the softmax over the offered classes' logits.
                    let mut none_used = false;
                    let mut classes: Vec<usize> = Vec::with_capacity(unpacked.labels.len());
                    for label in &unpacked.labels {
                        classes.push(class_index(label, &mut none_used)?);
                    }
                    let offered_logits: Vec<f64> =
                        classes.iter().map(|index| logits[*index]).collect();
                    let probabilities =
                        temperature_softmax(&offered_logits, super::CALIBRATION_TEMPERATURE)?;
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
