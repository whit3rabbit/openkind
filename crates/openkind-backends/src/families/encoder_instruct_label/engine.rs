//! `DecisionEngine` adapter for the pinned encoder-instruct-label profile.

use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{EncoderInstructLabelModel, VerifiedArtifacts};
use super::renderer::EncoderInstructLabelRenderer;
use super::{
    EncoderInstructLabelEngineConfig, EncoderInstructLabelError, BACKBONE_ID,
    CALIBRATION_TEMPERATURE, MAX_CANDIDATES, PROFILE_ID,
};

/// Loaded pinned label-marker engine.
pub struct EncoderInstructLabelEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: EncoderInstructLabelRenderer,
    model: EncoderInstructLabelModel,
}

/// Numerically stable temperature-calibrated logistic sigmoid.
fn calibrated_sigmoid(logit: f64, temperature: f64) -> Result<f64, FamilyError> {
    if !logit.is_finite() {
        return Err(FamilyError::Numerical(format!(
            "marker logit is not finite: {logit}"
        )));
    }
    let scaled = logit / temperature;
    if scaled >= 0.0 {
        Ok(1.0 / (1.0 + (-scaled).exp()))
    } else {
        let exp = scaled.exp();
        Ok(exp / (1.0 + exp))
    }
}

/// Renormalize calibrated sigmoid supports into a conditional distribution.
fn renormalize(supports: &[f64]) -> Result<Vec<f64>, FamilyError> {
    let sum: f64 = supports.iter().sum();
    if !sum.is_finite() || sum <= f64::MIN_POSITIVE {
        return Err(FamilyError::Numerical(format!(
            "offered options produced no decidable support mass: {sum}"
        )));
    }
    Ok(supports.iter().map(|support| support / sum).collect())
}

impl EncoderInstructLabelEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    pub fn load(
        config: EncoderInstructLabelEngineConfig,
    ) -> Result<BoundedFamilyEngine, EncoderInstructLabelError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = EncoderInstructLabelRenderer::load(&artifacts.tokenizer)?;
        let model = EncoderInstructLabelModel::load(&artifacts)?;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl Inner {
    /// Raw marker logits for one question: render every candidate marker
    /// plus the state, run the single forward pass, and return one logit
    /// per marker in candidate order. Also returns the encoded token count.
    fn candidate_logits(
        &self,
        markers: &[String],
        state: &str,
    ) -> Result<(Vec<f64>, usize), FamilyError> {
        let ids = self.renderer.render(markers, state)?;
        let count = ids.len();
        let logits = self.model.marker_logits(&ids)?;
        if logits.len() != markers.len() {
            return Err(FamilyError::ContractMismatch {
                field: "marker.count",
                expected: format!("{} marker logits", markers.len()),
                actual: format!("{} logits", logits.len()),
            });
        }
        Ok((logits, count))
    }
}

/// The proposition marker for a `Noul` question: the caller's `true`
/// criterion when explicit criteria are supplied, the question instruction
/// verbatim otherwise. The false criterion is not scored — the readout is
/// the support sigmoid of the proposition itself, so default meta-criteria
/// that merely restate the proposition would invert it.
fn noul_marker(
    unpacked: &crate::families::wire::UnpackedQuestion,
    instruction: &str,
    explicit: bool,
) -> String {
    if explicit {
        unpacked
            .criteria
            .iter()
            .zip(unpacked.labels.iter())
            .find(|(_, label)| label.as_str() == "true")
            .map(|(criterion, _)| criterion.clone())
            .expect("noul unpacking provides a true criterion")
    } else {
        instruction.to_owned()
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

impl FamilyEvaluator for EncoderInstructLabelEngine {
    fn backend_id(&self) -> &str {
        "encoder-instruct-label/cpu-fp32"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} label-marker decision engine ({PROFILE_ID}, FP32 CPU)."
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
            let instruction = crate::families::wire::instruction_text(question)?;
            let explicit_noul_criteria = noul_criteria_are_explicit(question);
            let markers: Vec<String> = match unpacked.primitive {
                crate::families::wire::QuestionPrimitive::Noul => {
                    vec![noul_marker(&unpacked, &instruction, explicit_noul_criteria)]
                }
                crate::families::wire::QuestionPrimitive::Choice
                | crate::families::wire::QuestionPrimitive::Score => unpacked.criteria.clone(),
            };
            let (logits, tokens) = self.inner.candidate_logits(&markers, &state)?;
            input_tokens = input_tokens.saturating_add(tokens as u64);
            let answer = match unpacked.primitive {
                crate::families::wire::QuestionPrimitive::Noul => {
                    let support = calibrated_sigmoid(logits[0], CALIBRATION_TEMPERATURE)?;
                    openkind_core::Answer::Noul(openkind_core::NoulAnswer { noul: support })
                }
                crate::families::wire::QuestionPrimitive::Choice => {
                    let supports = logits
                        .iter()
                        .map(|logit| calibrated_sigmoid(*logit, CALIBRATION_TEMPERATURE))
                        .collect::<Result<Vec<f64>, FamilyError>>()?;
                    let probabilities = renormalize(&supports)?;
                    crate::families::wire::answer_from_probabilities(&unpacked, &probabilities)?
                }
                crate::families::wire::QuestionPrimitive::Score => {
                    let probabilities = temperature_softmax(&logits, CALIBRATION_TEMPERATURE)?;
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
