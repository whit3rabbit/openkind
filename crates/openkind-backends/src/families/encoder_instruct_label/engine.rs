//! `DecisionEngine` adapter for the pinned encoder-instruct-label profile.

use std::sync::Arc;

use candle_core::Device;
use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::device::FamilyExecution;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{EncoderInstructLabelModel, VerifiedArtifacts};
use super::renderer::EncoderInstructLabelRenderer;
use super::{
    EncoderInstructLabelEngineConfig, EncoderInstructLabelError, BACKBONE_ID,
    CALIBRATION_TEMPERATURE, MAX_CANDIDATES, PROFILE_ID,
};

/// Label-marker logit source shared by the candle and ONNX models.
trait MarkerLogits: Send + Sync {
    /// One raw dot-product logit per `<<LABEL>>` marker in token order.
    fn marker_logits(&self, token_ids: &[u32]) -> Result<Vec<f64>, FamilyError>;
}

impl MarkerLogits for EncoderInstructLabelModel {
    fn marker_logits(&self, token_ids: &[u32]) -> Result<Vec<f64>, FamilyError> {
        EncoderInstructLabelModel::marker_logits(self, token_ids)
    }
}

/// ONNX model readout bridging the shared logit source (feature `onnx`).
#[cfg(feature = "onnx")]
impl MarkerLogits for super::onnx::EncoderInstructLabelOnnxModel {
    fn marker_logits(&self, token_ids: &[u32]) -> Result<Vec<f64>, FamilyError> {
        super::onnx::EncoderInstructLabelOnnxModel::marker_logits(self, token_ids)
    }
}

/// Loaded pinned label-marker engine.
pub struct EncoderInstructLabelEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: EncoderInstructLabelRenderer,
    model: Box<dyn MarkerLogits>,
    backend_id: String,
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
    ///
    /// The reference execution is FP32 CPU; see [`Self::load_with_execution`]
    /// for accelerated backends.
    pub fn load(
        config: EncoderInstructLabelEngineConfig,
    ) -> Result<BoundedFamilyEngine, EncoderInstructLabelError> {
        Self::load_with_execution(config, FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// Artifact verification (digests, pinned config) is identical on every
    /// backend; only the readout execution differs, and ONNX additionally
    /// requires `model.onnx` and its digest manifest in the model root.
    /// Loads fail closed when a backend is unavailable.
    pub fn load_with_execution(
        config: EncoderInstructLabelEngineConfig,
        execution: FamilyExecution,
    ) -> Result<BoundedFamilyEngine, EncoderInstructLabelError> {
        let artifacts = VerifiedArtifacts::verify_for_execution(&config.model_root, execution)?;
        let renderer = EncoderInstructLabelRenderer::load(&artifacts.tokenizer)?;
        let model: Box<dyn MarkerLogits> = match execution {
            FamilyExecution::Cpu => {
                Box::new(EncoderInstructLabelModel::load(&artifacts, Device::Cpu)?)
            }
            #[cfg(feature = "cuda")]
            FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device().map_err(FamilyError::from)?;
                Box::new(EncoderInstructLabelModel::load(&artifacts, device)?)
            }
            #[cfg(feature = "onnx")]
            FamilyExecution::Onnx { device_id } => {
                let acceleration = crate::onnx::OnnxAcceleration::from_onnx_execution(device_id)
                    .map_err(FamilyError::from)?;
                Box::new(super::onnx::EncoderInstructLabelOnnxModel::load(
                    &config.model_root,
                    acceleration,
                )?)
            }
            #[cfg(feature = "onnx-rocm")]
            FamilyExecution::OnnxRocm { device_id } => {
                Box::new(super::onnx::EncoderInstructLabelOnnxModel::load(
                    &config.model_root,
                    crate::onnx::OnnxAcceleration::Rocm { device_id },
                )?)
            }
        };
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                backend_id: format!("encoder-instruct-label/{}", execution.id_fragment()),
            }),
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
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} label-marker decision engine ({PROFILE_ID}, {}).",
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
        evaluate_with_logits(
            &|markers: &[String], state: &str| self.inner.candidate_logits(markers, state),
            request,
            control,
        )
    }
}

/// Model-agnostic evaluation shared by every execution backend: the
/// backend-specific `candidate_logits` source is the only input, so both
/// backends apply the identical wire mapping and calibration.
pub(crate) fn evaluate_with_logits<L>(
    logits_source: &L,
    request: SystemRequest,
    control: &FamilyControl,
) -> Result<SystemResponse, FamilyError>
where
    L: Fn(&[String], &str) -> Result<(Vec<f64>, usize), FamilyError>,
{
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
    >::with_capacity_and_hasher(question_ids.len(), Default::default());
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
        let (logits, tokens) = logits_source(&markers, &state)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_sigmoid_rejects_non_finite_logits_and_spans_the_range() {
        assert!(calibrated_sigmoid(f64::NAN, 1.0).is_err());
        assert!(calibrated_sigmoid(f64::INFINITY, 1.0).is_err());
        assert!(calibrated_sigmoid(f64::NEG_INFINITY, 1.0).is_err());

        for logit in [-800.0_f64, -30.0, -1.0, 0.0, 1.0, 30.0, 800.0] {
            let probability = calibrated_sigmoid(logit, 1.0).expect("finite logit");
            assert!(
                (0.0..=1.0).contains(&probability),
                "logit {logit} produced {probability}"
            );
        }
        assert!((calibrated_sigmoid(0.0, 1.0).unwrap() - 0.5).abs() < 1e-12);
        // The two stability arms must agree where they overlap.
        assert!(
            (calibrated_sigmoid(-2.0, 1.0).unwrap()
                - (1.0 - calibrated_sigmoid(2.0, 1.0).unwrap()))
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn renormalize_rejects_unusable_support_mass() {
        assert!(renormalize(&[0.0, 0.0]).is_err());
        assert!(renormalize(&[f64::NAN, 1.0]).is_err());
        assert!(renormalize(&[1.0, f64::INFINITY]).is_err());

        let probabilities = renormalize(&[1.0, 3.0]).expect("usable support");
        assert!((probabilities[0] - 0.25).abs() < 1e-12);
        assert!((probabilities[1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn noul_marker_selects_the_true_criterion_when_explicit() {
        let unpacked = crate::families::wire::UnpackedQuestion {
            id: "q".to_owned(),
            primitive: crate::families::wire::QuestionPrimitive::Noul,
            labels: vec!["false".to_owned(), "true".to_owned()],
            criteria: vec!["not helpful".to_owned(), "helpful".to_owned()],
            ordered: false,
        };
        assert_eq!(noul_marker(&unpacked, "default text", true), "helpful");
        assert_eq!(
            noul_marker(&unpacked, "default text", false),
            "default text"
        );
    }
}
