//! `DecisionEngine` adapter for the pinned von profile.

use std::sync::Arc;

use candle_core::Device;
use openkind_core::{ModelInfo, Question, State, SystemRequest, SystemResponse, Usage};

use crate::device::FamilyExecution;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{VerifiedArtifacts, VonModel};
use super::renderer::VonRenderer;
use super::{VonEngineConfig, VonError, VonProfile, MAX_CANDIDATES};

/// Option-marker logit source shared by the candle and ONNX models.
trait VonOptionLogits: Send + Sync {
    /// One logit per option marker in marker order, before calibration.
    fn option_logits(&self, token_ids: &[u32], markers: &[usize]) -> Result<Vec<f64>, FamilyError>;
}

impl VonOptionLogits for VonModel {
    fn option_logits(&self, token_ids: &[u32], markers: &[usize]) -> Result<Vec<f64>, FamilyError> {
        VonModel::option_logits(self, token_ids, markers)
    }
}

/// ONNX model readout bridging the shared logit source (feature `onnx`).
#[cfg(feature = "onnx")]
impl VonOptionLogits for super::onnx::VonOnnxModel {
    fn option_logits(&self, token_ids: &[u32], markers: &[usize]) -> Result<Vec<f64>, FamilyError> {
        super::onnx::VonOnnxModel::option_logits(self, token_ids, markers)
    }
}

/// Loaded pinned von engine.
pub struct VonEngine {
    inner: Arc<Inner>,
}

struct Inner {
    profile: &'static VonProfile,
    renderer: VonRenderer,
    model: Box<dyn VonOptionLogits>,
    backend_id: String,
}

/// Serialize the wire state the way the reference formats its input with
/// Python `str()`: string states pass through; object states render as
/// `f"{k}: {v}"` lines (values `str()`-ed, strings unquoted); array states
/// render as Python `str(list)`.
pub(crate) fn von_state_text(state: &State) -> Result<String, FamilyError> {
    match state {
        State::Text(text) => Ok(text.clone()),
        State::Object(map) => {
            let value = serde_json::to_value(map).map_err(|error| {
                FamilyError::InvalidInput(format!("state serialization failed: {error}"))
            })?;
            let mut out = String::new();
            if let serde_json::Value::Object(entries) = value {
                for (key, item) in &entries {
                    out.push_str(key);
                    out.push_str(": ");
                    python_str(item, &mut out);
                    out.push('\n');
                }
            }
            // Trim the trailing newline the reference's join leaves off.
            while out.ends_with('\n') {
                out.pop();
            }
            Ok(out)
        }
        State::Array(items) => {
            let value = serde_json::to_value(items).map_err(|error| {
                FamilyError::InvalidInput(format!("state serialization failed: {error}"))
            })?;
            let mut out = String::new();
            python_repr(&value, &mut out);
            Ok(out)
        }
    }
}

/// Python `str()` of a scalar at f-string top level: strings stay raw,
/// containers fall back to their repr.
fn python_str(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::String(text) => out.push_str(text),
        other => python_repr(other, out),
    }
}

/// Python `repr()` of a JSON value: `None`/`True`/`False` keywords, single-
/// quoted strings, and bracketed containers.
fn python_repr(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(number) => {
            use std::fmt::Write;
            if let Some(int) = number
                .as_i64()
                .or_else(|| number.as_u64().map(|u| u as i64))
            {
                let _ = write!(out, "{int}");
            } else {
                let float = number.as_f64().unwrap_or(f64::NAN);
                // Python prints integral floats with a trailing ".0" and
                // otherwise uses the shortest round-trip repr.
                if float.fract() == 0.0 && float.abs() < 1e16 {
                    let _ = write!(out, "{float:.1}");
                } else {
                    let _ = write!(out, "{float}");
                }
            }
        }
        Value::String(text) => python_quote(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_repr(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_quote(key, out);
                out.push_str(": ");
                python_repr(item, out);
            }
            out.push('}');
        }
    }
}

/// Python repr of a string: single quotes unless the text holds a single
/// quote and no double quote; backslashes, newlines, and the active quote
/// escape with a backslash; other control characters escape as `\uXXXX`-style
/// only when non-printable.
fn python_quote(text: &str, out: &mut String) {
    let use_double = text.contains('\'') && !text.contains('"');
    let (open, close) = if use_double { ('"', '"') } else { ('\'', '\'') };
    out.push(open);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == open => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(close);
}

/// One question's rendering inputs: the option descriptions in candidate
/// order and the raw criteria presence flag for the noul debias decision.
struct RenderedQuestionInputs {
    /// Rendered option texts in candidate order (true first for noul).
    pub options: Vec<String>,
    /// Noul only: whether the caller supplied any explicit criterion.
    pub has_explicit_criteria: bool,
}

/// Render von option descriptions from the raw wire question, mirroring the
/// reference `evaluate_*` description rules.
fn render_question_inputs(question: &Question) -> Result<RenderedQuestionInputs, FamilyError> {
    match question {
        Question::Choice(_) => {
            let unpacked = crate::families::wire::unpack_question("choice", question)?;
            // The unpacked criteria are the caller descriptions, falling back
            // to the bare label — exactly the reference's option text rule.
            Ok(RenderedQuestionInputs {
                options: unpacked.criteria,
                has_explicit_criteria: true,
            })
        }
        Question::Score(score) => Ok(RenderedQuestionInputs {
            options: score
                .criteria
                .iter()
                .map(|level| level.trim().to_owned())
                .collect(),
            has_explicit_criteria: true,
        }),
        Question::Noul(noul) => {
            const DEFAULT_TRUE: &str = "Yes, condition holds true.";
            const DEFAULT_FALSE: &str = "No, condition is false.";
            let (true_criterion, false_criterion) = noul
                .criteria
                .as_ref()
                .map(|criteria| (criteria.r#true.as_str(), criteria.r#false.as_str()))
                .unwrap_or(("", ""));
            let has_explicit = !true_criterion.is_empty() || !false_criterion.is_empty();
            let true_text = if true_criterion.is_empty() {
                DEFAULT_TRUE
            } else {
                true_criterion
            };
            let false_text = if false_criterion.is_empty() {
                DEFAULT_FALSE
            } else {
                false_criterion
            };
            Ok(RenderedQuestionInputs {
                // The reference packs [positive, negative] — true first.
                options: vec![true_text.to_owned(), false_text.to_owned()],
                has_explicit_criteria: has_explicit,
            })
        }
    }
}

impl VonEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The reference execution is FP32 CPU; see [`Self::load_with_execution`]
    /// for accelerated backends.
    pub fn load(config: VonEngineConfig) -> Result<BoundedFamilyEngine, VonError> {
        Self::load_with_execution(config, FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// Artifact verification (digests, pinned configs) is identical on every
    /// backend; only the readout execution differs, and ONNX additionally
    /// requires `model.onnx` and its digest manifest in the model root.
    /// Loads fail closed when a backend is unavailable.
    pub fn load_with_execution(
        config: VonEngineConfig,
        execution: FamilyExecution,
    ) -> Result<BoundedFamilyEngine, VonError> {
        let profile = config.profile;
        let artifacts = VerifiedArtifacts::verify(&config.model_root, profile)?;
        let renderer = VonRenderer::load(&artifacts.tokenizer, &profile.specials)?;
        let model: Box<dyn VonOptionLogits> = match execution {
            FamilyExecution::Cpu => Box::new(VonModel::load(profile, &artifacts, Device::Cpu)?),
            #[cfg(feature = "cuda")]
            FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device()?;
                Box::new(VonModel::load(profile, &artifacts, device)?)
            }
            #[cfg(feature = "onnx")]
            FamilyExecution::Onnx { device_id } => {
                let acceleration = crate::onnx::OnnxAcceleration::from_onnx_execution(device_id)
                    .map_err(FamilyError::from)?;
                Box::new(super::onnx::VonOnnxModel::load(
                    &config.model_root,
                    acceleration,
                )?)
            }
            #[cfg(feature = "onnx-rocm")]
            FamilyExecution::OnnxRocm { device_id } => Box::new(super::onnx::VonOnnxModel::load(
                &config.model_root,
                crate::onnx::OnnxAcceleration::Rocm { device_id },
            )?),
        };
        let engine = Self {
            inner: Arc::new(Inner {
                profile,
                renderer,
                model,
                backend_id: format!("{}/{}", profile.loader_id, execution.id_fragment()),
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for VonEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        let profile = self.inner.profile;
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} option-marker decision encoder ({}, {}).",
                profile.backbone_id, profile.profile_id, self.inner.backend_id
            ),
            release_date: "2026-09-30".into(),
        }
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        control.check()?;
        let profile = self.inner.profile;
        let state_serialized = von_state_text(&request.state)?;
        if state_serialized.trim().is_empty() {
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
            if unpacked.labels.len() > MAX_CANDIDATES {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` offers {} options but the profile accepts at most {MAX_CANDIDATES}",
                    unpacked.labels.len()
                )));
            }
            let inputs = render_question_inputs(question)?;
            if inputs.options.len() != unpacked.labels.len() {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` rendered {} option texts for {} candidates",
                    inputs.options.len(),
                    unpacked.labels.len()
                )));
            }
            let instructions = crate::families::wire::instruction_text(question)?;
            let fitted = self.inner.renderer.fit_state(
                profile,
                &state_serialized,
                &instructions,
                &inputs.options,
            )?;
            let packed = self
                .inner
                .renderer
                .pack(&fitted, &instructions, &inputs.options);
            let rendered = self.inner.renderer.encode_packed(&packed)?;
            if rendered.markers.len() != inputs.options.len() {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` packed {} option markers for {} options; packing is corrupt",
                    rendered.markers.len(),
                    inputs.options.len()
                )));
            }
            input_tokens = input_tokens.saturating_add(rendered.ids.len() as u64);
            let mut logits = self
                .inner
                .model
                .option_logits(&rendered.ids, &rendered.markers)?;

            // Noul zero-shot debias: without explicit criteria the reference
            // runs a second pass over an empty state and cancels the
            // intrinsic polarity prior from the true logit.
            let is_noul = matches!(question, Question::Noul(_));
            if is_noul && !inputs.has_explicit_criteria {
                let null_packed = self.inner.renderer.pack("", &instructions, &inputs.options);
                let null_rendered = self.inner.renderer.encode_packed(&null_packed)?;
                input_tokens = input_tokens.saturating_add(null_rendered.ids.len() as u64);
                let null_logits = self
                    .inner
                    .model
                    .option_logits(&null_rendered.ids, &null_rendered.markers)?;
                let (a, b) = profile.noul_prior;
                let bias = null_logits[0] - null_logits[1];
                logits[0] -= a * bias + b;
            }

            let temperature = effective_temperature(
                &profile.calibration_map,
                &logits,
                &self.inner.renderer.encode_plain(&fitted)?,
                inputs.options.len(),
            );
            let probabilities = temperature_softmax(&logits, temperature)?;
            // The unpacked wire order is [false, true] for noul while the
            // reference packs true first; flip so the shared mapping reads
            // the right slot.
            let wire_probabilities: Vec<f64> = if is_noul {
                vec![probabilities[1], probabilities[0]]
            } else {
                probabilities
            };
            let answer =
                crate::families::wire::answer_from_probabilities(&unpacked, &wire_probabilities)?;
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

/// The reference's input-conditioned temperature: a bounded linear function
/// of the option distribution's normalized entropy, `log10(state_tokens)/4`,
/// and the option count over 8. Monotonic in the logits, so it never moves
/// an answer. Port of `_effective_temperature` with the shipped map.
pub(crate) fn effective_temperature(
    map: &super::CalibrationMap,
    logits: &[f64],
    state_tokens: &[u32],
    n_options: usize,
) -> f64 {
    let n = logits.len().max(1);
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let probs: Vec<f64> = logits
        .iter()
        .map(|l| (l - max_logit).exp() / sum_exp)
        .collect();
    let entropy = -probs
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| p * p.ln())
        .sum::<f64>()
        / (n as f64).ln();
    let tokens = (state_tokens.len() as f64).max(1.0);
    let raw = map.bias
        + map.entropy * entropy
        + map.log_tokens * (tokens.log10() / 4.0)
        + map.n_options * (n_options as f64 / 8.0);
    raw.clamp(map.lo, map.hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_map_matches_the_reference_formula() {
        let map = super::super::VON.calibration_map;
        // Sharp two-option logits over a one-token state: the raw map goes
        // negative (the option-count term dominates) and clamps at the floor.
        let sharp = effective_temperature(&map, &[8.0, -8.0], &[7], 2);
        assert_eq!(sharp, map.lo);
        // Uniform logits over many options on a long state: exact reference
        // value — full normalized entropy, the length term, and the option
        // count term all applied before the clamp.
        let flat: Vec<f64> = vec![0.0; 16];
        let long_state: Vec<u32> = Vec::from(&[7u32; 4096][..]);
        let flat_temperature = effective_temperature(&map, &flat, &long_state, 16);
        let flat_expected = map.bias
            + map.entropy * 1.0
            + map.log_tokens * ((4096f64).log10() / 4.0)
            + map.n_options * (16.0 / 8.0);
        assert!(
            (flat_temperature - flat_expected).abs() < 1e-9,
            "hard flat questions apply every map term: {flat_temperature} vs {flat_expected}"
        );
        assert!(
            flat_temperature > 1.0,
            "but stay far below the clamp: {flat_temperature}"
        );
        // A pathologically long state alone saturates the clamp ceiling.
        let huge_state: Vec<u32> = Vec::from(&[7u32; 8_192_000][..]);
        let saturated = effective_temperature(&map, &[8.0, -8.0], &huge_state, 2);
        assert_eq!(saturated, map.hi);
        // The map never leaves its bounds.
        for temperature in [sharp, flat_temperature, saturated] {
            assert!((map.lo..=map.hi).contains(&temperature));
        }
    }

    #[test]
    fn object_state_renders_python_str_lines() {
        let state = State::Object(
            [
                ("asset".to_owned(), serde_json::json!("Windows")),
                ("count".to_owned(), serde_json::json!(3)),
                ("flag".to_owned(), serde_json::json!(true)),
            ]
            .into_iter()
            .collect(),
        );
        let text = von_state_text(&state).expect("serialize");
        assert_eq!(text, "asset: Windows\ncount: 3\nflag: True");
    }

    #[test]
    fn array_state_renders_python_list_repr() {
        let state = State::Array(vec![
            serde_json::json!("first turn"),
            serde_json::json!({"who": "user"}),
        ]);
        let text = von_state_text(&state).expect("serialize");
        assert_eq!(text, "['first turn', {'who': 'user'}]");
    }

    #[test]
    fn python_quote_prefers_single_quotes_and_escapes() {
        let mut out = String::new();
        python_quote("plain", &mut out);
        assert_eq!(out, "'plain'");
        out.clear();
        python_quote("it's", &mut out);
        assert_eq!(out, "\"it's\"");
        out.clear();
        python_quote("line\nbreak", &mut out);
        assert_eq!(out, "'line\\nbreak'");
    }
}
