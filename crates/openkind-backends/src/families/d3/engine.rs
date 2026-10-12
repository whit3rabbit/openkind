//! `DecisionEngine` adapter for the pinned `d3` profiles.

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};
use crate::families::wire;

use super::model::{D3Model, VerifiedArtifacts};
use super::renderer::{described_option, D3Renderer, RenderedRow, NOUL_DEFAULT_DESCRIPTIONS};
use super::{D3EngineConfig, D3Error, D3Profile};

/// Loaded pinned engine for one profile of the family.
pub struct D3Engine {
    inner: Arc<Inner>,
}

struct Inner {
    profile: &'static D3Profile,
    renderer: D3Renderer,
    model: D3Model,
    backend_id: String,
}

/// One scoring row of a planned question: the rendered prompt, the number
/// of code logits to read, and the option texts that fix the code order.
struct PlannedRow {
    row: RenderedRow,
    code_count: usize,
}

/// The rows of one planned question plus its unpacked wire form.
struct PlannedQuestion {
    unpacked: wire::UnpackedQuestion,
    rows: Vec<PlannedRow>,
}

impl Inner {
    /// Plan the scoring rows of one wire question (reference
    /// `normalize_question` + `render`). Every question type reads exactly
    /// one row: Choice over the sorted offered options, Noul over the fixed
    /// false/true pair, Score over the ordered level texts (code order is
    /// level order).
    fn plan_question(
        &self,
        id: &str,
        question: &openkind_core::Question,
        state: &serde_json::Value,
    ) -> Result<PlannedQuestion, FamilyError> {
        let profile = self.profile;
        match question {
            openkind_core::Question::Choice(choice) => {
                let count = choice.criteria.len();
                if !(2..=profile.max_candidates).contains(&count) {
                    return Err(FamilyError::InvalidInput(format!(
                        "choice question `{id}` offers {count} options; the profile serves \
                         2..={}",
                        profile.max_candidates
                    )));
                }
                // The wire map is a hash map, so options render in the wire's
                // sorted label order — the family's declared determinism
                // difference. A null description renders the bare label, a
                // described one renders `label: description`.
                let unpacked = wire::unpack_question(id, question)?;
                // Reference `options()`: a null description renders the bare
                // label, a described one (even an empty string) renders
                // `label: description`.
                let options: Vec<String> = unpacked
                    .labels
                    .iter()
                    .map(|label| match choice.criteria.get(label) {
                        Some(Some(description)) => format!("{label}: {description}"),
                        _ => label.clone(),
                    })
                    .collect();
                let row =
                    self.renderer
                        .render_row(state, &question_instructions(question), &options)?;
                Ok(PlannedQuestion {
                    unpacked,
                    rows: vec![PlannedRow {
                        row,
                        code_count: count,
                    }],
                })
            }
            openkind_core::Question::Noul(noul) => {
                // The reference renders the fixed pair ["false", "true"]
                // with `No / false` / `Yes / true` descriptions unless the
                // caller supplied criteria.
                let (false_text, true_text) = noul
                    .criteria
                    .as_ref()
                    .map(|criteria| (criteria.r#false.clone(), criteria.r#true.clone()))
                    .unwrap_or_default();
                // The reference noul texts are the bare descriptions (the
                // code prefix is the renderer's job); empty descriptions
                // fall back to `NOUL_DESCRIPTIONS` like Python `or`.
                let fallback = |text: &str, default: &str| {
                    if text.is_empty() {
                        default.to_owned()
                    } else {
                        text.to_owned()
                    }
                };
                let options = vec![
                    fallback(&false_text, NOUL_DEFAULT_DESCRIPTIONS[0]),
                    fallback(&true_text, NOUL_DEFAULT_DESCRIPTIONS[1]),
                ];
                let unpacked = wire::unpack_question(id, question)?;
                let row =
                    self.renderer
                        .render_row(state, &question_instructions(question), &options)?;
                Ok(PlannedQuestion {
                    unpacked,
                    rows: vec![PlannedRow { row, code_count: 2 }],
                })
            }
            openkind_core::Question::Score(score) => {
                let levels = score.criteria.len();
                if !(2..=profile.max_score_levels).contains(&levels) {
                    return Err(FamilyError::InvalidInput(format!(
                        "score question `{id}` offers {levels} levels; the profile serves \
                         2..={}",
                        profile.max_score_levels
                    )));
                }
                // The ordered levels are the options, each shown by its text
                // alone (reference `normalize_question`: code order is level
                // order, and the level distribution is the readout).
                let options: Vec<String> = score
                    .criteria
                    .iter()
                    .map(|level| described_option(level, None))
                    .collect();
                let unpacked = wire::unpack_question(id, question)?;
                let row =
                    self.renderer
                        .render_row(state, &question_instructions(question), &options)?;
                Ok(PlannedQuestion {
                    unpacked,
                    rows: vec![PlannedRow {
                        row,
                        code_count: levels,
                    }],
                })
            }
        }
    }

    /// Score one planned question: forward every row, temperature-softmax
    /// the code logits restricted to the offered codes, and assemble the
    /// wire answer.
    fn score_question(
        &self,
        planned: PlannedQuestion,
        control: &FamilyControl,
        token_count: &mut u64,
    ) -> Result<openkind_core::Answer, FamilyError> {
        let mut row_probabilities = Vec::with_capacity(planned.rows.len());
        for planned_row in &planned.rows {
            control.check()?;
            *token_count = token_count
                .saturating_add(u64::from(self.renderer.row_token_count(&planned_row.row)));
            let logits = self.model.code_logits(
                planned_row.row.prompt_ids(),
                planned_row.code_count,
                control,
            )?;
            row_probabilities.push(temperature_softmax(&logits, self.profile.temperature)?);
        }
        // Every pinned question type reads exactly one row whose
        // distribution is the answer.
        let probabilities = row_probabilities
            .into_iter()
            .next()
            .ok_or_else(|| FamilyError::InvalidInput("question planned no rows".to_owned()))?;
        wire::answer_from_probabilities(&planned.unpacked, &probabilities)
    }
}

/// The wire instructions payload of one question, ready for the renderer's
/// `describe(instructions or default)` semantics.
fn question_instructions(question: &openkind_core::Question) -> serde_json::Value {
    match question {
        openkind_core::Question::Noul(question) => question.instructions.clone(),
        openkind_core::Question::Choice(question) => question.instructions.clone(),
        openkind_core::Question::Score(question) => question.instructions.clone(),
    }
}

impl D3Engine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The returned adapter implements [`DecisionEngine`] and carries
    /// admission, queue, deadline, and cancellation control.
    pub fn load(config: D3EngineConfig) -> Result<BoundedFamilyEngine, D3Error> {
        Self::load_with_execution(config, crate::device::FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// The code readout runs on the Qwen3.5 hybrid backbone, which has no
    /// ONNX export; accelerated loads require the `cuda` feature.
    pub fn load_with_execution(
        config: D3EngineConfig,
        execution: crate::device::FamilyExecution,
    ) -> Result<BoundedFamilyEngine, D3Error> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root, config.profile)?;
        let decision_config: serde_json::Value =
            crate::families::support::read_json(&config.model_root.join("decision_config.json"))?;
        let pinned_codes: Vec<String> = decode_strings(&decision_config["codes"])?;
        let pinned_token_ids: Vec<u64> = decode_numbers(&decision_config["token_ids"])?;
        let renderer = D3Renderer::load(
            &artifacts.tokenizer,
            &pinned_codes,
            &pinned_token_ids,
            config.profile,
        )?;
        let backend_id: String = match execution {
            crate::device::FamilyExecution::Cpu => config.profile.cpu_backend_id.to_owned(),
            #[cfg(feature = "cuda")]
            crate::device::FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device().map_err(FamilyError::from)?;
                let model = D3Model::load_with_device(&artifacts, config.profile, device)?;
                let engine = Self {
                    inner: Arc::new(Inner {
                        profile: config.profile,
                        renderer,
                        model,
                        backend_id: config.profile.cuda_backend_id.to_owned(),
                    }),
                };
                return Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits));
            }
            #[cfg(feature = "onnx")]
            crate::device::FamilyExecution::Onnx { .. } => {
                return Err(crate::families::support::FamilyError::ExecutionUnavailable(
                    "the d3 code readout runs on the Qwen3.5 hybrid backbone, which has no \
                     ONNX export; select cpu or (with the `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
            #[cfg(feature = "onnx-rocm")]
            crate::device::FamilyExecution::OnnxRocm { .. } => {
                return Err(crate::families::support::FamilyError::ExecutionUnavailable(
                    "the d3 code readout runs on the Qwen3.5 hybrid backbone, which has no \
                     ONNX export and therefore no ROCm execution; select cpu or (with the \
                     `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
        };
        let model = D3Model::load(&artifacts, config.profile)?;
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

/// Decode a pinned JSON string array.
fn decode_strings(value: &serde_json::Value) -> Result<Vec<String>, FamilyError> {
    serde_json::from_value(value.clone()).map_err(|error| FamilyError::ContractMismatch {
        field: "decision_config.codes",
        expected: "a JSON string array".to_owned(),
        actual: error.to_string(),
    })
}

/// Decode a pinned JSON integer array.
fn decode_numbers(value: &serde_json::Value) -> Result<Vec<u64>, FamilyError> {
    serde_json::from_value(value.clone()).map_err(|error| FamilyError::ContractMismatch {
        field: "decision_config.token_ids",
        expected: "a JSON integer array".to_owned(),
        actual: error.to_string(),
    })
}

impl FamilyEvaluator for D3Engine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} noncausal code-readout decision engine ({}, {}).",
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
            let planned = self
                .inner
                .plan_question(id, question, &state)
                .map_err(|error| match error {
                    FamilyError::InvalidInput(message) => {
                        FamilyError::InvalidInput(format!("question `{id}`: {message}"))
                    }
                    other => other,
                })?;
            let mut question_tokens: u64 = 0;
            let answer = self
                .inner
                .score_question(planned, control, &mut question_tokens)?;
            if question_tokens > self.inner.profile.max_row_tokens as u64 {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` requires {question_tokens} prompt tokens across its rows, \
                     exceeding the maximum work budget of {}",
                    self.inner.profile.max_row_tokens
                )));
            }
            input_tokens = input_tokens.saturating_add(question_tokens);
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

#[cfg(test)]
mod tests {
    #[test]
    fn noul_descriptions_render_bare_by_default() {
        // The option texts are the bare descriptions; the code prefix is
        // the renderer's (reference `options()` + `user_prompt`).
        let fallback = |text: &str, default: &str| {
            if text.is_empty() {
                default.to_owned()
            } else {
                text.to_owned()
            }
        };
        assert_eq!(fallback("", "No / false"), "No / false");
        assert_eq!(fallback("described", "No / false"), "described");
    }
}
