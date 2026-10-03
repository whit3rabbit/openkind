//! `DecisionEngine` adapter for the pinned `decider` profiles.

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::decoder_logit_qwen35::QuestionKind;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};
use crate::families::wire;

use super::model::{DeciderModel, VerifiedArtifacts};
use super::renderer::{DeciderRenderer, RenderedRow};
use super::{DeciderEngineConfig, DeciderError, DeciderProfile};

/// Loaded pinned engine for one profile of the family.
pub struct DeciderEngine {
    inner: Arc<Inner>,
}

struct Inner {
    profile: &'static DeciderProfile,
    renderer: DeciderRenderer,
    model: DeciderModel,
    backend_id: String,
}

/// One scoring row of a planned question: the rendered prompt, the number of
/// label logits to read, and the row's answer-type temperature key.
struct PlannedRow {
    row: RenderedRow,
    label_count: usize,
    kind: QuestionKind,
}

/// The rows of one planned question plus its unpacked wire form.
struct PlannedQuestion {
    unpacked: wire::UnpackedQuestion,
    kind: QuestionKind,
    rows: Vec<PlannedRow>,
}

/// Reference fallback instruction for a Noul question that carries none
/// (`NOUL_WITHOUT_INSTRUCTIONS`).
const NOUL_WITHOUT_INSTRUCTIONS: &str = "Which answer fits the context?";

/// Strip a leading level number from a legend description
/// (`strip_level_number`): `"2: somewhat"` -> `"somewhat"`.
fn strip_level_number(text: &str) -> String {
    let trimmed = text.trim_start();
    let without_sign = trimmed.strip_prefix('-').unwrap_or(trimmed);
    let digits: String = without_sign
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        return text.to_owned();
    }
    let rest = &without_sign[digits.len()..];
    let rest = rest.trim_start();
    match rest.strip_prefix(':') {
        Some(after) => after.trim_start().to_owned(),
        None => text.to_owned(),
    }
}

impl Inner {
    /// Plan the scoring rows of one wire question (reference
    /// `render_question` + `plan_rows` with `isolated_levels`).
    fn plan_question(
        &self,
        id: &str,
        question: &openkind_core::Question,
        state_ids: &[u32],
    ) -> Result<PlannedQuestion, FamilyError> {
        let profile = self.profile;
        match question {
            openkind_core::Question::Choice(choice) => {
                let count = choice.criteria.len();
                if !(2..=profile.max_candidates).contains(&count) {
                    return Err(FamilyError::InvalidInput(format!(
                        "choice question `{id}` offers {count} options; the profile serves 2..={}",
                        profile.max_candidates
                    )));
                }
                // The wire map is a hash map, so options render in the wire's
                // sorted label order — the family's declared determinism
                // difference. A null description renders the bare label, a
                // described one renders `label: description`.
                let mut labels: Vec<&String> = choice.criteria.keys().collect();
                labels.sort();
                let options = labels
                    .iter()
                    .map(|label| match choice.criteria.get(*label) {
                        Some(Some(description))
                            if !description.trim().is_empty() && description != *label =>
                        {
                            format!("{label}: {description}")
                        }
                        _ => (*label).clone(),
                    })
                    .collect::<Vec<_>>();
                let row = self.renderer.render_row(
                    state_ids,
                    &wire::instruction_text(question)?,
                    &options,
                )?;
                Ok(PlannedQuestion {
                    unpacked: wire::unpack_question(id, question)?,
                    kind: QuestionKind::Choice,
                    rows: vec![PlannedRow {
                        row,
                        label_count: count,
                        kind: QuestionKind::Choice,
                    }],
                })
            }
            openkind_core::Question::Noul(noul) => {
                let mut instruction = wire::instruction_text(question)?;
                if instruction.trim().is_empty() {
                    instruction = NOUL_WITHOUT_INSTRUCTIONS.to_owned();
                }
                // The reference renders the fixed pair ["no", "yes"] in the
                // wire's false-first order; described criteria prefix them.
                let (no_text, yes_text) = noul
                    .criteria
                    .as_ref()
                    .map(|criteria| (criteria.r#false.clone(), criteria.r#true.clone()))
                    .unwrap_or_default();
                let options = vec![
                    described_option("no", &no_text),
                    described_option("yes", &yes_text),
                ];
                let row = self
                    .renderer
                    .render_row(state_ids, &instruction, &options)?;
                Ok(PlannedQuestion {
                    unpacked: wire::UnpackedQuestion {
                        id: id.to_owned(),
                        primitive: wire::QuestionPrimitive::Noul,
                        labels: vec!["false".into(), "true".into()],
                        criteria: vec![
                            if no_text.is_empty() {
                                "The proposition is false.".to_owned()
                            } else {
                                no_text
                            },
                            if yes_text.is_empty() {
                                "The proposition is true.".to_owned()
                            } else {
                                yes_text
                            },
                        ],
                        ordered: false,
                    },
                    kind: QuestionKind::Noul,
                    rows: vec![PlannedRow {
                        row,
                        label_count: 2,
                        kind: QuestionKind::Noul,
                    }],
                })
            }
            openkind_core::Question::Score(score) => {
                let levels = score.criteria.len();
                if !(2..=profile.max_score_levels).contains(&levels) {
                    return Err(FamilyError::InvalidInput(format!(
                        "score question `{id}` offers {levels} levels; the profile serves 2..={}",
                        profile.max_score_levels
                    )));
                }
                let instruction = wire::instruction_text(question)?;
                // Isolated levels (the pinned `isolated_levels: true`): one
                // yes/no row per level, judged without its neighbours.
                let mut rows = Vec::with_capacity(levels);
                for level in &score.criteria {
                    let level_text = strip_level_number(level);
                    let row_question = format!(
                        "{instruction}\nProposed answer: {level_text}\nDoes the proposed answer fit?"
                    );
                    let row = self.renderer.render_row(
                        state_ids,
                        &row_question,
                        &["no".to_owned(), "yes".to_owned()],
                    )?;
                    rows.push(PlannedRow {
                        row,
                        label_count: 2,
                        kind: QuestionKind::Score,
                    });
                }
                let unpacked = wire::unpack_question(id, question)?;
                Ok(PlannedQuestion {
                    unpacked,
                    kind: QuestionKind::Score,
                    rows,
                })
            }
        }
    }

    /// Score one planned question: forward every row, temperature-softmax
    /// each read, and assemble the wire answer.
    fn score_question(
        &self,
        planned: PlannedQuestion,
        control: &FamilyControl,
        token_count: &mut u64,
    ) -> Result<openkind_core::Answer, FamilyError> {
        let temperature_for = |kind| match kind {
            QuestionKind::Choice => self.profile.calibration.resolve(kind),
            QuestionKind::Noul => self.profile.calibration.resolve(kind),
            QuestionKind::Score => self.profile.calibration.resolve(kind),
        };
        let mut row_probabilities = Vec::with_capacity(planned.rows.len());
        for planned_row in &planned.rows {
            control.check()?;
            *token_count = token_count
                .saturating_add(u64::from(self.renderer.row_token_count(&planned_row.row)));
            let label_ids = self
                .renderer
                .label_ids_for(planned_row.label_count)
                .to_vec();
            let logits =
                self.model
                    .slot_logits(planned_row.row.prompt_ids(), &label_ids, control)?;
            row_probabilities.push(temperature_softmax(
                &logits,
                temperature_for(planned_row.kind),
            )?);
        }
        let probabilities = match planned.kind {
            // Choice and Noul read one row whose distribution is the answer.
            QuestionKind::Choice | QuestionKind::Noul => row_probabilities
                .into_iter()
                .next()
                .ok_or_else(|| FamilyError::InvalidInput("question planned no rows".to_owned()))?,
            // Score combines the per-level fit probabilities: each row
            // contributes P(fits) and the level distribution normalizes
            // them (`combine_isolated`).
            QuestionKind::Score => {
                let mut fits = Vec::with_capacity(row_probabilities.len());
                for probabilities in &row_probabilities {
                    let fit = probabilities.last().copied().ok_or_else(|| {
                        FamilyError::Numerical("level row lost its yes mass".to_owned())
                    })?;
                    fits.push(fit);
                }
                let total: f64 = fits.iter().sum();
                if !total.is_finite() || total <= 0.0 {
                    return Err(FamilyError::Numerical(format!(
                        "isolated level fits do not normalize: {total}"
                    )));
                }
                fits.iter().map(|fit| fit / total).collect()
            }
        };
        wire::answer_from_probabilities(&planned.unpacked, &probabilities)
    }
}

/// Render one reference option: bare word when undescribed, `word: text`
/// otherwise.
fn described_option(word: &str, text: &str) -> String {
    if text.trim().is_empty() {
        word.to_owned()
    } else {
        format!("{word}: {text}")
    }
}

impl DeciderEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The returned adapter implements [`DecisionEngine`] and carries
    /// admission, queue, deadline, and cancellation control.
    /// Load every pinned artifact offline and build the bounded engine on
    /// the CPU reference backend.
    pub fn load(config: DeciderEngineConfig) -> Result<BoundedFamilyEngine, DeciderError> {
        Self::load_with_execution(config, crate::device::FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// The slot-logit readout runs on the Qwen3.5 hybrid backbone, which has
    /// no ONNX export; accelerated loads require the `cuda` feature.
    pub fn load_with_execution(
        config: DeciderEngineConfig,
        execution: crate::device::FamilyExecution,
    ) -> Result<BoundedFamilyEngine, DeciderError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root, config.profile)?;
        let renderer = DeciderRenderer::load(&artifacts.tokenizer, config.profile)?;
        let backend_id: String = match execution {
            crate::device::FamilyExecution::Cpu => config.profile.cpu_backend_id.to_owned(),
            #[cfg(feature = "cuda")]
            crate::device::FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device()?;
                let model = DeciderModel::load_with_device(&artifacts, device)?;
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
                    "the decider slot-logit readout runs on the Qwen3.5 hybrid backbone,                      which has no ONNX export; select cpu or (with the `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
            #[cfg(feature = "onnx-rocm")]
            crate::device::FamilyExecution::OnnxRocm { .. } => {
                return Err(crate::families::support::FamilyError::ExecutionUnavailable(
                    "the decider slot-logit readout runs on the Qwen3.5 hybrid backbone, which \
                     has no ONNX export and therefore no ROCm execution; select cpu or (with \
                     the `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
        };
        let model = DeciderModel::load(&artifacts)?;
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

impl FamilyEvaluator for DeciderEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} slot-logit decision engine ({}, {}).",
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
        let state_ids = self.inner.renderer.state_ids(&state)?;
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
                .plan_question(id, question, &state_ids)
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
    use super::*;

    #[test]
    fn strip_level_number_removes_only_a_leading_index_prefix() {
        assert_eq!(strip_level_number("2: somewhat"), "somewhat");
        assert_eq!(strip_level_number("10: high"), "high");
        assert_eq!(strip_level_number("-1: below"), "below");
        assert_eq!(strip_level_number("severe"), "severe");
        // No colon directly after the digits: the regex does not match and
        // the text stays unchanged, as in the reference.
        assert_eq!(
            strip_level_number("2 colon later: text"),
            "2 colon later: text"
        );
    }
}
