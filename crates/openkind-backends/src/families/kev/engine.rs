//! `DecisionEngine` adapter for the pinned kev profile.
//!
//! Wire mapping mirrors the reference `kev.api.to_record`: `Noul` scores
//! the two options `no`/`yes` (annotated with the caller's rendered
//! criteria when explicit criteria are supplied) and reports `p(true)`;
//! `Choice` scores one option per offered key in the shared sorted order;
//! `Score` scores the rendered level criteria. The option softmax is
//! temperature-calibrated and mapped back through the shared family wire
//! helpers.

use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{KevModel, QuestionRow, VerifiedArtifacts};
use super::renderer::{option_text, render_json, KevRenderer};
use super::{
    KevEngineConfig, KevError, BACKBONE_ID, BASE_MODEL_ID, CALIBRATION_TEMPERATURE, PROFILE_ID,
};

/// Loaded pinned kev engine.
pub struct KevEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: KevRenderer,
    model: KevModel,
}

/// The offered options of one question, in evaluation order, exactly as the
/// reference `kev.api.to_record` builds them: `Noul` annotates `no`/`yes`
/// with the caller's criteria when explicit criteria are supplied; `Choice`
/// uses `key` or `key: description` (a null description leaves the bare
/// key); `Score` uses the bare level criteria text.
fn question_options(
    unpacked: &crate::families::wire::UnpackedQuestion,
    question: &openkind_core::Question,
    explicit_noul_criteria: bool,
) -> Vec<String> {
    match unpacked.primitive {
        crate::families::wire::QuestionPrimitive::Noul => {
            let criterion = |index: usize| {
                if explicit_noul_criteria {
                    Some(unpacked.criteria[index].as_str())
                } else {
                    None
                }
            };
            vec![
                option_text("no", criterion(0)),
                option_text("yes", criterion(1)),
            ]
        }
        crate::families::wire::QuestionPrimitive::Choice => {
            let criteria = match question {
                openkind_core::Question::Choice(choice) => &choice.criteria,
                _ => unreachable!("primitive and wire shape agree"),
            };
            unpacked
                .labels
                .iter()
                .map(|label| {
                    let description = criteria
                        .get(label)
                        .and_then(|description| description.as_deref())
                        .filter(|text| !text.trim().is_empty());
                    option_text(label, description)
                })
                .collect()
        }
        crate::families::wire::QuestionPrimitive::Score => unpacked.criteria.clone(),
    }
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

impl KevEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    pub fn load(config: KevEngineConfig) -> Result<BoundedFamilyEngine, KevError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root, &config.base_root)?;
        let renderer = KevRenderer::load(&artifacts.tokenizer)?;
        let model = KevModel::load(&artifacts)?;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl Inner {
    /// Pointer logits for one question: the state prefix plus this
    /// question's branch as one causal row.
    fn logits(
        &self,
        state_tokens: &[u32],
        instruction: &str,
        options: &[String],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        control.check()?;
        let QuestionRow { ids, decide, opts } =
            self.renderer
                .encode_row(state_tokens, instruction, options)?;
        let row_len = ids.len();
        let mut logits = self
            .model
            .rows_logits(&ids, &[row_len], &[(decide, opts)])?;
        let logits = logits.pop().ok_or_else(|| {
            FamilyError::Numerical("question produced no pointer logits".to_owned())
        })?;
        if logits.len() != options.len() {
            return Err(FamilyError::ContractMismatch {
                field: "option.count",
                expected: format!("{} option logits", options.len()),
                actual: format!("{} logits", logits.len()),
            });
        }
        Ok(logits)
    }
}

impl FamilyEvaluator for KevEngine {
    fn backend_id(&self) -> &str {
        "kev/cpu-fp32"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} pointer decision engine on {BASE_MODEL_ID} \
                 ({PROFILE_ID}, FP32 CPU)."
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
            State::Object(map) => render_json(&serde_json::Value::Object(map.clone())),
            State::Array(items) => render_json(&serde_json::Value::Array(items.clone())),
        };
        if state.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "state must contain non-whitespace text".to_owned(),
            ));
        }
        let state_tokens = self.inner.renderer.state_tokens(&state)?;
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
            let explicit_noul_criteria = noul_criteria_are_explicit(question);
            let options = question_options(&unpacked, question, explicit_noul_criteria);
            // The instruction renders like any other JSON value (a string
            // instruction stays verbatim).
            let instruction = render_json(question_instructions(question));
            let logits = self
                .inner
                .logits(&state_tokens, &instruction, &options, control)?;
            input_tokens = input_tokens.saturating_add(state_tokens.len() as u64);
            let probabilities = temperature_softmax(&logits, CALIBRATION_TEMPERATURE)?;
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
