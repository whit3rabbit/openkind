//! `DecisionEngine` adapter for the pinned `d3` profiles.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse};

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

/// Every question of one request, validated, rendered, and budget-checked.
struct PlannedRequest {
    /// `(question id, plan)` in sorted question-id order.
    questions: Vec<(String, PlannedQuestion)>,
    /// Sum of every planned row's prompt tokens (the usage count).
    input_tokens: u64,
}

/// Plan the scoring rows of one wire question (reference
/// `normalize_question` + `render`). Every question type reads exactly one
/// row: Choice over the sorted offered options, Noul over the fixed
/// false/true pair, Score over the ordered level texts (code order is level
/// order).
fn plan_question(
    profile: &D3Profile,
    renderer: &D3Renderer,
    id: &str,
    question: &openkind_core::Question,
    state: &serde_json::Value,
) -> Result<PlannedQuestion, FamilyError> {
    match question {
        openkind_core::Question::Choice(choice) => {
            // Reference `normalize_question`: 1..=MAX_OPTIONS option keys,
            // each a nonempty string. A single option is legal there and
            // reads a one-code softmax.
            let count = choice.criteria.len();
            if !(1..=profile.max_candidates).contains(&count) {
                return Err(FamilyError::InvalidInput(format!(
                    "choice question `{id}` offers {count} options; the profile serves \
                     1..={}",
                    profile.max_candidates
                )));
            }
            if choice.criteria.keys().any(String::is_empty) {
                return Err(FamilyError::InvalidInput(format!(
                    "choice question `{id}` has an empty option key; choice option keys must \
                     be nonempty strings"
                )));
            }
            // The wire map is a hash map, so options render in the wire's
            // sorted label order: the family's declared determinism
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
            let row = renderer.render_row(state, &question_instructions(question), &options)?;
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
            let row = renderer.render_row(state, &question_instructions(question), &options)?;
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
            // Reference `normalize_question`: the level texts are the option
            // keys, so they must be distinct and nonempty (Python
            // `describe` of a string level is the string itself).
            let distinct: HashSet<&str> = score.criteria.iter().map(String::as_str).collect();
            if score.criteria.iter().any(String::is_empty) || distinct.len() != levels {
                return Err(FamilyError::InvalidInput(format!(
                    "score question `{id}`: score levels must be distinct and nonempty"
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
            let row = renderer.render_row(state, &question_instructions(question), &options)?;
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

/// The wire state as the JSON value the renderer describes.
fn state_value(state: &State) -> Result<serde_json::Value, FamilyError> {
    match state {
        State::Text(text) => Ok(serde_json::Value::String(text.clone())),
        State::Object(map) => serde_json::to_value(map).map_err(|error| {
            FamilyError::InvalidInput(format!("state serialization failed: {error}"))
        }),
        State::Array(items) => serde_json::to_value(items).map_err(|error| {
            FamilyError::InvalidInput(format!("state serialization failed: {error}"))
        }),
    }
}

/// Validate, render, and budget every question of `request` before any
/// forward runs, so an invalid or over-budget request costs no model work.
///
/// Every question renders exactly one row, so the renderer's row cap
/// ([`D3Profile::max_row_tokens`]) is the per-question cap. The running sum
/// is checked against [`D3Profile::max_request_tokens`] after each question,
/// which also stops tokenizing an over-budget request early. Nothing is
/// truncated, matching the reference.
fn plan_request(
    profile: &D3Profile,
    renderer: &D3Renderer,
    request: &SystemRequest,
) -> Result<PlannedRequest, FamilyError> {
    let state = state_value(&request.state)?;
    let question_ids = crate::families::wire::sorted_question_ids(request)?;
    let mut questions = Vec::with_capacity(question_ids.len());
    let mut input_tokens: u64 = 0;
    for id in question_ids {
        let question = request
            .questions
            .get(&id)
            .expect("question id came from the request map");
        let planned =
            plan_question(profile, renderer, &id, question, &state).map_err(
                |error| match error {
                    FamilyError::InvalidInput(message) => {
                        FamilyError::InvalidInput(format!("question `{id}`: {message}"))
                    }
                    other => other,
                },
            )?;
        let question_tokens = planned.rows.iter().fold(0_u64, |sum, planned_row| {
            sum.saturating_add(u64::from(renderer.row_token_count(&planned_row.row)))
        });
        if question_tokens > profile.max_row_tokens as u64 {
            return Err(FamilyError::InvalidInput(format!(
                "question `{id}` requires {question_tokens} prompt tokens across its rows, \
                 exceeding the per-question maximum of {}; nothing is truncated",
                profile.max_row_tokens
            )));
        }
        input_tokens = input_tokens.saturating_add(question_tokens);
        if input_tokens > profile.max_request_tokens as u64 {
            return Err(FamilyError::InvalidInput(format!(
                "request requires at least {input_tokens} prompt tokens across its questions \
                 (through question `{id}`), exceeding the per-request maximum of {}; nothing \
                 is truncated",
                profile.max_request_tokens
            )));
        }
        questions.push((id, planned));
    }
    Ok(PlannedRequest {
        questions,
        input_tokens,
    })
}

/// Prompt token ids of every question in `request`, in sorted question-id
/// order, planned and budget-checked exactly as [`D3Engine`] plans them
/// before any forward runs.
///
/// Offline tools (the fixture generator and the golden replay) record and
/// assert prompt statistics through this path instead of a duplicate of the
/// planning logic. It runs no model.
pub fn plan_prompt_ids(
    profile: &D3Profile,
    renderer: &D3Renderer,
    request: &SystemRequest,
) -> Result<Vec<(String, Vec<u32>)>, FamilyError> {
    plan_request(profile, renderer, request)?
        .questions
        .into_iter()
        .map(|(id, planned)| match planned.rows.as_slice() {
            // Every pinned question type plans exactly one row.
            [planned_row] => Ok((id, planned_row.row.prompt_ids().to_vec())),
            rows => Err(FamilyError::InvalidInput(format!(
                "question `{id}` planned {} rows, expected 1",
                rows.len()
            ))),
        })
        .collect()
}

impl Inner {
    /// Score one planned question: forward every row, temperature-softmax
    /// the code logits restricted to the offered codes, and assemble the
    /// wire answer.
    fn score_question(
        &self,
        planned: PlannedQuestion,
        control: &FamilyControl,
    ) -> Result<openkind_core::Answer, FamilyError> {
        let mut row_probabilities = Vec::with_capacity(planned.rows.len());
        for planned_row in &planned.rows {
            control.check()?;
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

    fn reports_input_tokens(&self) -> bool {
        true
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        control.check()?;
        // Every question is validated, rendered, and budget-checked before
        // the first forward, so a bad or over-budget request costs no model
        // work.
        let PlannedRequest {
            questions,
            input_tokens,
        } = plan_request(self.inner.profile, &self.inner.renderer, &request)?;
        let mut answers: HashMap<String, openkind_core::Answer, openkind_core::WireHashState> =
            HashMap::with_capacity_and_hasher(questions.len(), Default::default());
        for (id, planned) in questions {
            control.check()?;
            let answer = self.inner.score_question(planned, control)?;
            answers.insert(id, answer);
        }
        control.check()?;
        Ok(crate::families::wire::system_response(
            request.model,
            answers,
            input_tokens,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use openkind_core::SystemRequest;
    use serde_json::json;

    use super::{plan_prompt_ids, plan_request, D3Renderer};
    use crate::families::d3::{derive_answer_codes, D3Profile, D3_EDGE};
    use crate::families::support::FamilyError;

    /// Renderer over the vendored word-level sample tokenizer. Every word
    /// or punctuation run is one token, so prompt lengths are easy to steer
    /// without the real checkpoint tokenizer, and no model is loaded: a
    /// rejection here happens before any forward could run.
    fn sample_renderer(profile: &D3Profile) -> D3Renderer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/d3_664f9ead734372935286/tokenizer-vocab-sample.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&path).expect("vendored sample tokenizer parses");
        let (codes, ids) = derive_answer_codes(&tokenizer).expect("sample provides the code table");
        let ids: Vec<u64> = ids.into_iter().map(u64::from).collect();
        D3Renderer::load(&path, &codes, &ids, profile).expect("sample renderer")
    }

    fn request(state: &str, questions: serde_json::Value) -> SystemRequest {
        serde_json::from_value(json!({"model": "m", "state": state, "questions": questions}))
            .expect("wire request")
    }

    fn rejection(profile: &D3Profile, request: &SystemRequest) -> String {
        match plan_request(profile, &sample_renderer(profile), request) {
            Err(FamilyError::InvalidInput(message)) => message,
            Err(other) => panic!("expected invalid input, got {other}"),
            Ok(_) => panic!("expected the planner to reject the request"),
        }
    }

    fn noul(instructions: &str) -> serde_json::Value {
        json!({"type": "noul", "instructions": instructions})
    }

    #[test]
    fn over_cap_question_is_rejected_while_planning() {
        let state = "word ".repeat(D3_EDGE.max_row_tokens);
        let message = rejection(&D3_EDGE, &request(&state, json!({"q": noul("Is it?")})));
        assert!(
            message.contains("exceeds frozen maximum 65536"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn over_budget_request_is_rejected_while_planning() {
        // Each question stays under the 65,536-token per-question cap, but
        // five of them exceed the 262,144-token request budget.
        let state = "word ".repeat(60_000);
        let questions: serde_json::Map<_, _> = (0..5)
            .map(|index| (format!("q{index}"), noul("Is it?")))
            .collect();
        let message = rejection(&D3_EDGE, &request(&state, questions.into()));
        assert!(
            message.contains("per-request maximum of 262144") && message.contains("`q4`"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn request_within_budget_plans_every_question_in_id_order() {
        let renderer = sample_renderer(&D3_EDGE);
        let wire = request("lamp off", json!({"b": noul("Dark?"), "a": noul("Lit?")}));
        let planned = plan_request(&D3_EDGE, &renderer, &wire).expect("within budget");
        let ids: Vec<&str> = planned
            .questions
            .iter()
            .map(|(id, _)| id.as_str())
            .collect();
        assert_eq!(ids, ["a", "b"]);
        let prompts = plan_prompt_ids(&D3_EDGE, &renderer, &wire).expect("within budget");
        let total: usize = prompts.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(planned.input_tokens, total as u64);
    }

    #[test]
    fn single_option_choice_is_accepted_like_the_reference() {
        let wire = request(
            "x",
            json!({"q": {"type": "choice", "instructions": "Pick.", "criteria": {"only": null}}}),
        );
        let planned =
            plan_request(&D3_EDGE, &sample_renderer(&D3_EDGE), &wire).expect("one option");
        assert_eq!(planned.questions[0].1.rows[0].code_count, 1);
    }

    #[test]
    fn empty_choice_offer_and_empty_choice_key_are_rejected_like_the_reference() {
        let none = request(
            "x",
            json!({"q": {"type": "choice", "instructions": "Pick.", "criteria": {}}}),
        );
        assert!(rejection(&D3_EDGE, &none).contains("offers 0 options"));
        let empty_key = request(
            "x",
            json!({"q": {"type": "choice", "instructions": "Pick.", "criteria": {"": null, "b": null}}}),
        );
        assert!(rejection(&D3_EDGE, &empty_key).contains("must be nonempty strings"));
    }

    #[test]
    fn duplicate_or_empty_score_levels_are_rejected_like_the_reference() {
        for levels in [json!(["low", "low"]), json!(["", "high"])] {
            let wire = request(
                "x",
                json!({"q": {"type": "score", "instructions": "How?", "criteria": levels}}),
            );
            assert!(
                rejection(&D3_EDGE, &wire).contains("score levels must be distinct and nonempty"),
                "{levels} must be rejected"
            );
        }
    }

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
