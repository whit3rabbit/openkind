//! `DecisionEngine` adapter for the pinned `sieve` profiles.

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse};

use crate::families::support::QuestionKind;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};
use crate::families::wire;

use super::model::{SieveModel, VerifiedArtifacts};
use super::renderer::{oneline, render, SieveRenderer};
use super::{SieveEngineConfig, SieveError, SieveProfile};

/// Loaded pinned engine for one profile of the family.
pub struct SieveEngine {
    inner: Arc<Inner>,
}

struct Inner {
    profile: &'static SieveProfile,
    renderer: SieveRenderer,
    model: SieveModel,
    backend_id: String,
}

/// One planned question: the rendered branches (one per listing order),
/// the number of code logits to read, and the listing-to-canonical
/// permutation of each branch.
struct PlannedQuestion {
    unpacked: wire::UnpackedQuestion,
    kind: QuestionKind,
    rows: Vec<PlannedRow>,
}

struct PlannedRow {
    /// Branch token ids without the shared prefix, so a planned request
    /// holds the prefix once rather than once per row.
    branch_ids: Vec<u32>,
    code_count: usize,
    /// Listing position -> canonical option index (the reference `perm`).
    perm: Vec<usize>,
}

impl PlannedQuestion {
    /// Token count of the question's longest branch (reference
    /// `engine.py:evaluate` `per_q`); noul's two listings count once.
    fn longest_branch(&self) -> usize {
        self.rows
            .iter()
            .map(|row| row.branch_ids.len())
            .max()
            .unwrap_or(0)
    }
}

/// The validated, budget-checked scoring work of one request.
///
/// [`plan_request`] builds it without touching the model, and
/// [`SieveEngine`] scores it only after every question rendered and both
/// token caps passed.
pub struct SieveRequestPlan {
    prefix_ids: Vec<u32>,
    /// Questions in evaluation order (sorted wire ids).
    questions: Vec<(String, PlannedQuestion)>,
    input_tokens: u64,
}

impl SieveRequestPlan {
    /// Shared prefix token ids (preamble head plus fenced state).
    pub fn prefix_ids(&self) -> &[u32] {
        &self.prefix_ids
    }

    /// Billed input tokens (reference `engine.py:evaluate` with its default
    /// `bill_preamble=False`): the state tokens once, preamble excluded,
    /// plus each question's longest branch.
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
    }

    /// Every scoring row as `(question id, prefix then branch token ids)`,
    /// in evaluation order (noul Yes-first, then No-first).
    pub fn rows(&self) -> Vec<(&str, Vec<u32>)> {
        self.questions
            .iter()
            .flat_map(|(id, planned)| {
                planned
                    .rows
                    .iter()
                    .map(|row| (id.as_str(), row_ids(&self.prefix_ids, &row.branch_ids)))
            })
            .collect()
    }
}

/// One causal scoring row: the shared prefix followed by one branch.
fn row_ids(prefix_ids: &[u32], branch_ids: &[u32]) -> Vec<u32> {
    let mut ids = Vec::with_capacity(prefix_ids.len() + branch_ids.len());
    ids.extend_from_slice(prefix_ids);
    ids.extend_from_slice(branch_ids);
    ids
}

/// Plan one request the way the engine evaluates it (reference
/// `engine.py:evaluate` up to the forward): serialize the state, validate
/// and render every question, then apply both token caps.
///
/// Nothing here runs the model, so an invalid or over-length question fails
/// the request before any forward, wherever it sits in the request.
pub fn plan_request(
    renderer: &SieveRenderer,
    profile: &SieveProfile,
    request: &SystemRequest,
) -> Result<SieveRequestPlan, FamilyError> {
    // Offline tools plan borrowed requests; only they pay for these copies.
    plan_owned_request(
        renderer,
        profile,
        request.state.clone(),
        request.questions.clone(),
    )
}

/// [`plan_request`] over a request whose state and questions are already
/// owned, so serving a request moves them instead of copying.
fn plan_owned_request(
    renderer: &SieveRenderer,
    profile: &SieveProfile,
    state: State,
    questions: HashMap<String, openkind_core::Question, openkind_core::WireHashState>,
) -> Result<SieveRequestPlan, FamilyError> {
    let sorted = wire::into_sorted_questions(questions)?;
    let state = wire::state_value(state);
    let prefix_ids = renderer.render_prefix(&state)?;
    let mut questions = Vec::with_capacity(sorted.len());
    for (id, question) in sorted {
        let planned =
            plan_question(renderer, profile, &id, &question).map_err(|error| match error {
                FamilyError::InvalidInput(message) => {
                    FamilyError::InvalidInput(format!("question `{id}`: {message}"))
                }
                other => other,
            })?;
        questions.push((id, planned));
    }
    // Reference: the state is the prefix minus the fixed preamble head.
    let state_tokens = prefix_ids
        .len()
        .saturating_sub(renderer.preamble_token_count());
    let per_question: Vec<usize> = questions
        .iter()
        .map(|(_, planned)| planned.longest_branch())
        .collect();
    let input_tokens = check_token_budget(profile, state_tokens, &per_question)?;
    Ok(SieveRequestPlan {
        prefix_ids,
        questions,
        input_tokens,
    })
}

/// Apply the reference token caps (`contract.py:token_limit_error`) and
/// return the billed input tokens.
///
/// Both caps exclude the preamble. The per-question cap is checked first,
/// matching the reference's error precedence.
fn check_token_budget(
    profile: &SieveProfile,
    state_tokens: usize,
    per_question: &[usize],
) -> Result<u64, FamilyError> {
    let longest = per_question.iter().copied().max().unwrap_or(0);
    let longest_branch = state_tokens.saturating_add(longest);
    if longest_branch > profile.max_branch_tokens {
        return Err(FamilyError::InvalidInput(format!(
            "State plus the longest question is {longest_branch} tokens, which exceeds the \
             maximum context length of {} tokens per question.",
            profile.max_branch_tokens
        )));
    }
    let total = per_question
        .iter()
        .fold(state_tokens, |sum, tokens| sum.saturating_add(*tokens));
    if total > profile.max_request_tokens {
        return Err(FamilyError::InvalidInput(format!(
            "Request is {total} tokens, which exceeds the maximum context length of {} tokens \
             per request.",
            profile.max_request_tokens
        )));
    }
    Ok(total as u64)
}

/// Plan the scoring branches of one wire question (reference
/// `prompt.py:encode`): one branch for Choice and Score, two branches for
/// Noul (Yes-first and No-first listing, averaged to cancel listing bias).
///
/// Branches render without the prefix; [`row_ids`] joins them at forward
/// time.
fn plan_question(
    renderer: &SieveRenderer,
    profile: &SieveProfile,
    id: &str,
    question: &openkind_core::Question,
) -> Result<PlannedQuestion, FamilyError> {
    match question {
        openkind_core::Question::Choice(choice) => {
            let count = choice.criteria.len();
            // Reference `contract.py:check_limits` and `prompt.py:branch`
            // accept 1..=255 options, as does core validation.
            if !(1..=profile.max_candidates).contains(&count) {
                return Err(FamilyError::InvalidInput(format!(
                    "choice question `{id}` offers {count} options; the profile serves \
                     1..={}",
                    profile.max_candidates
                )));
            }
            let unpacked = wire::unpack_question(id, question)?;
            // The wire map is a hash map, so options render in the wire's
            // sorted label order: the family's declared determinism
            // difference. Option text: `key` or `key — description`
            // (null descriptions render the bare key).
            let options: Vec<String> = unpacked
                .labels
                .iter()
                .map(|label| match choice.criteria.get(label) {
                    Some(Some(description)) if !description.is_empty() => {
                        format!(
                            "{} — {}",
                            oneline(&render(&serde_json::Value::String(label.clone()))),
                            oneline(&render(&serde_json::Value::String(description.clone())))
                        )
                    }
                    _ => oneline(&render(&serde_json::Value::String(label.clone()))),
                })
                .collect();
            let listing: Vec<usize> = (0..count).collect();
            let row = renderer.render_branch(
                &[],
                "choice",
                &question_instructions(question),
                &options,
                &listing,
            )?;
            Ok(PlannedQuestion {
                unpacked,
                kind: QuestionKind::Choice,
                rows: vec![PlannedRow {
                    branch_ids: row.prompt_ids().to_vec(),
                    code_count: count,
                    perm: listing,
                }],
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
            let unpacked = wire::unpack_question(id, question)?;
            let options: Vec<String> = score
                .criteria
                .iter()
                .map(|level| oneline(&render(&serde_json::Value::String(level.clone()))))
                .collect();
            let listing: Vec<usize> = (0..levels).collect();
            let row = renderer.render_branch(
                &[],
                "score",
                &question_instructions(question),
                &options,
                &listing,
            )?;
            Ok(PlannedQuestion {
                unpacked,
                kind: QuestionKind::Score,
                rows: vec![PlannedRow {
                    branch_ids: row.prompt_ids().to_vec(),
                    code_count: levels,
                    perm: listing,
                }],
            })
        }
        openkind_core::Question::Noul(noul) => {
            let unpacked = wire::unpack_question(id, question)?;
            // Canonical option texts are [No, Yes]; the default listing
            // is Yes-first (perm [1, 0]) and the second branch flips it.
            let (false_text, true_text) = noul
                .criteria
                .as_ref()
                .map(|criteria| (criteria.r#false.clone(), criteria.r#true.clone()))
                .unwrap_or_default();
            let describe = |text: &str| -> String {
                if text.is_empty() {
                    String::new()
                } else {
                    format!(
                        " — {}",
                        oneline(&render(&serde_json::Value::String(text.to_owned())))
                    )
                }
            };
            let options = vec![
                format!("No{}", describe(&false_text)),
                format!("Yes{}", describe(&true_text)),
            ];
            let mut rows = Vec::with_capacity(2);
            for listing in [[1_usize, 0_usize], [0_usize, 1_usize]] {
                let row = renderer.render_branch(
                    &[],
                    "noul",
                    &question_instructions(question),
                    &options,
                    &listing,
                )?;
                rows.push(PlannedRow {
                    branch_ids: row.prompt_ids().to_vec(),
                    code_count: 2,
                    perm: listing.to_vec(),
                });
            }
            Ok(PlannedQuestion {
                unpacked,
                kind: QuestionKind::Noul,
                rows,
            })
        }
    }
}

impl Inner {
    /// Score one planned question: forward every row, temperature-softmax
    /// the code logits, map each listing distribution back to canonical
    /// option order, and average the rows (noul's two orders).
    fn score_question(
        &self,
        prefix_ids: &[u32],
        planned: PlannedQuestion,
        control: &FamilyControl,
    ) -> Result<openkind_core::Answer, FamilyError> {
        let temperature = self.profile.calibration.resolve(planned.kind);
        let mut acc = vec![0.0_f64; planned.unpacked.labels.len()];
        let mut rows = 0_usize;
        for planned_row in &planned.rows {
            control.check()?;
            let logits = self.model.code_logits(
                &row_ids(prefix_ids, &planned_row.branch_ids),
                planned_row.code_count,
                control,
            )?;
            let listing_probabilities = temperature_softmax(&logits, temperature)?;
            if listing_probabilities.len() != planned_row.perm.len() {
                return Err(FamilyError::Numerical(
                    "code readout lost an option's probability".to_owned(),
                ));
            }
            // Reference `canonical_probs`: listing position -> canonical
            // option index.
            for (position, canonical_index) in planned_row.perm.iter().enumerate() {
                acc[*canonical_index] += listing_probabilities[position];
            }
            rows += 1;
        }
        for value in &mut acc {
            *value /= rows as f64;
        }
        wire::answer_from_probabilities(&planned.unpacked, &acc)
    }
}

/// The wire instructions payload of one question.
fn question_instructions(question: &openkind_core::Question) -> serde_json::Value {
    match question {
        openkind_core::Question::Noul(question) => question.instructions.clone(),
        openkind_core::Question::Choice(question) => question.instructions.clone(),
        openkind_core::Question::Score(question) => question.instructions.clone(),
    }
}

impl SieveEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    pub fn load(config: SieveEngineConfig) -> Result<BoundedFamilyEngine, SieveError> {
        Self::load_with_execution(config, crate::device::FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// The code head runs on the Qwen3.5 hybrid backbone, which has no ONNX
    /// export; accelerated loads require the `cuda` feature.
    pub fn load_with_execution(
        config: SieveEngineConfig,
        execution: crate::device::FamilyExecution,
    ) -> Result<BoundedFamilyEngine, SieveError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root, config.profile)?;
        let sieve_json: serde_json::Value = crate::families::support::read_json(
            &config
                .model_root
                .join("checkpoint")
                .join("sieve_config.json"),
        )?;
        let pinned_codes: Vec<String> = decode_strings(&sieve_json["codes"])?;
        let pinned_token_ids: Vec<u64> = decode_numbers(&sieve_json["code_ids"])?;
        let renderer = SieveRenderer::load(&artifacts.tokenizer, &pinned_codes, &pinned_token_ids)?;
        let backend_id: String = match execution {
            crate::device::FamilyExecution::Cpu => config.profile.cpu_backend_id.to_owned(),
            #[cfg(feature = "cuda")]
            crate::device::FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device().map_err(FamilyError::from)?;
                let model = SieveModel::load_with_device(&artifacts, config.profile, device)?;
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
                    "the sieve code head runs on the Qwen3.5 hybrid backbone, which has no \
                     ONNX export; select cpu or (with the `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
            #[cfg(feature = "onnx-rocm")]
            crate::device::FamilyExecution::OnnxRocm { .. } => {
                return Err(crate::families::support::FamilyError::ExecutionUnavailable(
                    "the sieve code head runs on the Qwen3.5 hybrid backbone, which has no \
                     ONNX export and therefore no ROCm execution; select cpu or (with the \
                     `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
        };
        let model = SieveModel::load(&artifacts, config.profile)?;
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
        field: "sieve_config.codes",
        expected: "a JSON string array".to_owned(),
        actual: error.to_string(),
    })
}

/// Decode a pinned JSON integer array.
fn decode_numbers(value: &serde_json::Value) -> Result<Vec<u64>, FamilyError> {
    serde_json::from_value(value.clone()).map_err(|error| FamilyError::ContractMismatch {
        field: "sieve_config.code_ids",
        expected: "a JSON integer array".to_owned(),
        actual: error.to_string(),
    })
}

impl FamilyEvaluator for SieveEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} LoRA-merged code-readout decision engine ({}, {}).",
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
        // Validate, render, and budget every question before the first
        // forward (reference `engine.py:evaluate`), so a later invalid or
        // over-length question never costs the earlier ones' forwards.
        let SieveRequestPlan {
            prefix_ids,
            questions,
            input_tokens,
        } = plan_owned_request(
            &self.inner.renderer,
            self.inner.profile,
            request.state,
            request.questions,
        )?;
        let mut answers: HashMap<String, openkind_core::Answer, openkind_core::WireHashState> =
            HashMap::with_capacity_and_hasher(questions.len(), Default::default());
        for (id, planned) in questions {
            control.check()?;
            let answer = self.inner.score_question(&prefix_ids, planned, control)?;
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
    use super::*;
    use crate::families::sieve::SIEVE_4B;

    #[test]
    fn budget_bills_the_state_once_plus_each_longest_branch() {
        // Golden `choice-two`: 17 state tokens plus one 57-token branch.
        assert_eq!(check_token_budget(&SIEVE_4B, 17, &[57]).unwrap(), 74);
        // Two questions share one state.
        assert_eq!(check_token_budget(&SIEVE_4B, 21, &[47, 40]).unwrap(), 108);
    }

    #[test]
    fn budget_accepts_each_cap_and_rejects_one_token_over() {
        assert!(check_token_budget(&SIEVE_4B, 32_000, &[768]).is_ok());
        let error = check_token_budget(&SIEVE_4B, 32_000, &[769]).unwrap_err();
        assert!(
            error.to_string().contains("32769 tokens")
                && error.to_string().contains("per question"),
            "{error}"
        );
        // 1_000 + 3 * 21_512 = 65_536 exactly; each branch stays under the
        // per-question cap.
        assert!(check_token_budget(&SIEVE_4B, 1_000, &[21_512, 21_512, 21_512]).is_ok());
        let error = check_token_budget(&SIEVE_4B, 1_000, &[21_512, 21_512, 21_513]).unwrap_err();
        assert!(
            error.to_string().contains("65537 tokens") && error.to_string().contains("per request"),
            "{error}"
        );
    }

    #[test]
    fn per_question_cap_is_reported_before_the_request_cap() {
        let error = check_token_budget(&SIEVE_4B, 40_000, &[30_000, 30_000]).unwrap_err();
        assert!(error.to_string().contains("per question"), "{error}");
    }
}
