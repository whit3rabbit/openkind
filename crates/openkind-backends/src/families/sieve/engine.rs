//! `DecisionEngine` adapter for the pinned `sieve` profiles.

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::decoder_logit_qwen35::QuestionKind;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};
use crate::families::wire;

use super::model::{SieveModel, VerifiedArtifacts};
use super::renderer::{oneline, render, RenderedRow, SieveRenderer};
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

/// One planned question: the rendered rows (one per listing order), the
/// number of code logits to read, and the listing-to-canonical permutation
/// of each row.
struct PlannedQuestion {
    unpacked: wire::UnpackedQuestion,
    kind: QuestionKind,
    rows: Vec<PlannedRow>,
}

struct PlannedRow {
    row: RenderedRow,
    code_count: usize,
    /// Listing position -> canonical option index (the reference `perm`).
    perm: Vec<usize>,
}

impl Inner {
    /// Plan the scoring rows of one wire question (reference
    /// `prompt.py:encode`): one branch for Choice and Score, two branches
    /// for Noul (Yes-first and No-first listing, averaged to cancel listing
    /// bias).
    fn plan_question(
        &self,
        id: &str,
        question: &openkind_core::Question,
        prefix_ids: &[u32],
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
                let unpacked = wire::unpack_question(id, question)?;
                // The wire map is a hash map, so options render in the wire's
                // sorted label order — the family's declared determinism
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
                let row = self.renderer.render_branch(
                    prefix_ids,
                    "choice",
                    &question_instructions(question),
                    &options,
                    &listing,
                )?;
                Ok(PlannedQuestion {
                    unpacked,
                    kind: QuestionKind::Choice,
                    rows: vec![PlannedRow {
                        row,
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
                let row = self.renderer.render_branch(
                    prefix_ids,
                    "score",
                    &question_instructions(question),
                    &options,
                    &listing,
                )?;
                Ok(PlannedQuestion {
                    unpacked,
                    kind: QuestionKind::Score,
                    rows: vec![PlannedRow {
                        row,
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
                    let row = self.renderer.render_branch(
                        prefix_ids,
                        "noul",
                        &question_instructions(question),
                        &options,
                        &listing,
                    )?;
                    rows.push(PlannedRow {
                        row,
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

    /// Score one planned question: forward every row, temperature-softmax
    /// the code logits, map each listing distribution back to canonical
    /// option order, and average the rows (noul's two orders).
    fn score_question(
        &self,
        planned: PlannedQuestion,
        control: &FamilyControl,
        token_count: &mut u64,
    ) -> Result<openkind_core::Answer, FamilyError> {
        let temperature = self.profile.calibration.resolve(planned.kind);
        let mut acc = vec![0.0_f64; planned.unpacked.labels.len()];
        let mut rows = 0_usize;
        for planned_row in &planned.rows {
            control.check()?;
            *token_count = token_count
                .saturating_add(u64::from(self.renderer.row_token_count(&planned_row.row)));
            let logits = self.model.code_logits(
                planned_row.row.prompt_ids(),
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
        let renderer = SieveRenderer::load(
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
        let prefix_ids = self.inner.renderer.render_prefix(&state)?;
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
                .plan_question(id, question, &prefix_ids)
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
