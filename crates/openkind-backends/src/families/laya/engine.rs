//! `DecisionEngine` adapter for the pinned laya profiles.

use std::sync::Arc;

use openkind_core::{ModelInfo, Question, State, SystemRequest, SystemResponse, Usage};

use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{LayaModel, VerifiedArtifacts};
use super::renderer::{build_sequence, LayaRenderer, SequencePlan};
use super::{
    LayaEngineConfig, LayaError, LayaProfile, MAX_CANDIDATES, QTYPE_CHOICE, QTYPE_NOUL, QTYPE_SCORE,
};

/// Loaded pinned laya engine.
pub struct LayaEngine {
    inner: Arc<Inner>,
}

struct Inner {
    profile: &'static LayaProfile,
    renderer: LayaRenderer,
    model: LayaModel,
}

/// The reference's temperature clamp: confine to `[0.5, 5.0]`, falling back
/// to 1.0 for non-finite values. Applied to the pinned raw tables, exactly
/// as the reference applies it at load.
pub(crate) fn clamp_temperature(value: f64) -> f64 {
    if !value.is_finite() {
        return 1.0;
    }
    value.clamp(0.5, 5.0)
}

/// The reference's per-option-count bucket key: `"{type}:{2|3-5|6-10|11+}"`.
fn temp_bucket(type_name: &str, options: usize) -> String {
    let size = if options <= 2 {
        "2"
    } else if options <= 5 {
        "3-5"
    } else if options <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{type_name}:{size}")
}

fn resolve_temperature(
    profile: &LayaProfile,
    type_name: &str,
    qtype: usize,
    options: usize,
) -> f64 {
    let bucket = temp_bucket(type_name, options);
    let raw = profile
        .temperature_by_options
        .iter()
        .find(|(key, _)| *key == bucket)
        .map(|&(_, temperature)| temperature)
        .unwrap_or(profile.temperature[qtype]);
    clamp_temperature(raw)
}

/// Serialize the wire state as the reference's `json.dumps` payload:
/// compact separators `", "` / `": "`, non-ASCII kept, keys in the parsed
/// map's (sorted) order. Strings pass through untouched.
fn laya_state_text(state: &State) -> Result<String, FamilyError> {
    match state {
        State::Text(text) => Ok(text.clone()),
        State::Object(_) | State::Array(_) => {
            let value = match state {
                State::Object(map) => serde_json::to_value(map),
                State::Array(items) => serde_json::to_value(items),
                _ => unreachable!("matched above"),
            }
            .map_err(|error| {
                FamilyError::InvalidInput(format!("state serialization failed: {error}"))
            })?;
            let mut out = String::new();
            write_python_json(&value, &mut out);
            Ok(out)
        }
    }
}

/// Minimal Python-`json.dumps`-shaped writer (default separators, sorted map
/// order from the parsed `serde_json` map).
fn write_python_json(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    use std::fmt::Write;
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => {
            let _ = write!(out, "{number}");
        }
        Value::String(text) => {
            // serde_json's string escaping is the same minimal set as
            // Python's ensure_ascii=False output.
            let _ = write!(
                out,
                "{}",
                serde_json::to_string(text).expect("string serializes")
            );
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                let _ = write!(
                    out,
                    "{}: ",
                    serde_json::to_string(key).expect("key serializes")
                );
                write_python_json(item, out);
            }
            out.push('}');
        }
    }
}

/// One question's rendering inputs: type name, type-embedding row, and the
/// rendered option texts in candidate order.
struct RenderedQuestionInputs {
    type_name: &'static str,
    qtype: usize,
    options: Vec<String>,
}

/// Render laya option texts from the raw wire question, mirroring the
/// reference's `render_options` (including its noul criteria defaults).
fn render_question_inputs(question: &Question) -> Result<RenderedQuestionInputs, FamilyError> {
    match question {
        Question::Choice(choice) => {
            // Candidate order is the shared wire contract (sorted labels,
            // including a reserved `__none__` key); the option text keeps
            // the reference's `label: description` rule.
            let unpacked = crate::families::wire::unpack_question("choice", question)?;
            let mut options = Vec::with_capacity(unpacked.labels.len());
            for label in &unpacked.labels {
                let description = choice
                    .criteria
                    .get(label)
                    .and_then(|value| value.as_deref());
                let rendered = match description {
                    Some(text) if !text.is_empty() => format!("{label}: {text}"),
                    _ => label.clone(),
                };
                options.push(rendered);
            }
            Ok(RenderedQuestionInputs {
                type_name: "choice",
                qtype: QTYPE_CHOICE,
                options,
            })
        }
        Question::Score(score) => Ok(RenderedQuestionInputs {
            type_name: "score",
            qtype: QTYPE_SCORE,
            options: score
                .criteria
                .iter()
                .enumerate()
                .map(|(index, level)| format!("level {index}: {level}"))
                .collect(),
        }),
        Question::Noul(noul) => {
            const DEFAULT_FALSE: &str = "no, the statement does not hold";
            const DEFAULT_TRUE: &str = "yes, the statement holds";
            let (false_criterion, true_criterion) = noul
                .criteria
                .as_ref()
                .map(|criteria| (criteria.r#false.as_str(), criteria.r#true.as_str()))
                .unwrap_or(("", ""));
            let options = vec![
                format!(
                    "false: {}",
                    if false_criterion.is_empty() {
                        DEFAULT_FALSE
                    } else {
                        false_criterion
                    }
                ),
                format!(
                    "true: {}",
                    if true_criterion.is_empty() {
                        DEFAULT_TRUE
                    } else {
                        true_criterion
                    }
                ),
            ];
            Ok(RenderedQuestionInputs {
                type_name: "noul",
                qtype: QTYPE_NOUL,
                options,
            })
        }
    }
}

impl LayaEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    pub fn load(config: LayaEngineConfig) -> Result<BoundedFamilyEngine, LayaError> {
        let profile = config.profile;
        let artifacts = VerifiedArtifacts::verify(&config.model_root, profile)?;
        let renderer = LayaRenderer::load(&artifacts.tokenizer, &profile.specials)?;
        let model = LayaModel::load(profile, &artifacts)?;
        let engine = Self {
            inner: Arc::new(Inner {
                profile,
                renderer,
                model,
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for LayaEngine {
    fn backend_id(&self) -> &str {
        self.inner.profile.backend_id()
    }

    fn model_metadata(&self) -> ModelInfo {
        let profile = self.inner.profile;
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} decision encoder ({}, FP32 CPU).",
                profile.backbone_id, profile.profile_id
            ),
            release_date: "2026-09-24".into(),
        }
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        control.check()?;
        let profile = self.inner.profile;
        let state_serialized = laya_state_text(&request.state)?;
        if state_serialized.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "state must contain non-whitespace text".to_owned(),
            ));
        }
        // A chronological conversation list truncates from the left so the
        // newest turn survives; every other state keeps its first tokens.
        let truncate_left = matches!(request.state, State::Array(_));
        let state_text =
            state_serialized.replace(super::model::mask_token_of(&profile.specials), " ");
        let state_ids = self.inner.renderer.encode_plain(&state_text)?;

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
            let inputs = render_question_inputs(question)?;
            if inputs.options.len() > MAX_CANDIDATES {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` offers {} options but the profile accepts at most {MAX_CANDIDATES}",
                    inputs.options.len()
                )));
            }
            let plan = SequencePlan {
                renderer: &self.inner.renderer,
                specials: &profile.specials,
                max_len: profile.max_sequence_tokens,
                head_max_len: profile.head_max_len,
                truncate_left,
            };
            let rendered = build_sequence(
                &plan,
                inputs.type_name,
                &crate::families::wire::instruction_text(question)?,
                &inputs.options,
                &state_ids,
            )?;
            input_tokens = input_tokens.saturating_add(rendered.ids.len() as u64);
            let logits =
                self.inner
                    .model
                    .option_logits(&rendered.ids, &rendered.markers, inputs.qtype)?;
            let temperature = resolve_temperature(
                profile,
                inputs.type_name,
                inputs.qtype,
                inputs.options.len(),
            );
            let probabilities = temperature_softmax(&logits, temperature)?;
            let unpacked = crate::families::wire::unpack_question(id, question)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_buckets_match_the_reference_sizes() {
        assert_eq!(temp_bucket("choice", 1), "choice:2");
        assert_eq!(temp_bucket("choice", 2), "choice:2");
        assert_eq!(temp_bucket("choice", 3), "choice:3-5");
        assert_eq!(temp_bucket("score", 5), "score:3-5");
        assert_eq!(temp_bucket("score", 6), "score:6-10");
        assert_eq!(temp_bucket("noul", 11), "noul:11+");
    }

    #[test]
    fn resolve_temperature_prefers_buckets_and_clamps() {
        // English profile: the 11+ sharpening bucket clamps to the floor.
        let profile = &super::super::LAYA_ENGLISH;
        assert_eq!(
            resolve_temperature(profile, "choice", QTYPE_CHOICE, 2),
            1.906_356_334_686_279_3
        );
        assert_eq!(
            resolve_temperature(profile, "choice", QTYPE_CHOICE, 11),
            0.5
        );
        // No fitted bucket for noul sizes beyond 2: per-type fallback.
        assert_eq!(
            resolve_temperature(profile, "noul", QTYPE_NOUL, 2),
            clamp_temperature(1.983_399_510_383_606)
        );
        // Multilingual ships no buckets at all.
        let profile = &super::super::LAYA_MULTILINGUAL;
        assert_eq!(resolve_temperature(profile, "score", QTYPE_SCORE, 4), 1.0);
    }

    #[test]
    fn object_state_serializes_with_python_dumps_separators() {
        let value: serde_json::Value =
            serde_json::json!({"b": 1, "a": [true, null, "x"], "c": {"k": "v"}});
        let mut out = String::new();
        write_python_json(&value, &mut out);
        // serde_json's map sorts keys, matching the generator's sort_keys.
        assert_eq!(out, r#"{"a": [true, null, "x"], "b": 1, "c": {"k": "v"}}"#);
    }

    #[test]
    fn text_state_passes_through_untouched() {
        let state = State::Text("plain".into());
        assert_eq!(laya_state_text(&state).expect("serialize"), "plain");
    }
}
