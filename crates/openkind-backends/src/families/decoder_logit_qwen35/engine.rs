//! `DecisionEngine` adapter for the pinned `decoder-logit-qwen35` profiles.

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};
use crate::families::wire;

use super::knockout::{combine, PassReader};
use super::model::{Jevk5Model, VerifiedArtifacts};
use super::renderer::{RenderedPass, NOUL_DEFAULT_DESCRIPTIONS};
use super::{
    DecoderLogitQwen35EngineConfig, DecoderLogitQwen35Error, QuestionKind, Qwen35LogitProfile,
};

/// Loaded pinned engine for one profile of the family.
pub struct DecoderLogitQwen35Engine {
    inner: Arc<Inner>,
}

struct Inner {
    profile: &'static Qwen35LogitProfile,
    renderer: super::renderer::Jevk5Renderer,
    model: Jevk5Model,
}

impl Jevk5PassSource for Inner {
    fn renderer(&self) -> &super::renderer::Jevk5Renderer {
        &self.renderer
    }

    fn letter_logits(
        &self,
        pass: &RenderedPass,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        self.model
            .letter_logits(pass.prompt_ids(), pass.letter_ids(), control)
    }
}

impl ProfiledPassSource for Inner {
    fn profile(&self) -> &'static Qwen35LogitProfile {
        self.profile
    }
}

/// Per-pass forward source shared by every execution backend: rendering,
/// calibration, and the knockout schedule are family code, while only the
/// letter-logit forward differs per backend.
pub(crate) trait Jevk5PassSource {
    /// The family renderer (shared verbatim by every backend).
    fn renderer(&self) -> &super::renderer::Jevk5Renderer;

    /// Raw letter logits for one rendered pass (backend-specific forward).
    fn letter_logits(
        &self,
        pass: &RenderedPass,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError>;
}

/// A pass source that serves one pinned profile of this family. Execution
/// backends carry the profile alongside the renderer so the shared
/// evaluation path applies the profile's calibration and pass schedule.
pub(crate) trait ProfiledPassSource: Jevk5PassSource {
    /// The pinned profile this source serves.
    fn profile(&self) -> &'static Qwen35LogitProfile;
}

/// One calibrated read of at most 16 options: render, forward, temperature
/// softmax over the letter logits. Rendered prompt tokens accumulate into
/// `token_count` so usage covers every knockout pass.
struct PassRead<'a> {
    source: &'a dyn Jevk5PassSource,
    profile: &'static Qwen35LogitProfile,
    kind: QuestionKind,
    state: &'a serde_json::Value,
    criterion: &'a str,
    token_count: &'a std::cell::Cell<u64>,
    control: &'a FamilyControl,
}

fn account_question_tokens(
    token_count: &std::cell::Cell<u64>,
    tokens: u64,
    max_question_tokens: u64,
) -> Result<(), FamilyError> {
    token_count.set(token_count.get().saturating_add(tokens));
    if token_count.get() > max_question_tokens {
        return Err(FamilyError::InvalidInput(format!(
            "question requires {} prompt tokens across its passes, exceeding the maximum work budget of {}",
            token_count.get(),
            max_question_tokens
        )));
    }
    Ok(())
}

impl PassReader for PassRead<'_> {
    fn read(&self, options: &[(String, String)]) -> Result<Vec<f64>, FamilyError> {
        // Each knockout pass is a separate non-interruptible forward, so a
        // disconnected caller must not trigger the remaining passes.
        self.control.check()?;
        let pass = self
            .source
            .renderer()
            .render_pass(self.state, self.criterion, options)?;
        let tokens = u64::from(self.source.renderer().pass_token_count(&pass));
        account_question_tokens(self.token_count, tokens, self.profile.max_question_tokens)?;
        self.control.check()?;
        let logits = self.source.letter_logits(&pass, self.control)?;
        self.control.check()?;
        temperature_softmax(&logits, self.profile.calibration.resolve(self.kind))
    }
}

impl DecoderLogitQwen35Engine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The returned adapter implements [`DecisionEngine`] and carries
    /// admission, queue, deadline, and cancellation control.
    pub fn load(
        config: DecoderLogitQwen35EngineConfig,
    ) -> Result<BoundedFamilyEngine, DecoderLogitQwen35Error> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root, config.profile)?;
        let renderer = super::renderer::Jevk5Renderer::load(
            &artifacts.tokenizer,
            config.profile.max_sequence_tokens,
        )?;
        let model = Jevk5Model::load(&artifacts)?;
        let engine = Self {
            inner: Arc::new(Inner {
                profile: config.profile,
                renderer,
                model,
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

/// Score one question into a per-option probability distribution over any
/// execution backend.
///
/// Up to one pass width, a single calibrated read suffices. Wider questions
/// run the profile's reference knockout schedule when its runtime defines
/// one, and fail closed otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn question_probabilities(
    source: &dyn Jevk5PassSource,
    profile: &'static Qwen35LogitProfile,
    kind: QuestionKind,
    state: &serde_json::Value,
    criterion: &str,
    options: &[(String, String)],
    token_count: &std::cell::Cell<u64>,
    control: &FamilyControl,
) -> Result<Vec<f64>, FamilyError> {
    {
        if options.len() < super::MIN_CANDIDATES {
            return Err(FamilyError::InvalidInput(format!(
                "question needs at least {} candidates, found {}",
                super::MIN_CANDIDATES,
                options.len()
            )));
        }
        if options.len() > profile.max_candidates {
            return Err(FamilyError::InvalidInput(format!(
                "question offers {} candidates but profile {} covers at most {}; split the question",
                options.len(),
                profile.loader_id,
                profile.max_candidates
            )));
        }
        control.check()?;
        let reader = PassRead {
            source,
            profile,
            kind,
            state,
            criterion,
            token_count,
            control,
        };
        if options.len() <= super::MAX_OPTIONS_PER_PASS {
            return reader.read(options);
        }
        let Some(knockout_temperature) = profile.knockout_temperature else {
            return Err(FamilyError::InvalidInput(format!(
                "question offers {} candidates but profile {} serves at most {} in a single pass",
                options.len(),
                profile.loader_id,
                super::MAX_OPTIONS_PER_PASS
            )));
        };
        let combined = combine(&reader, options)?;
        // The reference sharpens the combined distribution by
        // q^(1/knockout_temperature) and renormalizes.
        let sharpened: Vec<f64> = combined
            .iter()
            .map(|&probability| {
                if probability <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    probability.ln() / knockout_temperature
                }
            })
            .collect();
        temperature_softmax(&sharpened, 1.0)
    }
}

impl FamilyEvaluator for DecoderLogitQwen35Engine {
    fn backend_id(&self) -> &str {
        self.inner.profile.cpu_backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {} letter-logit decision engine ({}, FP32 CPU).",
                self.inner.profile.backbone_id, self.inner.profile.profile_id
            ),
            release_date: self.inner.profile.release_date.into(),
        }
    }

    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError> {
        evaluate_with(self.inner.as_ref(), request, control)
    }
}

/// Model-agnostic evaluation shared by every execution backend: the
/// backend-specific [`ProfiledPassSource`] is the only input, so both
/// backends apply the identical rendering, calibration, and wire mapping.
pub(crate) fn evaluate_with(
    source: &dyn ProfiledPassSource,
    request: SystemRequest,
    control: &FamilyControl,
) -> Result<SystemResponse, FamilyError> {
    let profile = source.profile();
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
        let criterion = wire::instruction_text(question)?;
        // Noul keeps the reference runtime's default criterion text when
        // the caller supplied none; Choice and Score use the shared wire
        // unpacking.
        let (kind, unpacked) = match question {
            openkind_core::Question::Noul(noul) => {
                let (false_description, true_description) = noul
                    .criteria
                    .as_ref()
                    .map(|criteria| (criteria.r#false.clone(), criteria.r#true.clone()))
                    .unwrap_or_else(|| {
                        (
                            NOUL_DEFAULT_DESCRIPTIONS[0].to_owned(),
                            NOUL_DEFAULT_DESCRIPTIONS[1].to_owned(),
                        )
                    });
                (
                    QuestionKind::Noul,
                    wire::UnpackedQuestion {
                        id: id.clone(),
                        primitive: wire::QuestionPrimitive::Noul,
                        labels: vec!["false".into(), "true".into()],
                        criteria: vec![false_description, true_description],
                        ordered: false,
                    },
                )
            }
            other => {
                let unpacked = wire::unpack_question(id, other)?;
                let kind = match unpacked.primitive {
                    wire::QuestionPrimitive::Noul => QuestionKind::Noul,
                    wire::QuestionPrimitive::Choice => QuestionKind::Choice,
                    wire::QuestionPrimitive::Score => QuestionKind::Score,
                };
                (kind, unpacked)
            }
        };
        let options: Vec<(String, String)> = unpacked
            .labels
            .iter()
            .cloned()
            .zip(unpacked.criteria.iter().cloned())
            .collect();
        let token_count = std::cell::Cell::new(0_u64);
        // The reference renders Noul options true-first (`("true",
        // "false")` in jevk5's decision_options); our wire unpacking is
        // false-first. Render in reference order and map the calibrated
        // distribution back to the wire's candidate order.
        let mut options = options;
        if matches!(unpacked.primitive, wire::QuestionPrimitive::Noul) {
            options.reverse();
        }
        let mut probabilities = question_probabilities(
            source,
            profile,
            kind,
            &state,
            &criterion,
            &options,
            &token_count,
            control,
        )?;
        if matches!(unpacked.primitive, wire::QuestionPrimitive::Noul) {
            probabilities.reverse();
        }
        input_tokens = input_tokens.saturating_add(token_count.get());
        let answer = wire::answer_from_probabilities(&unpacked, &probabilities)?;
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
    fn aggregate_question_work_fails_closed() {
        let tokens = std::cell::Cell::new(super::super::JEVK5.max_question_tokens - 1);
        account_question_tokens(&tokens, 1, super::super::JEVK5.max_question_tokens)
            .expect("the exact budget is accepted");
        let error = account_question_tokens(&tokens, 1, super::super::JEVK5.max_question_tokens)
            .expect_err("excess work must fail");
        assert!(matches!(error, FamilyError::InvalidInput(_)));
    }
}

#[cfg(test)]
mod debug {
    //! Operator debug aid (never part of CI): dumps rendered prompt ids and
    //! raw letter logits for one hand-written case so the bytes can be
    //! diffed against the reference runtime.
    use super::*;

    #[test]
    fn dump_prompt_ids_and_letter_logits() {
        let Some(root) = std::env::var_os("OPENKIND_DECODER_LOGIT_QWEN35_MODEL_ROOT") else {
            eprintln!("skipping: model root not set");
            return;
        };
        let state = serde_json::Value::String("debug state".to_owned());
        let criterion = "debug criterion?";
        let options = vec![
            ("true".to_owned(), "The proposition is true.".to_owned()),
            ("false".to_owned(), "The proposition is false.".to_owned()),
        ];
        let artifacts =
            VerifiedArtifacts::verify(std::path::Path::new(&root), &super::super::JEVK5)
                .expect("verify");
        let renderer = super::super::renderer::Jevk5Renderer::load(
            &artifacts.tokenizer,
            super::super::JEVK5.max_sequence_tokens,
        )
        .expect("renderer");
        let model = Jevk5Model::load(&artifacts).expect("model");
        let pass = renderer
            .render_pass(&state, criterion, &options)
            .expect("render");
        let logits = model
            .letter_logits(
                pass.prompt_ids(),
                pass.letter_ids(),
                &FamilyControl::new(
                    std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                    None,
                    0,
                ),
            )
            .expect("logits");
        eprintln!(
            "PROMPT_IDS {}",
            pass.prompt_ids()
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        eprintln!("LETTER_LOGITS {logits:?}");
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct CancellingSource {
        renderer: super::super::renderer::Jevk5Renderer,
        cancelled: Arc<AtomicBool>,
        calls: Cell<usize>,
    }

    impl ProfiledPassSource for CancellingSource {
        fn profile(&self) -> &'static Qwen35LogitProfile {
            &super::super::JEVK5
        }
    }

    impl Jevk5PassSource for CancellingSource {
        fn renderer(&self) -> &super::super::renderer::Jevk5Renderer {
            &self.renderer
        }

        fn letter_logits(
            &self,
            pass: &RenderedPass,
            _control: &FamilyControl,
        ) -> Result<Vec<f64>, FamilyError> {
            self.calls.set(self.calls.get() + 1);
            self.cancelled.store(true, Ordering::Release);
            Ok(vec![0.0; pass.letter_ids().len()])
        }
    }

    #[test]
    fn cancellation_after_a_forward_stops_single_and_knockout_reads() {
        // The vendored synthetic tokenizer exercises pass control without
        // a checkpoint or the profile's large production tokenizer.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "tests/fixtures/decoder_logit_qwen35_415bcf4a064e6dadcf85/synthetic_control_tokenizer.json",
        );
        crate::families::support::verify_digest(
            &path,
            "790e5d78a52353fcb7766098a8e0044c6d229448602eb78b05a81c3f875315c4",
        )
        .expect("synthetic tokenizer digest");
        let renderer = super::super::renderer::Jevk5Renderer::load(
            &path,
            super::super::JEVK5.max_sequence_tokens,
        )
        .expect("renderer");
        let cancelled = Arc::new(AtomicBool::new(false));
        let control = FamilyControl::new(cancelled.clone(), None, 0);
        let source = CancellingSource {
            renderer,
            cancelled: cancelled.clone(),
            calls: Cell::new(0),
        };
        for count in [2, 20] {
            cancelled.store(false, Ordering::Release);
            source.calls.set(0);
            let options = (0..count)
                .map(|index| (format!("option{index}"), format!("Description {index}")))
                .collect::<Vec<_>>();
            let result = question_probabilities(
                &source,
                source.profile(),
                QuestionKind::Choice,
                &serde_json::json!("evidence"),
                "Pick an option",
                &options,
                &Cell::new(0),
                &control,
            );
            assert!(matches!(result, Err(FamilyError::Cancelled)));
            assert_eq!(source.calls.get(), 1, "no remaining forwards may run");
        }
    }
}
