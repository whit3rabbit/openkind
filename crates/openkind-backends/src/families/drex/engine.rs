//! `DecisionEngine` adapter for the pinned Drex v1.5 profile.
//!
//! Wire mapping is the bundled reference `kev.api.to_record`, the same as the
//! kev family: `Noul` scores `no`/`yes` (annotated with explicit criteria)
//! and reports `p(true)`, `Choice` scores one option per offered key, and
//! `Score` scores the rendered level criteria. Each question runs as its own
//! causal row, so questions never see each other.

use std::sync::Arc;

use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse};

use crate::families::kev::engine::{
    noul_criteria_are_explicit, question_instructions, question_options,
};
use crate::families::kev::model::QuestionRow;
use crate::families::kev::renderer::{render_json, KevRenderer};
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{DrexModel, VerifiedArtifacts};
use super::{
    DrexEngineConfig, DrexError, BACKBONE_ID, CALIBRATION_TEMPERATURE, CPU_BACKEND_ID,
    MAX_ROW_TOKENS, MAX_STATE_TOKENS,
};

/// Loaded pinned Drex engine.
pub struct DrexEngine {
    inner: Inner,
}

/// The family's execution seam: everything one backend contributes to the
/// shared rendering/scoring pipeline. The renderer, temperature, and wire
/// mapping are shared with the CPU oracle; only the pointer-logit readout
/// differs per execution path.
pub(crate) trait DrexPassSource: Send + Sync {
    /// The family renderer over the verified tokenizer.
    fn renderer(&self) -> &KevRenderer;

    /// Pointer logits for one rendered question row.
    fn row_logits(
        &self,
        row: &QuestionRow,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError>;
}

impl DrexPassSource for DrexEngine {
    fn renderer(&self) -> &KevRenderer {
        &self.inner.renderer
    }

    fn row_logits(
        &self,
        row: &QuestionRow,
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        self.inner.model.row_logits(row, control)
    }
}

impl DrexEngine {
    /// Load every pinned artifact offline and build the bounded CPU engine.
    pub fn load(config: DrexEngineConfig) -> Result<BoundedFamilyEngine, DrexError> {
        Self::load_with_execution(config, crate::device::FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend. Only the CPU
    /// oracle exists: other backends fail closed rather than silently
    /// falling back.
    pub fn load_with_execution(
        config: DrexEngineConfig,
        execution: crate::device::FamilyExecution,
    ) -> Result<BoundedFamilyEngine, DrexError> {
        match execution {
            crate::device::FamilyExecution::Cpu => {}
            #[allow(unreachable_patterns)]
            other => {
                return Err(FamilyError::ExecutionUnavailable(format!(
                    "drex-v1.5 has only the CPU BF16-weight FP32-compute path; {} is not \
                     implemented",
                    other.id_fragment()
                ))
                .into());
            }
        }
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer =
            KevRenderer::load_with_limits(&artifacts.tokenizer, MAX_STATE_TOKENS, MAX_ROW_TOKENS)?;
        let model = DrexModel::load(&artifacts)?;
        let engine = Self {
            inner: Inner { renderer, model },
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

struct Inner {
    renderer: KevRenderer,
    model: DrexModel,
}

impl FamilyEvaluator for DrexEngine {
    fn backend_id(&self) -> &str {
        CPU_BACKEND_ID
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} pointer decision engine on a Qwen3.5 9B hybrid \
                 backbone ({CPU_BACKEND_ID}). Nace AI Open RAIL-M weights: inference \
                 only, no distillation or teacher use."
            ),
            release_date: "2026-11-07".into(),
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
        evaluate_with(self, request, control)
    }
}

/// Evaluate one request through any [`DrexPassSource`] backend. The
/// rendering, temperature, and wire mapping are the CPU oracle's; only the
/// pointer-logit forward comes from the source.
pub(crate) fn evaluate_with(
    source: &impl DrexPassSource,
    request: SystemRequest,
    control: &FamilyControl,
) -> Result<SystemResponse, FamilyError> {
    control.check()?;
    let state = match request.state {
        State::Text(text) => text,
        structured => render_json(&crate::families::wire::state_value(structured)),
    };
    if state.trim().is_empty() {
        return Err(FamilyError::InvalidInput(
            "state must contain non-whitespace text".to_owned(),
        ));
    }
    let renderer = source.renderer();
    let state_tokens = renderer.state_tokens(&state)?;
    let questions = crate::families::wire::into_sorted_questions(request.questions)?;

    let mut answers = std::collections::HashMap::<
        String,
        openkind_core::Answer,
        openkind_core::WireHashState,
    >::with_capacity_and_hasher(questions.len(), Default::default());
    let mut input_tokens: u64 = 0;
    for (question_id, question) in questions {
        control.check()?;
        let unpacked = crate::families::wire::unpack_question(&question_id, &question)?;
        let options = question_options(&unpacked, &question, noul_criteria_are_explicit(&question));
        let instruction = render_json(question_instructions(&question));
        let row = renderer.encode_row(&state_tokens, &instruction, &options)?;
        let logits = source.row_logits(&row, control)?;
        if logits.len() != options.len() {
            return Err(FamilyError::ContractMismatch {
                field: "option.count",
                expected: format!("{} option logits", options.len()),
                actual: format!("{} logits", logits.len()),
            });
        }
        input_tokens =
            input_tokens.saturating_add(u64::try_from(row.ids.len()).unwrap_or(u64::MAX));
        let probabilities = temperature_softmax(&logits, CALIBRATION_TEMPERATURE)?;
        let answer = crate::families::wire::answer_from_probabilities(&unpacked, &probabilities)?;
        answers.insert(question_id, answer);
    }
    control.check()?;
    Ok(crate::families::wire::system_response(
        request.model,
        answers,
        input_tokens,
    ))
}
