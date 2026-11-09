//! MLX/Metal execution backend for the pinned `gemma4-decision` profile
//! (feature `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, Winnow renderer, calibration
//! temperature, and wire mapping feed the FP32-dequantized MLX backbone in
//! [`model`]. The candle CPU q8_0 path remains the correctness oracle; the
//! MLX backend executes every weight fully dequantized to FP32 — a
//! different arithmetic identity (the oracle keeps the FFN/PLE/LM-head
//! weights in candle's f16-block-scaled quantized kernels) whose parity
//! gate is the workspace 0.005 probability budget, named by the backend id.
//! Neither embedding table becomes resident on either backend: both stream
//! the rows a prompt or the letter readout references from the verified
//! GGUF.
//!
//! Every MLX operation — model load and per-pass forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

pub(super) mod model;

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{ModelInfo, Question, SystemRequest, SystemResponse};

use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};
use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

use super::backbone::Gemma4TextConfig;
use super::gguf::{gguf_letter_token_ids, VerifiedArtifacts};
use super::renderer::WinnowRenderer;
use super::{Gemma4EngineConfig, Gemma4Error, BACKBONE_ID, PROFILE_ID};

/// Backend id reported by the MLX engine serving this profile. The `-deq`
/// marks FP32-dequantized q8_0 execution (a declared arithmetic difference
/// from the CPU oracle's quantized kernels).
pub const MLX_BACKEND_ID: &str = "gemma4-decision/mlx-fp32-deq";

/// Arithmetic/device identity of the family MLX execution path.
pub use self::model::MLX_EXECUTION_ARITHMETIC_ID;

/// Loaded pinned gemma4-decision engine on the MLX backend.
pub struct Gemma4MlxEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: WinnowRenderer,
    model: model::Gemma4MlxModel,
}

impl Gemma4MlxEngine {
    /// Load every pinned artifact offline and build the bounded engine on
    /// the MLX backend.
    ///
    /// Artifact verification, the pinned GGUF metadata contract, and the
    /// effective-context derivation from the scratch budget are identical
    /// to the CPU engine; only the execution differs. The dequantized FP32
    /// compute weights (~15.9 GB for the pinned profile, see
    /// [`model::estimated_fp32_weight_bytes`]) become resident MLX arrays
    /// in unified memory — larger than the q8_0 artifact, which is the
    /// accepted cost of dequantized execution — while the embedding tables
    /// stay file-backed on both backends.
    pub fn load_mlx(config: Gemma4EngineConfig) -> Result<BoundedFamilyEngine, Gemma4Error> {
        let budget = super::scratch_budget_bytes()?;
        let cfg = Gemma4TextConfig::winnow_e4b();
        let effective_max_tokens = cfg.max_sequence_tokens_for_scratch_budget(budget);
        if effective_max_tokens < super::MIN_VIABLE_SEQUENCE_TOKENS {
            return Err(FamilyError::InvalidInput(format!(
                "scratch budget {budget} bytes caps the effective context at \
                 {effective_max_tokens} tokens, below the {}-token viability floor; \
                 raise {} or lower the frozen maximum",
                super::MIN_VIABLE_SEQUENCE_TOKENS,
                super::SCRATCH_BUDGET_ENV
            ))
            .into());
        }
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let gguf_letters = gguf_letter_token_ids(&artifacts, super::renderer::LETTER_MAX_OPTIONS)?;
        let renderer = WinnowRenderer::load(
            &artifacts.tokenizer,
            effective_max_tokens,
            Some(&gguf_letters),
        )?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        // The GGUF dequantization and every array construction run under
        // the process-wide MLX execution lock; the lock is not reentrant,
        // so this load must not already hold it. Forwards take the lock
        // themselves in `model.rs`.
        let model = runtime
            .execute(|| model::Gemma4MlxModel::load(&artifacts, &cfg, Some(budget), &runtime))??;
        let engine = Self {
            inner: Arc::new(Inner { renderer, model }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

/// Raw `instructions` payload of a wire question, rendered by the Winnow
/// protocol as JSON (a string instruction serializes with quotes) — the
/// same mapping the CPU engine applies.
fn instruction_value(question: &Question) -> &serde_json::Value {
    match question {
        Question::Noul(question) => &question.instructions,
        Question::Choice(question) => &question.instructions,
        Question::Score(question) => &question.instructions,
    }
}

impl FamilyEvaluator for Gemma4MlxEngine {
    fn backend_id(&self) -> &str {
        MLX_BACKEND_ID
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} Gemma 4 letter-logit decision engine ({PROFILE_ID}, \
                 GGUF q8_0 dequantized to FP32, ~{:.1} GiB resident weights, MLX).",
                model::estimated_fp32_weight_bytes(&Gemma4TextConfig::winnow_e4b()) as f64
                    / (1u64 << 30) as f64
            ),
            release_date: "2026-10-01".into(),
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
        let state = crate::families::wire::state_value(request.state);
        let questions = crate::families::wire::into_sorted_questions(request.questions)?;

        let mut answers: HashMap<String, openkind_core::Answer, openkind_core::WireHashState> =
            HashMap::with_capacity_and_hasher(questions.len(), Default::default());
        let mut input_tokens: u64 = 0;
        for (question_id, question) in questions {
            control.check()?;
            let id = &question_id;
            let question = &question;
            let unpacked = crate::families::wire::unpack_question(id, question)?;
            let rendered = self.inner.renderer.render(
                &state,
                instruction_value(question),
                &unpacked.labels,
                &unpacked.criteria,
                unpacked.ordered,
            )?;
            input_tokens = input_tokens
                .saturating_add(u64::from(self.inner.renderer.prompt_token_count(&rendered)));
            let prompt_ids = rendered.prompt_ids();
            // The renderer already rejects prompts over the budget-derived
            // effective maximum; the model re-gates the forward against the
            // scratch estimate and observes the control on both sides of
            // (and between the layers of) the pass.
            let logits = self.inner.model.letter_logits_controlled(
                &prompt_ids,
                rendered.letter_ids(),
                control,
            )?;
            let probabilities = temperature_softmax(&logits, super::CALIBRATION_TEMPERATURE)?;
            let answer =
                crate::families::wire::answer_from_probabilities(&unpacked, &probabilities)?;
            answers.insert(question_id, answer);
        }
        control.check()?;
        Ok(crate::families::wire::system_response(
            request.model,
            answers,
            input_tokens,
        ))
    }
}
