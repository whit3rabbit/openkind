//! `DecisionEngine` adapter for the pinned schema-scorer profile.

use std::sync::Arc;

use candle_core::Device;
use openkind_core::{ModelInfo, State, SystemRequest, SystemResponse, Usage};

use crate::device::FamilyExecution;
use crate::families::support::{
    temperature_softmax, BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator,
};

use super::model::{SchemaScorerModel, VerifiedArtifacts};
use super::renderer::SchemaScorerRenderer;
use super::{
    SchemaScorerEngineConfig, SchemaScorerError, BACKBONE_ID, MAX_CANDIDATES, MIN_CANDIDATES,
    PROFILE_ID,
};

/// Scalar relevance-logit source shared by the candle and ONNX models.
trait RelevanceLogits: Send + Sync {
    /// Scalar relevance logit of one query–passage pair.
    fn relevance_logit(&self, pair_ids: &[u32]) -> Result<f64, FamilyError>;
}

impl RelevanceLogits for SchemaScorerModel {
    fn relevance_logit(&self, pair_ids: &[u32]) -> Result<f64, FamilyError> {
        SchemaScorerModel::relevance_logit(self, pair_ids)
    }
}

/// ONNX model readout bridging the shared logit source (feature `onnx`).
#[cfg(feature = "onnx")]
impl RelevanceLogits for super::onnx::SchemaScorerOnnxModel {
    fn relevance_logit(&self, pair_ids: &[u32]) -> Result<f64, FamilyError> {
        super::onnx::SchemaScorerOnnxModel::relevance_logit(self, pair_ids)
    }
}

/// Loaded pinned schema-scorer engine.
pub struct SchemaScorerEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: SchemaScorerRenderer,
    model: Box<dyn RelevanceLogits>,
    backend_id: String,
}

impl SchemaScorerEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    ///
    /// The reference execution is FP32 CPU; see [`Self::load_with_execution`]
    /// for accelerated backends.
    pub fn load(
        config: SchemaScorerEngineConfig,
    ) -> Result<BoundedFamilyEngine, SchemaScorerError> {
        Self::load_with_execution(config, FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// Artifact verification (digests, pinned config) is identical on every
    /// backend; only the readout execution differs, and ONNX additionally
    /// requires `model.onnx` and its digest manifest in the model root.
    /// Loads fail closed when a backend is unavailable.
    pub fn load_with_execution(
        config: SchemaScorerEngineConfig,
        execution: FamilyExecution,
    ) -> Result<BoundedFamilyEngine, SchemaScorerError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = SchemaScorerRenderer::load(&artifacts.tokenizer)?;
        let model: Box<dyn RelevanceLogits> = match execution {
            FamilyExecution::Cpu => Box::new(SchemaScorerModel::load(&artifacts, Device::Cpu)?),
            #[cfg(feature = "cuda")]
            FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device()?;
                Box::new(SchemaScorerModel::load(&artifacts, device)?)
            }
            #[cfg(feature = "onnx")]
            FamilyExecution::Onnx { device_id } => {
                let acceleration = crate::onnx::OnnxAcceleration::from_onnx_execution(device_id)
                    .map_err(FamilyError::from)?;
                Box::new(super::onnx::SchemaScorerOnnxModel::load(
                    &config.model_root,
                    acceleration,
                )?)
            }
            #[cfg(feature = "onnx-rocm")]
            FamilyExecution::OnnxRocm { device_id } => {
                Box::new(super::onnx::SchemaScorerOnnxModel::load(
                    &config.model_root,
                    crate::onnx::OnnxAcceleration::Rocm { device_id },
                )?)
            }
        };
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                backend_id: format!("schema-scorer/{}", execution.id_fragment()),
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for SchemaScorerEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} scalar cross-encoder schema scorer ({PROFILE_ID}, {}).",
                self.inner.backend_id
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
            State::Object(map) => serde_json::to_string_pretty(map).map_err(|error| {
                FamilyError::InvalidInput(format!("state serialization failed: {error}"))
            })?,
            State::Array(items) => serde_json::to_string_pretty(items).map_err(|error| {
                FamilyError::InvalidInput(format!("state serialization failed: {error}"))
            })?,
        };
        if state.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "state must contain non-whitespace text".to_owned(),
            ));
        }
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
            let instruction = crate::families::wire::instruction_text(question)?;
            let unpacked = crate::families::wire::unpack_question(id, question)?;
            if unpacked.criteria.len() > MAX_CANDIDATES {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` offers {} candidates but the profile accepts at most {MAX_CANDIDATES}",
                    unpacked.criteria.len()
                )));
            }
            if unpacked.criteria.len() < MIN_CANDIDATES {
                return Err(FamilyError::InvalidInput(format!(
                    "question `{id}` needs at least {MIN_CANDIDATES} candidates"
                )));
            }
            // The Jev question schema renders into the query side: the state
            // document plus the question instruction. The candidate criterion
            // is the passage side.
            let query = format!("{state}\n\nQuestion: {instruction}");
            let mut logits = Vec::with_capacity(unpacked.criteria.len());
            for criterion in &unpacked.criteria {
                control.check()?;
                let pair_ids = self.inner.renderer.encode_pair(&query, criterion)?;
                let logit = self.inner.model.relevance_logit(&pair_ids)?;
                input_tokens = input_tokens.saturating_add(pair_ids.len() as u64);
                logits.push(logit);
            }
            let probabilities = temperature_softmax(&logits, super::CALIBRATION_TEMPERATURE)?;
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
