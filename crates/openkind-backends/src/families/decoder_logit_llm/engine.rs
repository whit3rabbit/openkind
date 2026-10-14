//! `DecisionEngine` adapter for the pinned decoder-logit-llm (GGUF) profile.

use std::sync::Arc;

use openkind_core::{ModelInfo, SystemRequest, SystemResponse};

use crate::families::letter_renderer::{evaluate_letter_request, LetterRenderer};
use crate::families::support::{BoundedFamilyEngine, FamilyControl, FamilyError, FamilyEvaluator};

use super::model::{DecoderLlmModel, VerifiedArtifacts};
use super::{DecoderLlmEngineConfig, DecoderLlmError, BACKBONE_ID, GGUF_VARIANT, PROFILE_ID};

/// Loaded pinned decoder-logit-llm engine.
pub struct DecoderLlmEngine {
    inner: Arc<Inner>,
}

struct Inner {
    renderer: LetterRenderer,
    model: DecoderLlmModel,
    backend_id: String,
}

impl DecoderLlmEngine {
    /// Load every pinned artifact offline and build the bounded engine.
    pub fn load(config: DecoderLlmEngineConfig) -> Result<BoundedFamilyEngine, DecoderLlmError> {
        Self::load_with_execution(config, crate::device::FamilyExecution::Cpu)
    }

    /// Load the engine on the selected execution backend.
    ///
    /// The quantized GGUF readout has no ONNX counterpart; accelerated
    /// loads require the `cuda` feature.
    pub fn load_with_execution(
        config: DecoderLlmEngineConfig,
        execution: crate::device::FamilyExecution,
    ) -> Result<BoundedFamilyEngine, DecoderLlmError> {
        let artifacts = VerifiedArtifacts::verify(&config.model_root)?;
        let renderer = LetterRenderer::load(&artifacts.tokenizer, super::MAX_SEQUENCE_TOKENS)?;
        let backend_id = match execution {
            crate::device::FamilyExecution::Cpu => "decoder-logit-llm/cpu-q8_0".to_owned(),
            #[cfg(feature = "cuda")]
            crate::device::FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device().map_err(FamilyError::from)?;
                let model = DecoderLlmModel::load_with_device(&artifacts, device)?;
                let engine = Self {
                    inner: Arc::new(Inner {
                        renderer,
                        model,
                        backend_id: "decoder-logit-llm/cuda-q8_0".to_owned(),
                    }),
                };
                return Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits));
            }
            #[cfg(feature = "onnx")]
            crate::device::FamilyExecution::Onnx { .. } => {
                return Err(FamilyError::ExecutionUnavailable(
                    "the quantized GGUF readout has no ONNX export; select cpu or (with the \
                     `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
            #[cfg(feature = "onnx-rocm")]
            crate::device::FamilyExecution::OnnxRocm { .. } => {
                return Err(FamilyError::ExecutionUnavailable(
                    "the quantized GGUF readout has no ONNX export and therefore no ROCm \
                     execution; select cpu or (with the `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
        };
        let model = DecoderLlmModel::load(&artifacts)?;
        let engine = Self {
            inner: Arc::new(Inner {
                renderer,
                model,
                backend_id,
            }),
        };
        Ok(BoundedFamilyEngine::new(Arc::new(engine), config.limits))
    }
}

impl FamilyEvaluator for DecoderLlmEngine {
    fn backend_id(&self) -> &str {
        &self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} ({GGUF_VARIANT}) letter-logit decision engine ({PROFILE_ID})."
            ),
            release_date: "2026-09-26".into(),
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
        evaluate_letter_request(
            &self.inner.renderer,
            &|prompt_ids: &[u32], letter_ids: &[u32]| {
                self.inner.model.letter_logits(prompt_ids, letter_ids)
            },
            super::CALIBRATION_TEMPERATURE,
            request,
            control,
        )
    }
}
