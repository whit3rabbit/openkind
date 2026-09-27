//! Composite `DecisionEngine` for the pinned winnow profile.

use std::sync::Arc;

use async_trait::async_trait;
use candle_transformers::models::qwen2::Config;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};
use openkind_engine::{DecisionEngine, EngineError, EngineResult};

use crate::families::support::{temperature_softmax, FamilyError};

use super::model::{VerifiedArtifacts, WinnowModel};
use super::renderer::WinnowRenderer;
use super::{WinnowEngineConfig, WinnowError, BACKBONE_ID, CALIBRATION_TEMPERATURE, PROFILE_ID};

/// Loaded pinned winnow engine: routes requests to registered siblings by a
/// learned letter-logit decision.
pub struct WinnowEngine {
    model: WinnowModel,
    renderer: WinnowRenderer,
    /// Sibling alias per routing label, in label order (`A`, `B`).
    siblings: Vec<(String, Arc<dyn DecisionEngine>)>,
}

impl WinnowEngine {
    /// Load every pinned artifact offline and bind the routing siblings.
    ///
    /// `siblings` holds one `(alias, engine)` handle per routing label, in
    /// label order (`A`, `B`); the daemon resolves the labels against the
    /// served aliases before calling this constructor.
    pub fn load(
        config: WinnowEngineConfig,
        siblings: Vec<(String, Arc<dyn DecisionEngine>)>,
    ) -> Result<Self, WinnowError> {
        if siblings.len() != 2 {
            return Err(FamilyError::InvalidInput(format!(
                "winnow routes exactly two labels (A, B); got {} siblings",
                siblings.len()
            ))
            .into());
        }
        let artifacts = VerifiedArtifacts::verify(
            &config.model_root,
            &config.adapter_path,
            crate::families::decoder_logit_letter::CHECKPOINT_SHA256,
        )?;
        let renderer = WinnowRenderer::load(&artifacts.tokenizer)?;
        let model = WinnowModel::load(&artifacts, &Self::pinned_config())?;
        Ok(Self {
            model,
            renderer,
            siblings,
        })
    }

    fn pinned_config() -> Config {
        serde_json::from_str(
            "{
                \"vocab_size\": 151936, \"hidden_size\": 896, \"intermediate_size\": 4864,
                \"num_hidden_layers\": 24, \"num_attention_heads\": 14,
                \"num_key_value_heads\": 2, \"max_position_embeddings\": 32768,
                \"sliding_window\": 32768, \"max_window_layers\": 21,
                \"tie_word_embeddings\": true, \"rope_theta\": 1000000.0,
                \"rms_norm_eps\": 1e-06, \"use_sliding_window\": false,
                \"hidden_act\": \"silu\"
            }",
        )
        .expect("pinned winnow base config is valid JSON")
    }

    /// Routing probabilities for pinned fixture generation and parity
    /// tests. Not part of the wire contract.
    #[doc(hidden)]
    pub fn debug_route_probabilities(&self, state: &str) -> Result<Vec<f64>, FamilyError> {
        let prompt_ids = self.renderer.render(state)?;
        let logits = self
            .model
            .letter_logits(&prompt_ids, self.renderer.letter_ids())?;
        temperature_softmax(&logits, CALIBRATION_TEMPERATURE)
    }

    /// Route one request's state text to a sibling handle.
    fn route(&self, state: &str) -> Result<RouteDecision, EngineError> {
        let prompt_ids = self
            .renderer
            .render(state)
            .map_err(|error| EngineError::Unsupported {
                backend: super::FAMILY_SLUG.to_owned(),
                message: error.to_string(),
            })?;
        let logits = self
            .model
            .letter_logits(&prompt_ids, self.renderer.letter_ids())
            .map_err(|error| EngineError::Backend {
                backend: super::FAMILY_SLUG.to_owned(),
                message: error.to_string(),
            })?;
        let probabilities =
            temperature_softmax(&logits, CALIBRATION_TEMPERATURE).map_err(|error| {
                EngineError::Backend {
                    backend: super::FAMILY_SLUG.to_owned(),
                    message: error.to_string(),
                }
            })?;
        let mut argmax = 0;
        for (index, probability) in probabilities.iter().enumerate() {
            if *probability > probabilities[argmax] {
                argmax = index;
            }
        }
        let (alias, engine) = self
            .siblings
            .get(argmax)
            .ok_or_else(|| EngineError::Backend {
                backend: super::FAMILY_SLUG.to_owned(),
                message: "routing label has no sibling bound".to_owned(),
            })?;
        Ok(RouteDecision {
            engine: Arc::clone(engine),
            alias: alias.clone(),
            probabilities,
        })
    }
}

/// One routing decision: the sibling handle, its alias, and the letter
/// distribution over the routing labels.
struct RouteDecision {
    engine: Arc<dyn DecisionEngine>,
    alias: String,
    probabilities: Vec<f64>,
}

#[async_trait]
impl DecisionEngine for WinnowEngine {
    fn backend_id(&self) -> &str {
        "winnow/cpu-fp32-lora"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} LoRA-learned router ({PROFILE_ID}, FP32 CPU)."
            ),
            release_date: "2026-09-26".into(),
        }
    }

    async fn evaluate(&self, request: SystemRequest) -> EngineResult<SystemResponse> {
        let state = match &request.state {
            openkind_core::State::Text(text) => text.clone(),
            openkind_core::State::Object(map) => {
                serde_json::to_string(map).map_err(|error| EngineError::Unsupported {
                    backend: super::FAMILY_SLUG.to_owned(),
                    message: format!("state serialization failed: {error}"),
                })?
            }
            openkind_core::State::Array(items) => {
                serde_json::to_string(items).map_err(|error| EngineError::Unsupported {
                    backend: super::FAMILY_SLUG.to_owned(),
                    message: format!("state serialization failed: {error}"),
                })?
            }
        };
        let routed_started = std::time::Instant::now();
        let RouteDecision {
            engine: sibling,
            alias,
            probabilities,
        } = self.route(&state)?;
        metrics::histogram!("openkind_winnow_route_seconds")
            .record(routed_started.elapsed().as_secs_f64());
        metrics::counter!("openkind_winnow_routes_total", "sibling" => alias.clone()).increment(1);
        metrics::gauge!("openkind_winnow_route_top_probability")
            .set(probabilities.iter().cloned().fold(0.0_f64, f64::max));
        sibling.evaluate(request).await
    }
}
