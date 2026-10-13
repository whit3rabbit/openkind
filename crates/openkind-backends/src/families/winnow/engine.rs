//! Composite `DecisionEngine` for the pinned winnow profile.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use candle_transformers::models::qwen2::Config;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};
use openkind_engine::{DecisionEngine, EngineError, EngineResult};

use crate::families::support::{
    error_outcome_label, map_family_error, outcome_label, saturating_millis, temperature_softmax,
    AdmissionGate, FamilyControl, FamilyError, FAMILY_GATE_METRICS,
};

use super::model::{VerifiedArtifacts, WinnowModel};
use super::renderer::WinnowRenderer;
use super::{WinnowEngineConfig, WinnowError, BACKBONE_ID, CALIBRATION_TEMPERATURE, PROFILE_ID};

/// Loaded pinned winnow engine: routes requests to registered siblings by a
/// learned letter-logit decision.
pub struct WinnowEngine {
    router: Arc<WinnowRouter>,
    /// Sibling alias per routing label, in label order (`A`, `B`).
    siblings: Vec<(String, Arc<dyn DecisionEngine>)>,
    gate: AdmissionGate,
    evaluation_timeout: Option<Duration>,
    backend_id: String,
}

struct WinnowRouter {
    model: WinnowModel,
    renderer: WinnowRenderer,
}

/// Backend identity of the router on the selected execution backend.
fn backend_id_for(execution: crate::device::FamilyExecution) -> String {
    format!("winnow/{}", execution.id_fragment())
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
        Self::load_with_execution(config, siblings, crate::device::FamilyExecution::Cpu)
    }

    /// Load the routing engine on the selected execution backend.
    ///
    /// The winnow router is a candle decoder plus engine-level routing
    /// logic; it has no ONNX export. Accelerated loads require the `cuda`
    /// feature.
    pub fn load_with_execution(
        config: WinnowEngineConfig,
        siblings: Vec<(String, Arc<dyn DecisionEngine>)>,
        execution: crate::device::FamilyExecution,
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
        let model = match execution {
            crate::device::FamilyExecution::Cpu => {
                WinnowModel::load(&artifacts, &Self::pinned_config())?
            }
            #[cfg(feature = "cuda")]
            crate::device::FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device().map_err(FamilyError::from)?;
                WinnowModel::load_with_device(&artifacts, &Self::pinned_config(), device)?
            }
            #[cfg(feature = "onnx")]
            crate::device::FamilyExecution::Onnx { .. } => {
                return Err(FamilyError::ExecutionUnavailable(
                    "the winnow router decoder has no ONNX export; select cpu or (with the \
                     `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
            #[cfg(feature = "onnx-rocm")]
            crate::device::FamilyExecution::OnnxRocm { .. } => {
                return Err(FamilyError::ExecutionUnavailable(
                    "the winnow router decoder has no ONNX export and therefore no ROCm \
                     execution; select cpu or (with the `cuda` feature) cuda"
                        .to_owned(),
                )
                .into());
            }
        };
        Ok(Self {
            router: Arc::new(WinnowRouter { model, renderer }),
            siblings,
            gate: AdmissionGate::new(
                config.limits.max_concurrent_requests,
                config.limits.max_queued_requests,
                config.limits.retry_after_ms,
                FAMILY_GATE_METRICS,
            ),
            evaluation_timeout: config.limits.evaluation_timeout,
            backend_id: backend_id_for(execution),
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
        let prompt_ids = self.router.renderer.render(state)?;
        // Offline probes have no caller to disconnect and no deadline.
        let control = FamilyControl::new(Arc::new(AtomicBool::new(false)), None, 0);
        let logits = self.router.model.letter_logits(
            &prompt_ids,
            self.router.renderer.letter_ids(),
            &control,
        )?;
        temperature_softmax(&logits, CALIBRATION_TEMPERATURE)
    }
}

impl WinnowRouter {
    fn route(
        &self,
        state: &str,
        siblings: &[(String, Arc<dyn DecisionEngine>)],
        control: &FamilyControl,
    ) -> Result<RouteDecision, EngineError> {
        let prompt_ids = self
            .renderer
            .render(state)
            .map_err(|error| EngineError::Unsupported {
                backend: super::FAMILY_SLUG.to_owned(),
                message: error.to_string(),
            })?;
        let logits = self
            .model
            .letter_logits(&prompt_ids, self.renderer.letter_ids(), control)
            .map_err(|error| match error {
                // A control failure keeps its class instead of being folded
                // into a generic backend error.
                FamilyError::DeadlineExceeded { timeout_ms } => EngineError::DeadlineExceeded {
                    backend: super::FAMILY_SLUG.to_owned(),
                    timeout_ms,
                },
                error => EngineError::Backend {
                    backend: super::FAMILY_SLUG.to_owned(),
                    message: error.to_string(),
                },
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
        let (alias, engine) = siblings.get(argmax).ok_or_else(|| EngineError::Backend {
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
        &self.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: format!(
                "Pinned {BACKBONE_ID} LoRA-learned router ({PROFILE_ID}, {}).",
                self.backend_id
            ),
            release_date: "2026-09-26".into(),
        }
    }

    async fn evaluate(&self, request: SystemRequest) -> EngineResult<SystemResponse> {
        let started = Instant::now();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut cancellation = self.gate.cancellation_guard(cancelled.clone());
        let result = self.evaluate_controlled(request, started, cancelled).await;
        cancellation.disarm();
        result
    }
}

impl WinnowEngine {
    async fn evaluate_controlled(
        &self,
        request: SystemRequest,
        started: Instant,
        cancelled: Arc<AtomicBool>,
    ) -> EngineResult<SystemResponse> {
        let backend = self.backend_id().to_owned();
        let deadline = match self.evaluation_timeout {
            Some(timeout) => match started.checked_add(timeout) {
                Some(deadline) => Some(deadline),
                None => {
                    // An unrepresentable configured deadline must not turn
                    // into an unbounded evaluation.
                    self.gate.record(started, "invalid");
                    return Err(EngineError::Unsupported {
                        backend,
                        message: "winnow evaluation timeout is outside the monotonic clock range"
                            .to_owned(),
                    });
                }
            },
            None => None,
        };
        let timeout_ms = self
            .evaluation_timeout
            .map(saturating_millis)
            .unwrap_or_default();

        let state = match state_text(&request.state) {
            Ok(state) => state,
            Err(error) => {
                self.gate.record(started, "invalid");
                return Err(error);
            }
        };
        let routed_started = Instant::now();
        let router = self.router.clone();
        let siblings = self.siblings.clone();
        let control = FamilyControl::new(cancelled, deadline, timeout_ms);
        let routing_backend = backend.clone();
        let routed = self
            .gate
            .run(&backend, started, deadline, timeout_ms, move || {
                // The forward pass is one uninterruptible model call, so
                // cancellation and the deadline are observed on either side
                // of it (and again before the work starts).
                control
                    .check()
                    .map_err(|error| map_family_error(&routing_backend, error))?;
                let decision = router.route(&state, &siblings, &control)?;
                control
                    .check()
                    .map_err(|error| map_family_error(&routing_backend, error))?;
                Ok(decision)
            })
            .await?;
        let RouteDecision {
            engine: sibling,
            alias,
            probabilities,
        } = match routed {
            Ok(decision) => decision,
            Err(error) => {
                self.gate.record(started, error_outcome_label(&error));
                return Err(error);
            }
        };
        metrics::histogram!("openkind_winnow_route_seconds")
            .record(routed_started.elapsed().as_secs_f64());
        metrics::counter!("openkind_winnow_routes_total", "sibling" => alias.clone()).increment(1);
        metrics::gauge!("openkind_winnow_route_top_probability")
            .set(probabilities.iter().cloned().fold(0.0_f64, f64::max));
        let result = if let Some(deadline) = deadline {
            match tokio::time::timeout_at(deadline.into(), sibling.evaluate(request)).await {
                Ok(result) => result,
                Err(_) => Err(EngineError::DeadlineExceeded {
                    backend,
                    timeout_ms,
                }),
            }
        } else {
            sibling.evaluate(request).await
        };
        self.gate.record(started, outcome_label(&result));
        result
    }
}

/// The request state as the compact JSON text the router prompt embeds.
fn state_text(state: &openkind_core::State) -> Result<String, EngineError> {
    match state {
        openkind_core::State::Text(text) => Ok(text.clone()),
        openkind_core::State::Object(map) => {
            serde_json::to_string(map).map_err(|error| EngineError::Unsupported {
                backend: super::FAMILY_SLUG.to_owned(),
                message: format!("state serialization failed: {error}"),
            })
        }
        openkind_core::State::Array(items) => {
            serde_json::to_string(items).map_err(|error| EngineError::Unsupported {
                backend: super::FAMILY_SLUG.to_owned(),
                message: format!("state serialization failed: {error}"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id_follows_the_execution_fragment() {
        assert_eq!(
            backend_id_for(crate::device::FamilyExecution::Cpu),
            "winnow/cpu-fp32"
        );
    }

    #[test]
    fn state_text_passes_text_through_and_serializes_structures_compactly() {
        use openkind_core::State;
        assert_eq!(state_text(&State::Text("plain".into())).unwrap(), "plain");
        let object: State =
            serde_json::from_value(serde_json::json!({"b": 1, "a": [true]})).expect("object state");
        assert_eq!(state_text(&object).unwrap(), "{\"a\":[true],\"b\":1}");
        let array: State =
            serde_json::from_value(serde_json::json!([1, "x"])).expect("array state");
        assert_eq!(state_text(&array).unwrap(), "[1,\"x\"]");
    }
}
