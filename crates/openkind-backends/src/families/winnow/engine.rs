//! Composite `DecisionEngine` for the pinned winnow profile.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use candle_transformers::models::qwen2::Config;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};
use openkind_engine::{DecisionEngine, EngineError, EngineResult};

use crate::families::support::{temperature_softmax, FamilyError};
use tokio::sync::Semaphore;

use super::model::{VerifiedArtifacts, WinnowModel};
use super::renderer::WinnowRenderer;
use super::{WinnowEngineConfig, WinnowError, BACKBONE_ID, CALIBRATION_TEMPERATURE, PROFILE_ID};

/// Loaded pinned winnow engine: routes requests to registered siblings by a
/// learned letter-logit decision.
pub struct WinnowEngine {
    router: Arc<WinnowRouter>,
    /// Sibling alias per routing label, in label order (`A`, `B`).
    siblings: Vec<(String, Arc<dyn DecisionEngine>)>,
    execution_slots: Arc<Semaphore>,
    admission_slots: Arc<Semaphore>,
    retry_after_ms: u64,
    evaluation_timeout: Option<Duration>,
    backend_id: String,
}

struct WinnowRouter {
    model: WinnowModel,
    renderer: WinnowRenderer,
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
        let backend_id = match execution {
            crate::device::FamilyExecution::Cpu => "winnow/cpu-fp32-lora".to_owned(),
            #[cfg(feature = "cuda")]
            crate::device::FamilyExecution::Cuda { .. } => {
                let device = execution.candle_device()?;
                let model =
                    WinnowModel::load_with_device(&artifacts, &Self::pinned_config(), device)?;
                let concurrent = config.limits.max_concurrent_requests.max(1);
                let admitted = concurrent.saturating_add(config.limits.max_queued_requests);
                return Ok(Self {
                    router: Arc::new(WinnowRouter { model, renderer }),
                    siblings,
                    execution_slots: Arc::new(Semaphore::new(concurrent)),
                    admission_slots: Arc::new(Semaphore::new(admitted)),
                    retry_after_ms: config.limits.retry_after_ms,
                    evaluation_timeout: config.limits.evaluation_timeout,
                    backend_id: "winnow/cuda-fp32-lora".to_owned(),
                });
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
        };
        let model = WinnowModel::load(&artifacts, &Self::pinned_config())?;
        let concurrent = config.limits.max_concurrent_requests.max(1);
        let admitted = concurrent.saturating_add(config.limits.max_queued_requests);
        Ok(Self {
            router: Arc::new(WinnowRouter { model, renderer }),
            siblings,
            execution_slots: Arc::new(Semaphore::new(concurrent)),
            admission_slots: Arc::new(Semaphore::new(admitted)),
            retry_after_ms: config.limits.retry_after_ms,
            evaluation_timeout: config.limits.evaluation_timeout,
            backend_id,
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
        let logits = self
            .router
            .model
            .letter_logits(&prompt_ids, self.router.renderer.letter_ids())?;
        temperature_softmax(&logits, CALIBRATION_TEMPERATURE)
    }
}

impl WinnowRouter {
    fn route(
        &self,
        state: &str,
        siblings: &[(String, Arc<dyn DecisionEngine>)],
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
        let backend = self.backend_id().to_owned();
        let started = Instant::now();
        let deadline =
            match self.evaluation_timeout {
                Some(timeout) => Some(started.checked_add(timeout).ok_or_else(|| {
                    EngineError::Unsupported {
                        backend: backend.clone(),
                        message: "winnow evaluation timeout is outside the monotonic clock range"
                            .to_owned(),
                    }
                })?),
                None => None,
            };
        let admission = self
            .admission_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| EngineError::Overloaded {
                backend: backend.clone(),
                retry_after_ms: self.retry_after_ms,
            })?;
        let execution = acquire_execution_slot(
            self.execution_slots.clone(),
            deadline,
            &backend,
            self.evaluation_timeout,
        )
        .await?;

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
        let router = self.router.clone();
        let siblings = self.siblings.clone();
        let routed = tokio::task::spawn_blocking(move || {
            let _execution = execution;
            let _admission = admission;
            if let Some(expired) = deadline.filter(|deadline| Instant::now() >= *deadline) {
                return Err(deadline_error(&backend, expired, started));
            }
            let routed = router.route(&state, &siblings)?;
            if let Some(expired) = deadline.filter(|deadline| Instant::now() >= *deadline) {
                return Err(deadline_error(&backend, expired, started));
            }
            Ok(routed)
        })
        .await
        .map_err(|error| EngineError::Backend {
            backend: self.backend_id().to_owned(),
            message: format!("winnow routing task failed: {error}"),
        })??;
        let RouteDecision {
            engine: sibling,
            alias,
            probabilities,
        } = routed;
        metrics::histogram!("openkind_winnow_route_seconds")
            .record(routed_started.elapsed().as_secs_f64());
        metrics::counter!("openkind_winnow_routes_total", "sibling" => alias.clone()).increment(1);
        metrics::gauge!("openkind_winnow_route_top_probability")
            .set(probabilities.iter().cloned().fold(0.0_f64, f64::max));
        if let Some(deadline) = deadline {
            tokio::time::timeout_at(deadline.into(), sibling.evaluate(request))
                .await
                .map_err(|_| deadline_error(self.backend_id(), deadline, started))?
        } else {
            sibling.evaluate(request).await
        }
    }
}

async fn acquire_execution_slot(
    slots: Arc<Semaphore>,
    deadline: Option<Instant>,
    backend: &str,
    timeout: Option<Duration>,
) -> EngineResult<tokio::sync::OwnedSemaphorePermit> {
    let acquire = slots.acquire_owned();
    if let Some(deadline) = deadline {
        match tokio::time::timeout_at(deadline.into(), acquire).await {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) => Err(EngineError::Backend {
                backend: backend.to_owned(),
                message: "winnow execution admission closed during shutdown".to_owned(),
            }),
            Err(_) => Err(EngineError::DeadlineExceeded {
                backend: backend.to_owned(),
                timeout_ms: timeout
                    .map(|timeout| timeout.as_millis().min(u128::from(u64::MAX)) as u64)
                    .unwrap_or_default(),
            }),
        }
    } else {
        acquire.await.map_err(|_| EngineError::Backend {
            backend: backend.to_owned(),
            message: "winnow execution admission closed during shutdown".to_owned(),
        })
    }
}

fn deadline_error(backend: &str, deadline: Instant, started: Instant) -> EngineError {
    EngineError::DeadlineExceeded {
        backend: backend.to_owned(),
        timeout_ms: deadline
            .saturating_duration_since(started)
            .as_millis()
            .min(u128::from(u64::MAX)) as u64,
    }
}
