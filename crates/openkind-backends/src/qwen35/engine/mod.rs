//! Direct Jev wire adapter for the pinned native Qwen3.5 profile.

mod backbone;
mod canonical;
mod eval;
mod mapping;

pub(super) use canonical::state_text;
pub(super) use mapping::{instruction_text, question_candidates};

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};
use openkind_engine::{DecisionEngine, EngineError, EngineResult, ProbabilitySpace};
use tokio::sync::Semaphore;

pub use self::backbone::Qwen35Backend;
use self::backbone::{load_backbone, EngineBackbone};
use self::eval::{evaluate_request, map_evaluation_error};
use super::{
    ExecutionControl, Qwen35Error, Qwen35Tokenizer, ReferenceBundle, SchedulerConfig,
    ScoreSummaryHead,
};

/// Reserved Choice criteria key that explicitly exposes the model's semantic-none mass.
///
/// Native Choice requests must include this key with a non-empty description. The adapter
/// never appends an unrequested option and never discards or renormalizes none probability.
pub const SEMANTIC_NONE_OPTION: &str = "__none__";

/// Filesystem and admission settings for the direct native engine.
#[derive(Debug, Clone)]
pub struct Qwen35EngineConfig {
    /// Selected-profile bundle directory containing the fitted head.
    pub bundle_root: PathBuf,
    /// Pinned Qwen checkpoint directory containing both safetensors shards.
    pub checkpoint_root: PathBuf,
    /// Digest-locked exported tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Execution backend executing tokens for this engine.
    pub backend: Qwen35Backend,
    /// Adaptive execution policy and memory admission settings.
    pub scheduler: SchedulerConfig,
    /// Maximum model evaluations executing concurrently.
    pub max_concurrent_requests: usize,
    /// Maximum additional requests allowed to wait for execution.
    pub max_queued_requests: usize,
    /// Suggested backoff returned when admission is full.
    pub retry_after_ms: u64,
    /// Queue-inclusive request deadline. `None` leaves model evaluation unbounded.
    pub evaluation_timeout: Option<Duration>,
}

pub(super) struct EngineInner {
    pub(super) tokenizer: Qwen35Tokenizer,
    pub(super) backbone: EngineBackbone,
    pub(super) head: ScoreSummaryHead,
    pub(super) scheduler: SchedulerConfig,
    pub(super) max_concurrent_requests: usize,
    pub(super) backend_id: &'static str,
    pub(super) evaluation_timeout: Option<Duration>,
}

/// Native CPU implementation registered directly behind [`DecisionEngine`].
pub struct Qwen35DecisionEngine {
    inner: Arc<EngineInner>,
    execution_slots: Arc<Semaphore>,
    admission_slots: Arc<Semaphore>,
    retry_after_ms: u64,
}

impl Qwen35DecisionEngine {
    /// Load every pinned artifact offline and construct the bounded native engine.
    pub fn load(config: Qwen35EngineConfig) -> Result<Self, Qwen35Error> {
        if config
            .evaluation_timeout
            .is_some_and(|timeout| Instant::now().checked_add(timeout).is_none())
        {
            return Err(Qwen35Error::InvalidInput(
                "native evaluation timeout is outside the monotonic clock range".into(),
            ));
        }
        let tokenizer = Qwen35Tokenizer::from_file(&config.tokenizer_path)?;
        let mut scheduler = config.scheduler;
        let backbone = load_backbone(config.backend, &config.checkpoint_root, &mut scheduler)?;
        let bundle = ReferenceBundle::load(&config.bundle_root)?;
        // The adapter branches on the declared probability space: this native
        // adapter implements exactly one, and a profile declaring any other
        // space fails explicitly at load instead of silently discarding
        // semantic-none mass or renormalizing the remainder.
        match bundle.profile().execution().probability_space() {
            ProbabilitySpace::OfferedOptionsPlusSemanticNone => {}
            declared => {
                return Err(Qwen35Error::ContractMismatch {
                    field: "execution.probability_space",
                    expected: ProbabilitySpace::OfferedOptionsPlusSemanticNone
                        .as_str()
                        .to_owned(),
                    actual: declared.as_str().to_owned(),
                });
            }
        }
        let concurrent = config.max_concurrent_requests.max(1);
        let admitted = concurrent.saturating_add(config.max_queued_requests);
        Ok(Self {
            inner: Arc::new(EngineInner {
                tokenizer,
                backbone,
                head: bundle.head().clone(),
                scheduler,
                max_concurrent_requests: concurrent,
                backend_id: config.backend.as_str(),
                evaluation_timeout: config.evaluation_timeout,
            }),
            execution_slots: Arc::new(Semaphore::new(concurrent)),
            admission_slots: Arc::new(Semaphore::new(admitted)),
            retry_after_ms: config.retry_after_ms,
        })
    }
}

#[async_trait]
impl DecisionEngine for Qwen35DecisionEngine {
    fn backend_id(&self) -> &str {
        self.inner.backend_id
    }

    fn model_metadata(&self) -> ModelInfo {
        let flavor = if self.inner.backend_id == Qwen35Backend::NativeCpu.as_str() {
            "native CPU reference"
        } else {
            "MLX reference-ops"
        };
        ModelInfo {
            name: String::new(),
            description: format!("Pinned Qwen3.5-4B {flavor} engine with explicit semantic none."),
            release_date: "2026-09-20".into(),
        }
    }

    async fn evaluate(&self, request: SystemRequest) -> EngineResult<SystemResponse> {
        let started = Instant::now();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut cancellation = CancellationOnDrop::new(cancelled.clone());
        let result = self.evaluate_controlled(request, started, cancelled).await;
        cancellation.disarm();
        result
    }
}

impl Qwen35DecisionEngine {
    async fn evaluate_controlled(
        &self,
        request: SystemRequest,
        started: Instant,
        cancelled: Arc<AtomicBool>,
    ) -> EngineResult<SystemResponse> {
        let timeout_ms = self
            .inner
            .evaluation_timeout
            .map(|timeout| timeout.as_millis().min(u128::from(u64::MAX)) as u64);
        let deadline = self
            .inner
            .evaluation_timeout
            .map(|timeout| started.checked_add(timeout).unwrap_or(started));
        let admission = self
            .admission_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| EngineError::Overloaded {
                backend: self.backend_id().to_owned(),
                retry_after_ms: self.retry_after_ms,
            });
        let admission = match admission {
            Ok(permit) => permit,
            Err(error) => {
                record_request(started, "overloaded");
                return Err(error);
            }
        };
        let queue_started = Instant::now();
        let execution = match acquire_execution_slot(
            self.execution_slots.clone(),
            deadline,
            self.backend_id(),
            timeout_ms.unwrap_or_default(),
        )
        .await
        {
            Ok(permit) => permit,
            Err(error) => {
                let outcome = if matches!(&error, EngineError::DeadlineExceeded { .. }) {
                    "deadline"
                } else {
                    "backend_error"
                };
                record_request(started, outcome);
                return Err(error);
            }
        };
        metrics::histogram!("openkind_native_queue_wait_seconds")
            .record(queue_started.elapsed().as_secs_f64());
        let inner = self.inner.clone();
        let backend = self.backend_id().to_owned();
        let control = ExecutionControl::new(cancelled, deadline);
        let execution_started = Instant::now();
        let joined = run_blocking_with_permits(execution, admission, move || {
            let result = evaluate_request(&inner, request, &control);
            metrics::histogram!("openkind_native_execution_seconds")
                .record(execution_started.elapsed().as_secs_f64());
            result
        })
        .await;
        let result = match joined {
            Ok(result) => result,
            Err(join) => {
                let error = EngineError::Backend {
                    backend,
                    message: format!("native evaluation task failed: {join}"),
                };
                record_request(started, "backend_error");
                return Err(error);
            }
        };
        let result = result.map_err(|error| map_evaluation_error(&backend, error, timeout_ms));
        let outcome = match &result {
            Ok(_) => "ok",
            Err(EngineError::DeadlineExceeded { .. }) => "deadline",
            Err(EngineError::Overloaded { .. }) => "overloaded",
            Err(EngineError::Unsupported { .. } | EngineError::Invalid(_)) => "invalid",
            Err(_) => "backend_error",
        };
        record_request(started, outcome);
        result
    }
}

struct CancellationOnDrop {
    cancelled: Arc<AtomicBool>,
    armed: bool,
}

impl CancellationOnDrop {
    fn new(cancelled: Arc<AtomicBool>) -> Self {
        Self {
            cancelled,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancellationOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancelled.store(true, Ordering::Release);
            metrics::counter!("openkind_native_cancellations_total").increment(1);
        }
    }
}

fn deadline_error(backend: &str, timeout_ms: u64) -> EngineError {
    EngineError::DeadlineExceeded {
        backend: backend.to_owned(),
        timeout_ms,
    }
}

async fn acquire_execution_slot(
    slots: Arc<Semaphore>,
    deadline: Option<Instant>,
    backend: &str,
    timeout_ms: u64,
) -> EngineResult<tokio::sync::OwnedSemaphorePermit> {
    let acquire = slots.acquire_owned();
    if let Some(deadline) = deadline {
        match tokio::time::timeout_at(deadline.into(), acquire).await {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) => Err(EngineError::Backend {
                backend: backend.to_owned(),
                message: "execution admission closed during shutdown".into(),
            }),
            Err(_) => Err(deadline_error(backend, timeout_ms)),
        }
    } else {
        acquire.await.map_err(|_| EngineError::Backend {
            backend: backend.to_owned(),
            message: "execution admission closed during shutdown".into(),
        })
    }
}

fn record_request(started: Instant, outcome: &'static str) {
    metrics::histogram!("openkind_native_request_seconds").record(started.elapsed().as_secs_f64());
    metrics::counter!("openkind_native_requests_total", "outcome" => outcome).increment(1);
}

async fn run_blocking_with_permits<T, F>(
    execution: tokio::sync::OwnedSemaphorePermit,
    admission: tokio::sync::OwnedSemaphorePermit,
    work: F,
) -> Result<T, tokio::task::JoinError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        // The blocking task owns both permits. Dropping a cancelled HTTP/gRPC
        // future therefore cannot admit replacement work while native compute
        // is still running in the blocking pool.
        let _execution = execution;
        let _admission = admission;
        work()
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cancellation_guard_signals_only_when_abandoned() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let guard = CancellationOnDrop::new(cancelled.clone());
        drop(guard);
        assert!(cancelled.load(Ordering::Acquire));

        let completed = Arc::new(AtomicBool::new(false));
        let mut guard = CancellationOnDrop::new(completed.clone());
        guard.disarm();
        drop(guard);
        assert!(!completed.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn execution_queue_observes_the_queue_inclusive_deadline() {
        let slots = Arc::new(Semaphore::new(0));
        let deadline = Instant::now() + Duration::from_millis(10);
        let result = acquire_execution_slot(slots, Some(deadline), "test-backend", 10).await;
        assert!(matches!(
            result,
            Err(EngineError::DeadlineExceeded {
                ref backend,
                timeout_ms: 10
            }) if backend == "test-backend"
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_keeps_permits_until_blocking_work_finishes() {
        let execution = Arc::new(Semaphore::new(1));
        let admission = Arc::new(Semaphore::new(1));
        let execution_permit = execution
            .clone()
            .acquire_owned()
            .await
            .expect("execution permit");
        let admission_permit = admission
            .clone()
            .acquire_owned()
            .await
            .expect("admission permit");
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let request = tokio::spawn(run_blocking_with_permits(
            execution_permit,
            admission_permit,
            move || {
                let _ = started_tx.send(());
                let _ = release_rx.blocking_recv();
            },
        ));

        started_rx.await.expect("blocking work started");
        request.abort();
        assert_eq!(execution.available_permits(), 0);
        assert_eq!(admission.available_permits(), 0);

        release_tx.send(()).expect("release blocking work");
        tokio::time::timeout(Duration::from_secs(2), async {
            while execution.available_permits() == 0 || admission.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("permits released after native work completed");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_panic_releases_permits_for_recovery() {
        let execution = Arc::new(Semaphore::new(1));
        let admission = Arc::new(Semaphore::new(1));
        let execution_permit = execution
            .clone()
            .acquire_owned()
            .await
            .expect("execution permit");
        let admission_permit = admission
            .clone()
            .acquire_owned()
            .await
            .expect("admission permit");

        let failed = run_blocking_with_permits(execution_permit, admission_permit, || {
            panic!("simulated native worker failure");
        })
        .await;
        assert!(failed.is_err());
        assert_eq!(execution.available_permits(), 1);
        assert_eq!(admission.available_permits(), 1);

        let recovered_execution = execution
            .clone()
            .try_acquire_owned()
            .expect("execution capacity recovered");
        let recovered_admission = admission
            .clone()
            .try_acquire_owned()
            .expect("admission capacity recovered");
        drop((recovered_execution, recovered_admission));
    }
}
