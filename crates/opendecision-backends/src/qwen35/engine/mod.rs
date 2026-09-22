//! Direct Jev wire adapter for the pinned native Qwen3.5 profile.

mod backbone;
mod canonical;
mod eval;
mod mapping;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use opendecision_core::{ModelInfo, SystemRequest, SystemResponse};
use opendecision_engine::{DecisionEngine, EngineError, EngineResult, ProbabilitySpace};
use tokio::sync::Semaphore;

pub use self::backbone::Qwen35Backend;
use self::backbone::{load_backbone, EngineBackbone};
use self::eval::{evaluate_request, map_evaluation_error};
use super::{Qwen35Error, Qwen35Tokenizer, ReferenceBundle, SchedulerConfig, ScoreSummaryHead};

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
}

pub(super) struct EngineInner {
    pub(super) tokenizer: Qwen35Tokenizer,
    pub(super) backbone: EngineBackbone,
    pub(super) head: ScoreSummaryHead,
    pub(super) scheduler: SchedulerConfig,
    pub(super) max_concurrent_requests: usize,
    pub(super) backend_id: &'static str,
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
        let admission = self
            .admission_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| EngineError::Overloaded {
                backend: self.backend_id().to_owned(),
                retry_after_ms: self.retry_after_ms,
            })?;
        let execution = self
            .execution_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| EngineError::Backend {
                backend: self.backend_id().to_owned(),
                message: "execution admission closed during shutdown".into(),
            })?;
        let inner = self.inner.clone();
        let backend = self.backend_id().to_owned();
        let result = run_blocking_with_permits(execution, admission, move || {
            evaluate_request(&inner, request)
        })
        .await
        .map_err(|join| EngineError::Backend {
            backend: backend.clone(),
            message: format!("native evaluation task failed: {join}"),
        })?;
        result.map_err(|error| map_evaluation_error(&backend, error))
    }
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
}
