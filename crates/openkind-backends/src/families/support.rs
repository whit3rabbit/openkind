//! Shared bounded-engine scaffold for surveyed-family implementations.
//!
//! Every surveyed family loads local pinned artifacts offline, evaluates a
//! `SystemRequest` on the host (never autoregressively), and registers behind
//! [`DecisionEngine`](openkind_engine::DecisionEngine) through
//! [`BoundedFamilyEngine`]. The scaffold centralizes the admission/queue/
//! deadline/cancellation plumbing so family modules only implement loading
//! and evaluation.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};
use openkind_engine::{DecisionEngine, EngineError, EngineResult};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::sync::Semaphore;

/// Errors shared by the surveyed-family loaders and evaluators.
#[derive(Debug, Error)]
pub enum FamilyError {
    /// A required artifact could not be read.
    #[error("failed to read `{path}`: {source}")]
    Io {
        /// Artifact path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },

    /// A JSON artifact could not be decoded.
    #[error("failed to decode JSON artifact `{path}`: {source}")]
    Json {
        /// Artifact path.
        path: PathBuf,
        /// Underlying JSON error.
        source: serde_json::Error,
    },

    /// A tokenizer artifact could not be decoded or executed.
    #[error("tokenizer failure: {0}")]
    Tokenizer(String),

    /// Native tensor execution failed.
    #[error("native family tensor execution failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// The MLX execution backend failed (feature `mlx`, macOS arm64).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    #[error("family MLX execution failed: {0}")]
    Mlx(String),

    /// The shared native Qwen3.5 backbone failed during a survey-profile
    /// forward.
    #[error("qwen35 native backbone failure: {0}")]
    Qwen35(#[from] crate::qwen35::Qwen35Error),

    /// A pinned contract field did not match the loaded artifact.
    #[error("reference contract mismatch for `{field}`: expected `{expected}`, found `{actual}`")]
    ContractMismatch {
        /// Name of the mismatched field.
        field: &'static str,
        /// Required pinned value.
        expected: String,
        /// Observed artifact value.
        actual: String,
    },

    /// An artifact did not match its pinned digest.
    #[error("SHA-256 mismatch for `{path}`: expected {expected}, found {actual}")]
    DigestMismatch {
        /// Artifact path.
        path: String,
        /// Required digest.
        expected: String,
        /// Observed digest.
        actual: String,
    },

    /// The request or artifact violated the family contract.
    #[error("invalid family input: {0}")]
    InvalidInput(String),

    /// Evaluation produced non-finite arithmetic instead of a distribution.
    #[error("family evaluation produced invalid arithmetic: {0}")]
    Numerical(String),

    /// A disconnected caller requested cooperative cancellation.
    #[error("family evaluation was cancelled after caller disconnect")]
    Cancelled,

    /// The configured queue-inclusive evaluation deadline elapsed.
    #[error("family evaluation exceeded its configured deadline of {timeout_ms} ms")]
    DeadlineExceeded {
        /// Configured queue-inclusive deadline in milliseconds.
        timeout_ms: u64,
    },
}

impl FamilyError {
    /// Construct a [`FamilyError::ContractMismatch`] from any string-ish values.
    pub fn contract(field: &'static str, expected: impl ToString, actual: impl ToString) -> Self {
        Self::ContractMismatch {
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }
}

/// Request-scoped cancellation and deadline checks for family evaluation.
#[derive(Clone, Debug)]
pub struct FamilyControl {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
    timeout_ms: u64,
}

impl FamilyControl {
    pub(crate) fn new(
        cancelled: Arc<AtomicBool>,
        deadline: Option<Instant>,
        timeout_ms: u64,
    ) -> Self {
        Self {
            cancelled,
            deadline,
            timeout_ms,
        }
    }

    /// Fail when the caller disconnected or the queue-inclusive deadline elapsed.
    pub fn check(&self) -> Result<(), FamilyError> {
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(FamilyError::DeadlineExceeded {
                timeout_ms: self.timeout_ms,
            });
        }
        if self.cancelled.load(Ordering::Acquire) {
            return Err(FamilyError::Cancelled);
        }
        Ok(())
    }
}

/// Blocking evaluation entry point implemented by every family engine.
pub trait FamilyEvaluator: Send + Sync {
    /// Stable backend identifier, e.g. `"encoder-nli-native-cpu"`.
    fn backend_id(&self) -> &str;

    /// Public metadata surfaced by `GET /v1/models`.
    fn model_metadata(&self) -> ModelInfo;

    /// Evaluate one request synchronously. Implementations must honor
    /// cancellation and deadlines through [`FamilyControl`].
    fn evaluate(
        &self,
        request: SystemRequest,
        control: &FamilyControl,
    ) -> Result<SystemResponse, FamilyError>;
}

/// Admission settings shared by every family engine.
#[derive(Debug, Clone)]
pub struct FamilyLimits {
    /// Maximum model evaluations executing concurrently.
    pub max_concurrent_requests: usize,
    /// Maximum additional requests allowed to wait for execution.
    pub max_queued_requests: usize,
    /// Suggested backoff returned when admission is full.
    pub retry_after_ms: u64,
    /// Queue-inclusive request deadline. `None` leaves evaluation unbounded.
    pub evaluation_timeout: Option<Duration>,
}

/// Bounded [`DecisionEngine`] wrapper executing a [`FamilyEvaluator`] on the
/// blocking pool with admission, queue, deadline, and cancellation control.
pub struct BoundedFamilyEngine {
    evaluator: Arc<dyn FamilyEvaluator>,
    execution_slots: Arc<Semaphore>,
    admission_slots: Arc<Semaphore>,
    retry_after_ms: u64,
    evaluation_timeout: Option<Duration>,
}

impl BoundedFamilyEngine {
    /// Wrap a loaded family evaluator with admission control.
    pub fn new(evaluator: Arc<dyn FamilyEvaluator>, limits: FamilyLimits) -> Self {
        let concurrent = limits.max_concurrent_requests.max(1);
        let admitted = concurrent.saturating_add(limits.max_queued_requests);
        Self {
            evaluator,
            execution_slots: Arc::new(Semaphore::new(concurrent)),
            admission_slots: Arc::new(Semaphore::new(admitted)),
            retry_after_ms: limits.retry_after_ms,
            evaluation_timeout: limits.evaluation_timeout,
        }
    }

    fn backend_id(&self) -> String {
        self.evaluator.backend_id().to_owned()
    }
}

#[async_trait]
impl DecisionEngine for BoundedFamilyEngine {
    fn backend_id(&self) -> &str {
        self.evaluator.backend_id()
    }

    fn model_metadata(&self) -> ModelInfo {
        self.evaluator.model_metadata()
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

impl BoundedFamilyEngine {
    async fn evaluate_controlled(
        &self,
        request: SystemRequest,
        started: Instant,
        cancelled: Arc<AtomicBool>,
    ) -> EngineResult<SystemResponse> {
        let backend = self.backend_id();
        let deadline = self
            .evaluation_timeout
            .and_then(|timeout| started.checked_add(timeout));
        let record = |outcome: &'static str| {
            metrics::histogram!("openkind_family_request_seconds")
                .record(started.elapsed().as_secs_f64());
            metrics::counter!("openkind_family_requests_total", "outcome" => outcome).increment(1);
        };

        let admission = self
            .admission_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| EngineError::Overloaded {
                backend: backend.clone(),
                retry_after_ms: self.retry_after_ms,
            });
        let admission = match admission {
            Ok(permit) => permit,
            Err(error) => {
                record("overloaded");
                return Err(error);
            }
        };
        let queue_started = Instant::now();
        let execution = match acquire_execution_slot(
            self.execution_slots.clone(),
            deadline,
            &backend,
            self.evaluation_timeout,
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
                record(outcome);
                return Err(error);
            }
        };
        metrics::histogram!("openkind_family_queue_wait_seconds")
            .record(queue_started.elapsed().as_secs_f64());

        let evaluator = self.evaluator.clone();
        let timeout_ms = self
            .evaluation_timeout
            .map(|timeout| timeout.as_millis().min(u128::from(u64::MAX)) as u64)
            .unwrap_or_default();
        let control = FamilyControl::new(cancelled, deadline, timeout_ms);
        let execution_started = Instant::now();
        let joined = tokio::task::spawn_blocking(move || {
            let _execution = execution;
            let _admission = admission;
            let result = evaluator.evaluate(request, &control);
            metrics::histogram!("openkind_family_execution_seconds")
                .record(execution_started.elapsed().as_secs_f64());
            result
        })
        .await;
        let result = match joined {
            Ok(result) => result.map_err(|error| map_family_error(&backend, error)),
            Err(join) => Err(EngineError::Backend {
                backend,
                message: format!("family evaluation task failed: {join}"),
            }),
        };
        let outcome = match &result {
            Ok(_) => "ok",
            Err(EngineError::DeadlineExceeded { .. }) => "deadline",
            Err(EngineError::Overloaded { .. }) => "overloaded",
            Err(EngineError::Unsupported { .. }) => "invalid",
            Err(_) => "backend_error",
        };
        record(outcome);
        result
    }
}

async fn acquire_execution_slot(
    slots: Arc<Semaphore>,
    deadline: Option<Instant>,
    backend: &str,
    evaluation_timeout: Option<Duration>,
) -> EngineResult<tokio::sync::OwnedSemaphorePermit> {
    let acquire = slots.acquire_owned();
    if let Some(deadline) = deadline {
        match tokio::time::timeout_at(deadline.into(), acquire).await {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) => Err(EngineError::Backend {
                backend: backend.to_owned(),
                message: "family execution admission closed during shutdown".into(),
            }),
            Err(_) => Err(EngineError::DeadlineExceeded {
                backend: backend.to_owned(),
                timeout_ms: evaluation_timeout
                    .map(|timeout| timeout.as_millis() as u64)
                    .unwrap_or_default(),
            }),
        }
    } else {
        acquire.await.map_err(|_| EngineError::Backend {
            backend: backend.to_owned(),
            message: "family execution admission closed during shutdown".into(),
        })
    }
}

fn map_family_error(backend: &str, error: FamilyError) -> EngineError {
    let backend = backend.to_owned();
    match error {
        FamilyError::InvalidInput(message) => EngineError::Unsupported { backend, message },
        FamilyError::DeadlineExceeded { timeout_ms } => EngineError::DeadlineExceeded {
            backend,
            timeout_ms,
        },
        error => EngineError::Backend {
            backend,
            message: error.to_string(),
        },
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
            metrics::counter!("openkind_family_cancellations_total").increment(1);
        }
    }
}

/// Stream a file through SHA-256 in place. Multi-gigabyte checkpoint shards
/// are never copied or staged: verification reads them where they live.
pub fn sha256_file(path: &Path) -> Result<String, FamilyError> {
    let file = File::open(path).map_err(|source| FamilyError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::with_capacity(8 << 20, file);
    let mut hasher = Sha256::new();
    let mut chunk = vec![0u8; 8 << 20];
    loop {
        let read = reader.read(&mut chunk).map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify a local artifact against its pinned SHA-256 digest, in place.
pub fn verify_digest(path: &Path, expected: &str) -> Result<(), FamilyError> {
    let actual = sha256_file(path)?;
    if actual != expected {
        return Err(FamilyError::DigestMismatch {
            path: path.display().to_string(),
            expected: expected.to_owned(),
            actual,
        });
    }
    Ok(())
}

/// Read and decode a JSON artifact from disk.
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, FamilyError> {
    let bytes = std::fs::read(path).map_err(|source| FamilyError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| FamilyError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// Derive a stable 20-hex profile ID from a family slug and pinned inputs.
///
/// The selected `encoder-state-first` profile reserves its own ID; generated
/// IDs only guarantee that identical (family, model, revision) inputs produce
/// identical profile IDs across runs and documents.
pub fn derive_profile_id(family: &str, backbone: &str, revision: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"openkind:family:");
    hasher.update(family.as_bytes());
    hasher.update([0u8]);
    hasher.update(backbone.as_bytes());
    hasher.update([0u8]);
    hasher.update(revision.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    digest[..20].to_owned()
}

/// Softmax with temperature over `f64` logits, returning finite probabilities.
pub fn temperature_softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>, FamilyError> {
    if logits.is_empty() {
        return Err(FamilyError::InvalidInput(
            "cannot normalize an empty logit vector".into(),
        ));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(FamilyError::InvalidInput(format!(
            "calibration temperature must be finite and positive, found {temperature}"
        )));
    }
    let scaled: Vec<f64> = logits.iter().map(|logit| logit / temperature).collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err(FamilyError::Numerical(
            "logit vector contains no finite maximum".into(),
        ));
    }
    let exponentials: Vec<f64> = scaled.iter().map(|value| (value - max).exp()).collect();
    let sum: f64 = exponentials.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(FamilyError::Numerical(format!(
            "softmax normalizer is not finite: {sum}"
        )));
    }
    Ok(exponentials.iter().map(|value| value / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use openkind_core::Usage;

    fn evaluator(backend: &'static str) -> impl FamilyEvaluator {
        struct Fixed(&'static str);
        impl FamilyEvaluator for Fixed {
            fn backend_id(&self) -> &str {
                self.0
            }
            fn model_metadata(&self) -> ModelInfo {
                ModelInfo {
                    name: String::new(),
                    description: "fixed".into(),
                    release_date: "1970-01-01".into(),
                }
            }
            fn evaluate(
                &self,
                request: SystemRequest,
                _control: &FamilyControl,
            ) -> Result<SystemResponse, FamilyError> {
                Ok(SystemResponse {
                    model: request.model,
                    answers: Default::default(),
                    usage: Usage {
                        input_tokens: 0,
                        output_tokens: 0,
                    },
                })
            }
        }
        Fixed(backend)
    }

    fn request() -> SystemRequest {
        serde_json::from_value(serde_json::json!({
            "state": "text",
            "model": "test",
            "questions": {}
        }))
        .expect("request")
    }

    #[tokio::test]
    async fn bounded_engine_evaluates_and_reports_backend_identity() {
        let engine = BoundedFamilyEngine::new(
            Arc::new(evaluator("test-family")),
            FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                retry_after_ms: 1,
                evaluation_timeout: Some(Duration::from_secs(10)),
            },
        );
        assert_eq!(engine.backend_id(), "test-family");
        let response = engine.evaluate(request()).await.expect("evaluate");
        assert_eq!(response.model, "test");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bounded_engine_admission_rejects_beyond_the_queue() {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

        struct Gated(Arc<AtomicBool>);
        impl FamilyEvaluator for Gated {
            fn backend_id(&self) -> &str {
                "gated-family"
            }
            fn model_metadata(&self) -> ModelInfo {
                ModelInfo {
                    name: String::new(),
                    description: "gated".into(),
                    release_date: "1970-01-01".into(),
                }
            }
            fn evaluate(
                &self,
                request: SystemRequest,
                _control: &FamilyControl,
            ) -> Result<SystemResponse, FamilyError> {
                // Park until the test releases the gate, mirroring a long
                // native evaluation occupying the single execution slot.
                while !self.0.load(AtomicOrdering::Acquire) {
                    std::thread::yield_now();
                }
                Ok(SystemResponse {
                    model: request.model,
                    answers: Default::default(),
                    usage: Usage {
                        input_tokens: 0,
                        output_tokens: 0,
                    },
                })
            }
        }

        let gate = Arc::new(AtomicBool::new(false));
        let engine = Arc::new(BoundedFamilyEngine::new(
            Arc::new(Gated(gate.clone())),
            FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                retry_after_ms: 25,
                evaluation_timeout: Some(Duration::from_secs(10)),
            },
        ));

        // Park one evaluation on the single execution slot.
        let busy_engine = Arc::clone(&engine);
        let busy = tokio::spawn(async move { busy_engine.evaluate(request()).await });
        tokio::time::sleep(Duration::from_millis(200)).await;

        let rejected = engine
            .evaluate(request())
            .await
            .expect_err("admission must reject while the execution slot is busy");
        assert!(
            matches!(rejected, EngineError::Overloaded { ref retry_after_ms, .. } if *retry_after_ms == 25)
        );

        // Release the gate: the parked evaluation completes and frees the
        // slot, so the engine admits work again.
        gate.store(true, AtomicOrdering::Release);
        let released = busy
            .await
            .expect("parked evaluation joins")
            .expect("parked evaluation succeeds");
        let _ = released;
        let recovered = engine.evaluate(request()).await;
        assert!(recovered.is_ok(), "engine recovers after the slot frees");
    }

    #[test]
    fn temperature_softmax_normalizes_and_respects_order() {
        let probabilities = temperature_softmax(&[1.0, 2.0, 3.0], 1.0).expect("softmax");
        let sum: f64 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(probabilities[2] > probabilities[1]);
        assert!(probabilities[1] > probabilities[0]);

        assert!(temperature_softmax(&[], 1.0).is_err());
        assert!(temperature_softmax(&[1.0], 0.0).is_err());
        assert!(temperature_softmax(&[1.0, f64::NAN], 1.0).is_err());
    }

    #[test]
    fn profile_id_is_stable_and_well_formed() {
        let first = derive_profile_id("encoder-nli", "model", "rev1");
        let second = derive_profile_id("encoder-nli", "model", "rev1");
        let other = derive_profile_id("encoder-nli", "model", "rev2");
        assert_eq!(first, second);
        assert_ne!(first, other);
        assert_eq!(first.len(), 20);
        assert!(first
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    }
}
