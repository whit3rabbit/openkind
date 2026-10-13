//! Shared bounded-engine scaffold for surveyed-family implementations.
//!
//! Every surveyed family loads local pinned artifacts offline, evaluates a
//! `SystemRequest` on the host (never autoregressively), and registers behind
//! [`DecisionEngine`](openkind_engine::DecisionEngine) through
//! [`BoundedFamilyEngine`]. The scaffold centralizes the admission/queue/
//! deadline/cancellation plumbing so family modules only implement loading
//! and evaluation.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};
use openkind_engine::{DecisionEngine, EngineError, EngineResult};
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

    /// The ONNX execution backend failed (feature `onnx`).
    #[cfg(feature = "onnx")]
    #[error("family ONNX execution failed: {0}")]
    Onnx(#[from] crate::onnx::OnnxError),

    /// The requested execution backend is not available in this build or on
    /// this host.
    #[error("execution backend unavailable: {0}")]
    ExecutionUnavailable(String),

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

#[cfg(feature = "onnx")]
pub(crate) fn onnx_marker_positions(
    markers: &[usize],
    token_count: usize,
) -> Result<Vec<i64>, FamilyError> {
    markers
        .iter()
        .map(|&position| {
            // A wrapped negative Gather index can select another token
            // successfully, so reject invalid positions before execution.
            if position >= token_count {
                return Err(FamilyError::InvalidInput(format!(
                    "marker position {position} exceeds the sequence length {token_count}"
                )));
            }
            i64::try_from(position).map_err(|_| {
                FamilyError::InvalidInput("marker position exceeds ONNX int64".to_owned())
            })
        })
        .collect()
}

#[cfg(all(test, feature = "onnx"))]
#[test]
fn onnx_markers_reject_wrapped_negative_and_out_of_bounds_indices() {
    assert!(onnx_marker_positions(&[usize::MAX], 3).is_err());
    assert!(onnx_marker_positions(&[3], 3).is_err());
    assert_eq!(onnx_marker_positions(&[2, 0], 3).unwrap(), vec![2, 0]);
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

/// Calibrated letter-logit temperature of one profile.
///
/// Uniform profiles apply one temperature to every question type; by-type
/// profiles follow the reference serving layer's per-type table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Calibration {
    /// One temperature for Choice, Score, and Noul questions.
    Uniform(f64),
    /// Per-type temperatures from the reference serving layer.
    ByType {
        /// Temperature for Choice questions.
        choice: f64,
        /// Temperature for Score questions.
        score: f64,
        /// Temperature for Noul questions.
        noul: f64,
    },
}

/// Question primitive of the pass being calibrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuestionKind {
    /// Choice questions.
    Choice,
    /// Score questions.
    Score,
    /// Noul (yes/no) questions.
    Noul,
}

impl Calibration {
    /// The temperature for one question primitive.
    pub(crate) fn resolve(&self, primitive: QuestionKind) -> f64 {
        match self {
            Self::Uniform(temperature) => *temperature,
            Self::ByType {
                choice,
                score,
                noul,
            } => match primitive {
                QuestionKind::Choice => *choice,
                QuestionKind::Score => *score,
                QuestionKind::Noul => *noul,
            },
        }
    }
}

/// The frozen backbone geometry, shared with sibling families that serve the
/// same 32-layer Qwen3.5 hybrid checkpoint layout (e.g. `decider`).
pub(crate) fn pinned_config_shared() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("architectures[0]", json!("Qwen3_5ForCausalLM")),
        ("model_type", json!("qwen3_5_text")),
        ("dtype", json!("bfloat16")),
        ("hidden_size", json!(2_560)),
        ("intermediate_size", json!(9_216)),
        ("num_hidden_layers", json!(32)),
        ("full_attention_interval", json!(4)),
        ("num_attention_heads", json!(16)),
        ("num_key_value_heads", json!(4)),
        ("head_dim", json!(256)),
        ("linear_num_key_heads", json!(16)),
        ("linear_num_value_heads", json!(32)),
        ("linear_key_head_dim", json!(128)),
        ("linear_value_head_dim", json!(128)),
        ("linear_conv_kernel_dim", json!(4)),
        ("vocab_size", json!(248_320)),
        ("tie_word_embeddings", json!(true)),
        ("rms_norm_eps", json!(1e-06)),
        ("partial_rotary_factor", json!(0.25)),
        ("rope_parameters.rope_theta", json!(10_000_000)),
        ("attn_output_gate", json!(true)),
        ("attention_bias", json!(false)),
    ]
}

/// Metric names one [`AdmissionGate`] records under.
///
/// The names are exported Prometheus series, so each engine family keeps its
/// own prefix: only the label strings differ between users of the gate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GateMetrics {
    /// Histogram of whole-request seconds.
    pub request_seconds: &'static str,
    /// Counter of requests, labelled by `outcome`.
    pub requests_total: &'static str,
    /// Histogram of seconds spent waiting for an execution slot.
    pub queue_wait_seconds: &'static str,
    /// Histogram of seconds spent inside the blocking task.
    pub execution_seconds: &'static str,
    /// Counter of requests abandoned by their caller.
    pub cancellations_total: &'static str,
}

/// Metrics of the surveyed-family engines (`BoundedFamilyEngine`, winnow).
pub(crate) const FAMILY_GATE_METRICS: GateMetrics = GateMetrics {
    request_seconds: "openkind_family_request_seconds",
    requests_total: "openkind_family_requests_total",
    queue_wait_seconds: "openkind_family_queue_wait_seconds",
    execution_seconds: "openkind_family_execution_seconds",
    cancellations_total: "openkind_family_cancellations_total",
};

/// Metrics of the native Qwen3.5 engine.
pub(crate) const NATIVE_GATE_METRICS: GateMetrics = GateMetrics {
    request_seconds: "openkind_native_request_seconds",
    requests_total: "openkind_native_requests_total",
    queue_wait_seconds: "openkind_native_queue_wait_seconds",
    execution_seconds: "openkind_native_execution_seconds",
    cancellations_total: "openkind_native_cancellations_total",
};

/// Duration in whole milliseconds, saturating at `u64::MAX`.
pub(crate) fn saturating_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Fixed-label outcome of one finished request, for `requests_total`.
pub(crate) fn outcome_label<T>(result: &EngineResult<T>) -> &'static str {
    match result {
        Ok(_) => "ok",
        Err(EngineError::DeadlineExceeded { .. }) => "deadline",
        Err(EngineError::Overloaded { .. }) => "overloaded",
        Err(EngineError::Unsupported { .. } | EngineError::Invalid(_)) => "invalid",
        Err(_) => "backend_error",
    }
}

/// Admission, queue, and blocking-pool plumbing shared by every engine that
/// runs model work off the async runtime.
///
/// A request first takes an admission permit (rejecting immediately when the
/// queue is full), then waits for an execution slot under its deadline, then
/// runs on the blocking pool. Both permits travel into the blocking task, so
/// dropping a cancelled caller cannot admit replacement work while model
/// compute is still running. Callers keep their own deadline policy and
/// result mapping; the gate owns only the permits and the metrics around them.
pub(crate) struct AdmissionGate {
    execution_slots: Arc<Semaphore>,
    admission_slots: Arc<Semaphore>,
    retry_after_ms: u64,
    metrics: GateMetrics,
}

impl AdmissionGate {
    /// Size the gate: at least one concurrent evaluation, plus the queue.
    pub(crate) fn new(
        max_concurrent_requests: usize,
        max_queued_requests: usize,
        retry_after_ms: u64,
        metrics: GateMetrics,
    ) -> Self {
        let concurrent = max_concurrent_requests.max(1);
        let admitted = concurrent.saturating_add(max_queued_requests);
        Self {
            execution_slots: Arc::new(Semaphore::new(concurrent)),
            admission_slots: Arc::new(Semaphore::new(admitted)),
            retry_after_ms,
            metrics,
        }
    }

    /// Arm a guard that raises `cancelled` if the caller's future is dropped
    /// before [`CancellationOnDrop::disarm`].
    pub(crate) fn cancellation_guard(&self, cancelled: Arc<AtomicBool>) -> CancellationOnDrop {
        CancellationOnDrop::new(cancelled, self.metrics.cancellations_total)
    }

    /// Record one finished request under the gate's metric names.
    pub(crate) fn record(&self, started: Instant, outcome: &'static str) {
        metrics::histogram!(self.metrics.request_seconds).record(started.elapsed().as_secs_f64());
        metrics::counter!(self.metrics.requests_total, "outcome" => outcome).increment(1);
    }

    /// Admit, queue, and run `work` on the blocking pool.
    ///
    /// Overload, a deadline that elapses while queued, a closed pool, and a
    /// failed (panicked) task are recorded here and returned as errors. A
    /// completed task returns its value untouched: the caller maps it to a
    /// wire result and records that final outcome itself.
    pub(crate) async fn run<T, F>(
        &self,
        backend: &str,
        started: Instant,
        deadline: Option<Instant>,
        timeout_ms: u64,
        work: F,
    ) -> EngineResult<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let admission = match self.admission_slots.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                self.record(started, "overloaded");
                return Err(EngineError::Overloaded {
                    backend: backend.to_owned(),
                    retry_after_ms: self.retry_after_ms,
                });
            }
        };
        let queue_started = Instant::now();
        let execution = match acquire_execution_slot(
            self.execution_slots.clone(),
            deadline,
            backend,
            timeout_ms,
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
                self.record(started, outcome);
                return Err(error);
            }
        };
        metrics::histogram!(self.metrics.queue_wait_seconds)
            .record(queue_started.elapsed().as_secs_f64());

        let execution_metric = self.metrics.execution_seconds;
        let execution_started = Instant::now();
        let joined = run_blocking_with_permits(execution, admission, move || {
            let result = work();
            metrics::histogram!(execution_metric).record(execution_started.elapsed().as_secs_f64());
            result
        })
        .await;
        joined.map_err(|join| {
            self.record(started, "backend_error");
            EngineError::Backend {
                backend: backend.to_owned(),
                message: format!("evaluation task failed: {join}"),
            }
        })
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
            Err(_) => Err(EngineError::DeadlineExceeded {
                backend: backend.to_owned(),
                timeout_ms,
            }),
        }
    } else {
        acquire.await.map_err(|_| EngineError::Backend {
            backend: backend.to_owned(),
            message: "execution admission closed during shutdown".into(),
        })
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
        // future therefore cannot admit replacement work while model compute
        // is still running in the blocking pool.
        let _execution = execution;
        let _admission = admission;
        work()
    })
    .await
}

/// Raises a shared cancellation flag when the caller's future is dropped
/// before the request completes.
///
/// Evaluators observe the flag through their cooperative control and exit,
/// releasing the permits the blocking task holds.
pub(crate) struct CancellationOnDrop {
    cancelled: Arc<AtomicBool>,
    metric: &'static str,
    armed: bool,
}

impl CancellationOnDrop {
    pub(crate) fn new(cancelled: Arc<AtomicBool>, metric: &'static str) -> Self {
        Self {
            cancelled,
            metric,
            armed: true,
        }
    }

    /// The request completed: dropping the guard no longer cancels.
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancellationOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancelled.store(true, Ordering::Release);
            metrics::counter!(self.metric).increment(1);
        }
    }
}

/// Bounded [`DecisionEngine`] wrapper executing a [`FamilyEvaluator`] on the
/// blocking pool with admission, queue, deadline, and cancellation control.
pub struct BoundedFamilyEngine {
    evaluator: Arc<dyn FamilyEvaluator>,
    gate: AdmissionGate,
    evaluation_timeout: Option<Duration>,
}

impl BoundedFamilyEngine {
    /// Wrap a loaded family evaluator with admission control.
    pub fn new(evaluator: Arc<dyn FamilyEvaluator>, limits: FamilyLimits) -> Self {
        Self {
            evaluator,
            gate: AdmissionGate::new(
                limits.max_concurrent_requests,
                limits.max_queued_requests,
                limits.retry_after_ms,
                FAMILY_GATE_METRICS,
            ),
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
        let mut cancellation = self.gate.cancellation_guard(cancelled.clone());
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
        let deadline = match self.evaluation_timeout {
            Some(timeout) => match started.checked_add(timeout) {
                Some(deadline) => Some(deadline),
                None => {
                    // An unrepresentable configured deadline must not turn
                    // into an unbounded evaluation.
                    self.gate.record(started, "invalid");
                    return Err(EngineError::Unsupported {
                        backend,
                        message: "family evaluation timeout is outside the monotonic clock range"
                            .into(),
                    });
                }
            },
            None => None,
        };
        let timeout_ms = self
            .evaluation_timeout
            .map(saturating_millis)
            .unwrap_or_default();
        let control = FamilyControl::new(cancelled, deadline, timeout_ms);
        let evaluator = self.evaluator.clone();
        let joined = self
            .gate
            .run(&backend, started, deadline, timeout_ms, move || {
                evaluator.evaluate(request, &control)
            })
            .await?;
        let result = joined.map_err(|error| map_family_error(&backend, error));
        self.gate.record(started, outcome_label(&result));
        result
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

/// Stream a file through SHA-256 in place. Multi-gigabyte checkpoint shards
/// are never copied or staged: verification reads them where they live.
pub fn sha256_file(path: &Path) -> Result<String, FamilyError> {
    openkind_runtime::hashing::sha256_file(path).map_err(|source| FamilyError::Io {
        path: path.to_path_buf(),
        source,
    })
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

/// Fail with the loader's standard NotFound error unless `path` is a file.
///
/// Callers run this over every small artifact before any digest so a missing
/// file always beats a bad digest of an earlier one. Keep that order: do not
/// fold the existence pass into the digest pass.
pub(crate) fn require_file(path: &Path) -> Result<(), FamilyError> {
    require_file_labeled(path, "artifact")
}

/// [`require_file`] with the artifact kind or name spelled in the message,
/// for loaders whose error text names the missing file (for example
/// `"shard model.safetensors"` yields `required pinned shard
/// model.safetensors is missing`).
pub(crate) fn require_file_labeled(path: &Path, label: &str) -> Result<(), FamilyError> {
    if path.is_file() {
        return Ok(());
    }
    Err(FamilyError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("required pinned {label} is missing"),
        ),
    })
}

/// Check a file's byte length against its pin, then its SHA-256, in place.
///
/// The size check runs first because it is a metadata read: a truncated
/// multi-gigabyte shard fails before the whole file is streamed. `field`
/// names the size pin in the contract error and differs per call site.
pub(crate) fn verify_sized_digest(
    path: &Path,
    name: &str,
    bytes: u64,
    sha256: &str,
    field: &'static str,
) -> Result<(), FamilyError> {
    let length = std::fs::metadata(path)
        .map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if length != bytes {
        return Err(FamilyError::ContractMismatch {
            field,
            expected: format!("{name} is {bytes} bytes"),
            actual: format!("{length} bytes"),
        });
    }
    verify_digest(path, sha256)
}

/// Resolve a dotted config path such as `rope_parameters.rope_theta` or an
/// `architectures[0]` list index.
///
/// Any segment may carry the `[0]` suffix, which makes this a superset of the
/// per-family copies that only accepted it on the last segment.
pub(crate) fn resolve_config_field(
    value: &serde_json::Value,
    field: &str,
) -> Option<serde_json::Value> {
    let mut current = value;
    for segment in field.split('.') {
        if let Some(name) = segment.strip_suffix("[0]") {
            current = current.get(name).and_then(|items| items.get(0))?;
        } else {
            current = current.get(segment)?;
        }
    }
    Some(current.clone())
}

/// Contract comparison with exact integer, string, and bool equality and a
/// parser-epsilon tolerance for floats.
///
/// serde_json's default decimal parse can land one ulp away from the
/// correctly rounded double, which must not fail a pinned checkpoint. Only the
/// von and laya contracts opt in; every other family compares strictly, and
/// qwen3 deliberately compares an integer against a float without tolerance.
pub(crate) fn values_match_tolerant(
    expected: &serde_json::Value,
    actual: &serde_json::Value,
) -> bool {
    use serde_json::Value;
    match (expected, actual) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => (a - b).abs() <= 1e-9,
            _ => a == b,
        },
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_match_tolerant(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(key, value)| {
                    b.get(key)
                        .is_some_and(|other| values_match_tolerant(value, other))
                })
        }
        (a, b) => a == b,
    }
}

/// Fail closed unless every pinned `(field, expected)` is present in `config`
/// with exactly the expected value.
pub(crate) fn check_pinned_fields(
    config: &serde_json::Value,
    pins: &[(&'static str, serde_json::Value)],
) -> Result<(), FamilyError> {
    check_pinned_fields_with(config, pins, |_, expected, actual| expected == actual)
}

/// [`check_pinned_fields`] with the float-tolerant comparison von and laya
/// use for their ModernBERT configs.
pub(crate) fn check_pinned_fields_tolerant(
    config: &serde_json::Value,
    pins: &[(&'static str, serde_json::Value)],
) -> Result<(), FamilyError> {
    check_pinned_fields_with(config, pins, |_, expected, actual| {
        values_match_tolerant(expected, actual)
    })
}

/// [`check_pinned_fields`] with a caller-supplied comparison.
///
/// `matches` receives the pinned field name, so one field can compare
/// differently from the rest (d3 and sieve compare `rms_norm_eps` through
/// f32). A field absent from `config` reports `actual: "missing"` before the
/// comparison runs.
pub(crate) fn check_pinned_fields_with(
    config: &serde_json::Value,
    pins: &[(&'static str, serde_json::Value)],
    matches: impl Fn(&'static str, &serde_json::Value, &serde_json::Value) -> bool,
) -> Result<(), FamilyError> {
    for (field, expected) in pins {
        let actual =
            resolve_config_field(config, field).ok_or_else(|| FamilyError::ContractMismatch {
                field,
                expected: expected.to_string(),
                actual: "missing".to_owned(),
            })?;
        if !matches(field, expected, &actual) {
            return Err(FamilyError::contract(field, expected, actual));
        }
    }
    Ok(())
}

/// Derive a stable 20-hex profile ID from a family slug and pinned inputs.
///
/// The selected `encoder-state-first` profile reserves its own ID; generated
/// IDs only guarantee that identical (family, model, revision) inputs produce
/// identical profile IDs across runs and documents.
pub fn derive_profile_id(family: &str, backbone: &str, revision: &str) -> String {
    let digest = openkind_runtime::hashing::sha256_hex_parts(&[
        b"openkind:family:",
        family.as_bytes(),
        &[0u8],
        backbone.as_bytes(),
        &[0u8],
        revision.as_bytes(),
    ]);
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

/// Fixtures for the load-time precedence tests that pin which error wins
/// when several pinned artifacts are wrong at once.
#[cfg(test)]
pub(crate) mod testing {
    use std::path::{Path, PathBuf};

    use serde_json::{json, Value};

    /// Leak a string so a test can build a profile with `&'static str`
    /// digests computed from fabricated files.
    pub(crate) fn leak(text: impl Into<String>) -> &'static str {
        Box::leak(text.into().into_boxed_str())
    }

    /// Lowercase hex SHA-256 of `bytes`.
    pub(crate) fn sha(bytes: &[u8]) -> String {
        openkind_runtime::hashing::sha256_hex(bytes)
    }

    /// Write `bytes` at `root/relative`, creating parent directories.
    pub(crate) fn put(root: &Path, relative: &str, bytes: &[u8]) -> PathBuf {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().expect("fixture path has a parent"))
            .expect("create fixture directory");
        std::fs::write(&path, bytes).expect("write fixture file");
        path
    }

    /// Build a nested config document that satisfies every `(field, value)`
    /// pin, using the same dotted and `name[0]` path syntax the loaders read.
    pub(crate) fn config_from_pins(pins: &[(&'static str, Value)]) -> Value {
        let mut root = json!({});
        for (field, value) in pins {
            let mut current = &mut root;
            let segments: Vec<&str> = field.split('.').collect();
            for (index, segment) in segments.iter().enumerate() {
                let last = index + 1 == segments.len();
                let (name, listed) = match segment.strip_suffix("[0]") {
                    Some(name) => (name, true),
                    None => (*segment, false),
                };
                let object = current.as_object_mut().expect("pin path crosses an object");
                if last && !listed {
                    object.insert(name.to_owned(), value.clone());
                    break;
                }
                let slot = object.entry(name.to_owned()).or_insert_with(|| {
                    if listed {
                        json!([null])
                    } else {
                        json!({})
                    }
                });
                if listed {
                    if last {
                        slot.as_array_mut().expect("list slot")[0] = value.clone();
                        break;
                    }
                    let element = &mut slot.as_array_mut().expect("list slot")[0];
                    if element.is_null() {
                        *element = json!({});
                    }
                    current = element;
                } else {
                    current = slot;
                }
            }
        }
        root
    }

    /// Serialize a config document the way the fixtures store it.
    pub(crate) fn json_bytes(value: &Value) -> Vec<u8> {
        serde_json::to_vec(value).expect("serialize fixture json")
    }
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

    #[tokio::test]
    async fn unrepresentable_timeout_does_not_disable_the_deadline() {
        let engine = BoundedFamilyEngine::new(
            Arc::new(evaluator("test-family")),
            FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                retry_after_ms: 1,
                evaluation_timeout: Some(Duration::MAX),
            },
        );
        let error = engine
            .evaluate(request())
            .await
            .expect_err("an invalid timeout must fail closed");
        assert!(matches!(
            error,
            EngineError::Unsupported { message, .. } if message.contains("monotonic clock range")
        ));
        assert_eq!(engine.gate.execution_slots.available_permits(), 1);
        assert_eq!(engine.gate.admission_slots.available_permits(), 1);
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

    /// Evaluator that performs open-ended work but honors
    /// [`FamilyControl`] while doing so, like a cooperative native forward.
    struct CooperativeWorker;

    impl FamilyEvaluator for CooperativeWorker {
        fn backend_id(&self) -> &str {
            "cooperative-family"
        }
        fn model_metadata(&self) -> ModelInfo {
            ModelInfo {
                name: String::new(),
                description: "cooperative".into(),
                release_date: "1970-01-01".into(),
            }
        }
        fn evaluate(
            &self,
            _request: SystemRequest,
            control: &FamilyControl,
        ) -> Result<SystemResponse, FamilyError> {
            loop {
                control.check()?;
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cooperative_deadline_failure_reclaims_execution_capacity() {
        let engine = BoundedFamilyEngine::new(
            Arc::new(CooperativeWorker),
            FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                retry_after_ms: 10,
                evaluation_timeout: Some(Duration::from_millis(50)),
            },
        );
        let result = tokio::time::timeout(Duration::from_secs(30), engine.evaluate(request()))
            .await
            .expect("a cooperative evaluator must return once its deadline elapses");
        let error = match result {
            Ok(_) => panic!("the elapsed deadline must fail the evaluation"),
            Err(error) => error,
        };
        assert!(
            matches!(error, EngineError::DeadlineExceeded { timeout_ms: 50, .. }),
            "unexpected error: {error}"
        );
        // The blocking task exited through the cooperative check, so both
        // the execution and admission permits are back.
        assert_eq!(engine.gate.execution_slots.available_permits(), 1);
        assert_eq!(engine.gate.admission_slots.available_permits(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn caller_disconnect_reclaims_capacity_from_a_cooperative_evaluator() {
        let engine = Arc::new(BoundedFamilyEngine::new(
            Arc::new(CooperativeWorker),
            FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                retry_after_ms: 10,
                evaluation_timeout: Some(Duration::from_secs(600)),
            },
        ));
        let busy_engine = Arc::clone(&engine);
        let busy = tokio::spawn(async move { busy_engine.evaluate(request()).await });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            engine.gate.execution_slots.available_permits(),
            0,
            "the running evaluation holds the only execution slot"
        );

        // A dropped caller only sets the cancellation flag; the cooperative
        // evaluator must observe it and release the slot promptly.
        busy.abort();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline && engine.gate.execution_slots.available_permits() < 1 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            engine.gate.execution_slots.available_permits(),
            1,
            "caller disconnect must reclaim the execution slot via cooperative cancellation"
        );
        assert_eq!(engine.gate.admission_slots.available_permits(), 1);
    }

    #[test]
    fn config_field_resolution_walks_nested_paths_and_list_indices() {
        let value = serde_json::json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "rope_parameters": {"rope_theta": 10_000_000},
            "id2label": {"0": "ENTAILMENT"},
            "encoder_config": {"model_type": "modernbert"},
            "stages": [{"name": "first"}],
        });
        let resolve = |field| resolve_config_field(&value, field);
        assert_eq!(
            resolve("architectures[0]"),
            Some(serde_json::json!("Qwen3_5ForCausalLM"))
        );
        assert_eq!(
            resolve("rope_parameters.rope_theta"),
            Some(serde_json::json!(10_000_000))
        );
        assert_eq!(resolve("id2label.0"), Some(serde_json::json!("ENTAILMENT")));
        assert_eq!(
            resolve("encoder_config.model_type"),
            Some(serde_json::json!("modernbert"))
        );
        // The qwen35 loop form also accepts the index on an inner segment.
        assert_eq!(resolve("stages[0].name"), Some(serde_json::json!("first")));
        assert_eq!(resolve("missing.path"), None);
        assert_eq!(resolve("architectures[1]"), None);
        assert_eq!(resolve("model_type"), None);
    }

    /// The per-family copies this resolver replaced: the split-on-dot form
    /// six families carried and the flat form decoder-letter carried.
    fn legacy_dotted(value: &serde_json::Value, field: &str) -> Option<serde_json::Value> {
        let mut current = value;
        let mut segment = field;
        while let Some((head, rest)) = segment.split_once('.') {
            current = current.get(head)?;
            segment = rest;
        }
        if let Some(name) = segment.strip_suffix("[0]") {
            return current.get(name).and_then(|items| items.get(0)).cloned();
        }
        current.get(segment).cloned()
    }

    fn legacy_flat(value: &serde_json::Value, field: &str) -> Option<serde_json::Value> {
        if let Some(name) = field.strip_suffix("[0]") {
            return value.get(name).and_then(|items| items.get(0)).cloned();
        }
        value.get(field).cloned()
    }

    #[test]
    fn shared_resolver_agrees_with_every_replaced_copy_on_every_pin() {
        use crate::families::laya::{LAYA_ENGLISH, LAYA_MULTILINGUAL, LAYA_TYPED_DECISIONS};
        use crate::families::von::VON;

        let mut dotted_pins = vec![
            crate::families::qwen3guard::pinned_config(),
            crate::families::schema_scorer::pinned_config(),
            crate::families::encoder_nli::pinned_config(),
            crate::families::encoder_instruct_label::pinned_config(),
            VON.pinned_config(),
            VON.pinned_calibration(),
        ];
        for profile in [&LAYA_ENGLISH, &LAYA_MULTILINGUAL, &LAYA_TYPED_DECISIONS] {
            dotted_pins.push(profile.pinned_agent_config());
            dotted_pins.push(profile.pinned_encoder_config());
        }
        for pins in &dotted_pins {
            let config = testing::config_from_pins(pins);
            for (field, expected) in pins {
                let shared = resolve_config_field(&config, field);
                assert_eq!(shared, legacy_dotted(&config, field), "{field}");
                assert_eq!(shared.as_ref(), Some(expected), "{field}");
            }
        }

        let letter = crate::families::decoder_logit_letter::pinned_config();
        let config = testing::config_from_pins(&letter);
        for (field, expected) in &letter {
            let shared = resolve_config_field(&config, field);
            assert_eq!(shared, legacy_flat(&config, field), "{field}");
            assert_eq!(shared.as_ref(), Some(expected), "{field}");
        }
    }

    #[test]
    fn pinned_field_checks_report_missing_and_mismatched_fields() {
        let pins = vec![("a.b", serde_json::json!(1)), ("c", serde_json::json!("x"))];
        let ok = serde_json::json!({"a": {"b": 1}, "c": "x"});
        check_pinned_fields(&ok, &pins).expect("matching config");

        let missing = serde_json::json!({"a": {"b": 1}});
        assert!(matches!(
            check_pinned_fields(&missing, &pins),
            Err(FamilyError::ContractMismatch { field: "c", expected, actual })
                if expected == "\"x\"" && actual == "missing"
        ));

        let wrong = serde_json::json!({"a": {"b": 2}, "c": "x"});
        assert!(matches!(
            check_pinned_fields(&wrong, &pins),
            Err(FamilyError::ContractMismatch { field: "a.b", expected, actual })
                if expected == "1" && actual == "2"
        ));

        // The strict form never forgives a float drift; the tolerant form
        // forgives only parser-epsilon drift and still rejects real ones.
        let float_pins = vec![("eps", serde_json::json!(1e-5))];
        let near = serde_json::json!({"eps": 1.00005e-5});
        let far = serde_json::json!({"eps": 1.0002e-5});
        assert!(check_pinned_fields(&near, &float_pins).is_err());
        check_pinned_fields_tolerant(&near, &float_pins).expect("within epsilon");
        assert!(check_pinned_fields_tolerant(&far, &float_pins).is_err());
        // An integer pin against a float is not forgiven by the strict form
        // that qwen3 relies on.
        let int_pins = vec![("n", serde_json::json!(2))];
        assert!(check_pinned_fields(&serde_json::json!({"n": 2.0}), &int_pins).is_err());

        // A per-field comparator replaces the default for that field only.
        let by_field = check_pinned_fields_with(&wrong, &pins, |field, expected, actual| {
            field == "a.b" || expected == actual
        });
        by_field.expect("comparator accepts the overridden field");
    }

    #[test]
    fn require_file_reports_the_standard_and_labeled_messages() {
        let dir = tempfile::tempdir().expect("tempdir");
        let present = testing::put(dir.path(), "present", b"x");
        require_file(&present).expect("present file");
        // A directory is not a file.
        assert!(require_file(dir.path()).is_err());

        let absent = dir.path().join("absent");
        let error = require_file(&absent).expect_err("absent file");
        assert!(matches!(
            &error,
            FamilyError::Io { path, source }
                if path == &absent
                    && source.kind() == std::io::ErrorKind::NotFound
                    && source.to_string() == "required pinned artifact is missing"
        ));
        let error = require_file_labeled(&absent, "shard absent").expect_err("absent file");
        assert!(matches!(
            &error,
            FamilyError::Io { source, .. }
                if source.to_string() == "required pinned shard absent is missing"
        ));
    }

    #[test]
    fn sized_digest_checks_size_before_the_digest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = testing::put(dir.path(), "blob", b"abc");
        let digest = testing::sha(b"abc");
        verify_sized_digest(&path, "blob", 3, &digest, "blob size").expect("matching file");

        let error =
            verify_sized_digest(&path, "blob", 4, &digest, "blob size").expect_err("wrong size");
        assert!(matches!(
            &error,
            FamilyError::ContractMismatch { field: "blob size", expected, actual }
                if expected == "blob is 4 bytes" && actual == "3 bytes"
        ));

        // A wrong digest only surfaces once the size passes.
        let error = verify_sized_digest(&path, "blob", 3, &testing::sha(b"abd"), "blob size")
            .expect_err("wrong digest");
        assert!(matches!(error, FamilyError::DigestMismatch { .. }));

        // A missing file reports the metadata error, not the pinned message.
        let absent = dir.path().join("absent");
        let error = verify_sized_digest(&absent, "absent", 3, &digest, "blob size")
            .expect_err("absent file");
        assert!(matches!(
            &error,
            FamilyError::Io { source, .. }
                if source.kind() == std::io::ErrorKind::NotFound
                    && source.to_string() != "required pinned artifact is missing"
        ));
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

    #[test]
    fn family_control_fails_on_elapsed_deadlines_and_cancellation() {
        use std::sync::atomic::{AtomicBool, Ordering};

        // A deadline already in the past fails before the cancellation flag
        // is even consulted.
        let cancelled = Arc::new(AtomicBool::new(false));
        let control = FamilyControl::new(
            cancelled.clone(),
            Some(Instant::now() - Duration::from_millis(1)),
            4_000,
        );
        let error = control.check().expect_err("elapsed deadline must fail");
        assert!(
            matches!(error, FamilyError::DeadlineExceeded { timeout_ms: 4_000 }),
            "unexpected error: {error}"
        );

        // Cancellation fires when the deadline has not elapsed.
        let control = FamilyControl::new(
            cancelled.clone(),
            Some(Instant::now() + Duration::from_secs(60)),
            60_000,
        );
        control.check().expect("open control must pass");
        cancelled.store(true, Ordering::Release);
        let error = control.check().expect_err("cancellation must fail");
        assert!(
            matches!(error, FamilyError::Cancelled),
            "unexpected error: {error}"
        );

        // A control without a deadline still honors cancellation only.
        let control = FamilyControl::new(cancelled, None, 0);
        assert!(matches!(
            control.check().expect_err("still cancelled"),
            FamilyError::Cancelled
        ));
    }

    #[test]
    fn map_family_error_preserves_the_error_taxonomy() {
        let invalid_input =
            map_family_error("b", FamilyError::InvalidInput("bad question".to_owned()));
        assert!(
            matches!(
                invalid_input,
                EngineError::Unsupported { ref backend, ref message }
                    if backend == "b" && message.contains("bad question")
            ),
            "unexpected error: {invalid_input}"
        );

        let deadline = map_family_error("b", FamilyError::DeadlineExceeded { timeout_ms: 1_234 });
        assert!(
            matches!(
                deadline,
                EngineError::DeadlineExceeded { ref backend, timeout_ms: 1_234 }
                    if backend == "b"
            ),
            "unexpected error: {deadline}"
        );

        let other = map_family_error("b", FamilyError::Cancelled);
        assert!(
            matches!(other, EngineError::Backend { ref backend, .. } if backend == "b"),
            "unexpected error: {other}"
        );
    }

    #[tokio::test]
    async fn execution_slot_acquires_and_recovers() {
        let slots = Arc::new(Semaphore::new(1));
        let permit = acquire_execution_slot(slots.clone(), None, "test-family", 0)
            .await
            .expect("open slot must be acquirable");
        assert_eq!(slots.available_permits(), 0);
        drop(permit);
        assert_eq!(slots.available_permits(), 1);
    }

    #[tokio::test]
    async fn execution_slot_fails_closed_when_admission_shuts_down() {
        let slots = Arc::new(Semaphore::new(0));
        slots.close();
        let error = acquire_execution_slot(slots, None, "test-family", 0)
            .await
            .expect_err("a closed pool must fail closed");
        assert!(
            matches!(error, EngineError::Backend { ref backend, .. } if backend == "test-family"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn execution_slot_reports_deadlines_elapsed_while_queued() {
        let slots = Arc::new(Semaphore::new(1));
        let held = slots
            .clone()
            .acquire_owned()
            .await
            .expect("park the only slot");

        let error = acquire_execution_slot(
            slots.clone(),
            Some(Instant::now() + Duration::from_millis(20)),
            "test-family",
            20,
        )
        .await
        .expect_err("an elapsed deadline must fail while queued");
        assert!(
            matches!(
                error,
                EngineError::DeadlineExceeded { ref backend, timeout_ms: 20 }
                    if backend == "test-family"
            ),
            "unexpected error: {error}"
        );

        // Releasing the slot makes the next acquisition succeed again.
        drop(held);
        let permit = acquire_execution_slot(slots, None, "test-family", 0)
            .await
            .expect("slot recovers after the deadline failure");
        drop(permit);
    }

    #[test]
    fn cancellation_guard_signals_only_when_abandoned() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let guard = CancellationOnDrop::new(cancelled.clone(), "test_cancellations_total");
        drop(guard);
        assert!(cancelled.load(Ordering::Acquire));

        let completed = Arc::new(AtomicBool::new(false));
        let mut guard = CancellationOnDrop::new(completed.clone(), "test_cancellations_total");
        guard.disarm();
        drop(guard);
        assert!(!completed.load(Ordering::Acquire));
    }

    #[test]
    fn gate_metric_names_keep_their_exported_prefixes() {
        for (metrics, prefix) in [
            (FAMILY_GATE_METRICS, "openkind_family_"),
            (NATIVE_GATE_METRICS, "openkind_native_"),
        ] {
            for name in [
                metrics.request_seconds,
                metrics.requests_total,
                metrics.queue_wait_seconds,
                metrics.execution_seconds,
                metrics.cancellations_total,
            ] {
                assert!(name.starts_with(prefix), "{name}");
            }
        }
        assert_eq!(
            NATIVE_GATE_METRICS.cancellations_total,
            "openkind_native_cancellations_total"
        );
    }

    #[test]
    fn millis_saturate_instead_of_wrapping() {
        assert_eq!(saturating_millis(Duration::from_millis(1_500)), 1_500);
        assert_eq!(saturating_millis(Duration::MAX), u64::MAX);
    }

    #[test]
    fn outcome_labels_cover_every_engine_error_class() {
        let backend = || "b".to_owned();
        let ok: EngineResult<()> = Ok(());
        assert_eq!(outcome_label(&ok), "ok");
        let cases: [(EngineError, &str); 5] = [
            (
                EngineError::DeadlineExceeded {
                    backend: backend(),
                    timeout_ms: 1,
                },
                "deadline",
            ),
            (
                EngineError::Overloaded {
                    backend: backend(),
                    retry_after_ms: 1,
                },
                "overloaded",
            ),
            (
                EngineError::Unsupported {
                    backend: backend(),
                    message: String::new(),
                },
                "invalid",
            ),
            (
                EngineError::Invalid(openkind_core::ValidationError::NoQuestions),
                "invalid",
            ),
            (
                EngineError::Backend {
                    backend: backend(),
                    message: String::new(),
                },
                "backend_error",
            ),
        ];
        for (error, label) in cases {
            assert_eq!(outcome_label::<()>(&Err(error)), label);
        }
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
        .expect("permits released after the blocking work completed");
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
            panic!("simulated worker failure");
        })
        .await;
        assert!(failed.is_err());
        assert_eq!(execution.available_permits(), 1);
        assert_eq!(admission.available_permits(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn gate_rejects_past_the_queue_and_reports_task_failure() {
        let gate = Arc::new(AdmissionGate::new(1, 0, 25, FAMILY_GATE_METRICS));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let busy_gate = Arc::clone(&gate);
        let busy = tokio::spawn(async move {
            busy_gate
                .run("gate-test", Instant::now(), None, 0, move || {
                    let _ = started_tx.send(());
                    let _ = release_rx.recv();
                    7_u32
                })
                .await
        });
        started_rx.await.expect("the first request holds the slot");

        let rejected = gate
            .run("gate-test", Instant::now(), None, 0, || 1_u32)
            .await
            .expect_err("a full gate rejects immediately");
        assert!(
            matches!(rejected, EngineError::Overloaded { ref backend, retry_after_ms: 25 } if backend == "gate-test"),
            "unexpected error: {rejected}"
        );

        release_tx.send(()).expect("release the parked request");
        assert_eq!(busy.await.expect("join").expect("parked request"), 7);

        // A panicking task is a backend error and frees both permits.
        let failed = gate
            .run("gate-test", Instant::now(), None, 0, || -> u32 {
                panic!("simulated worker failure")
            })
            .await
            .expect_err("a panicking task fails");
        assert!(
            matches!(failed, EngineError::Backend { ref backend, .. } if backend == "gate-test"),
            "unexpected error: {failed}"
        );
        assert_eq!(gate.execution_slots.available_permits(), 1);
        assert_eq!(gate.admission_slots.available_permits(), 1);
    }
}
