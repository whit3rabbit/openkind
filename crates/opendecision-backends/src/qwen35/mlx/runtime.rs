//! Serialized MLX execution context, runtime identity, and memory policy.
//!
//! MLX streams are **thread-affine**: a stream's GPU command encoder is
//! registered per thread, and mlx-c resolves evaluation through the calling
//! thread's stream registry (`mlx_stream_new_thread_unsafe` is the one
//! cross-thread exception, whose synchronization burden falls on the caller;
//! it is the designated 3M.5 refinement once benchmarks show encoder
//! contention). The registry does not migrate across threads: a lazy graph
//! records the stream of the thread that created it, and another thread
//! cannot resolve that stream's encoder ("There is no Stream(gpu, 0) in
//! current thread"). This backend therefore uses one simple discipline:
//!
//! - every MLX operation — graph creation, evaluation, synchronization, and
//!   state advancement — runs while holding the process-wide execution mutex,
//!   on the calling thread's default GPU stream (which mlx-c creates lazily
//!   per thread on first use);
//!
//! - threads never touch MLX concurrently, so per-thread default streams
//!   are mutually exclusive by construction, and arrays that cross a thread
//!   boundary do so **fully materialized**: weights are evaluated on the
//!   loading thread before the backbone is shared (see
//!   `MlxDecoderLayer::materialize`), and continuation state is evaluated at
//!   executor boundaries. Engine forwards may run on blocking-pool threads
//!   that never loaded the model, so no unevaluated lazy graph may ever
//!   cross a boundary.
//!
//! Continuation state produced through this runtime is fully materialized
//! (evaluated) at executor boundaries; unevaluated lazy graphs never cross
//! the boundary.

use std::ffi::CStr;
use std::panic::AssertUnwindSafe;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, MutexGuard, OnceLock,
};

use mlx_rs::Stream;
use mlx_rs::{memory, with_stream, Array};

use super::MlxError;

/// Pinned MLX core version supplied by the vendored mlx-c release.
///
/// `mlx-sys` 0.6.0 vendors mlx-c `v0.6.0-7-gc74db53`, which is MLX 0.32.2.
/// The linked runtime is verified against this constant at construction and
/// construction fails if it disagrees (fail-closed).
pub const MLX_CORE_VERSION: &str = "0.32.2";

/// Conservative default bound on MLX's inactive allocation cache.
///
/// SemIf uses 256 MiB for the same reason: without a bound, MLX's cache can
/// retain almost all unified memory after variable-sized workloads. The
/// default is a safety bound, not a measured optimum; tuning it is a 3M.8
/// measurement task.
pub const DEFAULT_INACTIVE_CACHE_LIMIT_BYTES: usize = 256 * 1024 * 1024;

/// Runtime settings for the MLX backend.
#[derive(Debug, Clone)]
pub struct MlxRuntimeConfig {
    /// Process-wide bound applied to MLX's inactive allocator cache
    /// (`set_cache_limit`). All runtimes in one process must use the same
    /// value because MLX exposes one allocator cache.
    pub inactive_cache_limit_bytes: usize,
}

impl Default for MlxRuntimeConfig {
    fn default() -> Self {
        Self {
            inactive_cache_limit_bytes: DEFAULT_INACTIVE_CACHE_LIMIT_BYTES,
        }
    }
}

/// Serialized MLX execution context.
///
/// Every MLX operation in this backend runs through [`MlxRuntime::execute`],
/// which takes the execution lock before any array work and releases it
/// only after the closure returns. Treat this as the single entry point for
/// array work; do not call `mlx_rs` free functions from backend code.
pub struct MlxRuntime {
    config: MlxRuntimeConfig,
    version: String,
    toolchain_identity: String,
    bf16_qualified: AtomicBool,
}

impl MlxRuntime {
    /// Construct the runtime: verify the linked MLX version against the
    /// pinned constant and bound the allocator cache.
    ///
    /// This also initializes the calling thread's default GPU stream; the
    /// runtime must be constructed before any other thread performs MLX
    /// work, which the engine's load path guarantees.
    pub fn new(config: MlxRuntimeConfig) -> Result<Self, MlxError> {
        let _guard = lock_global_execution()?;
        let stream = Stream::thread_local_or_default();
        let version = query_mlx_version(&stream)?;
        if version.trim() != MLX_CORE_VERSION {
            return Err(MlxError::VersionMismatch {
                expected: MLX_CORE_VERSION,
                found: version,
            });
        }
        let toolchain_identity = query_toolchain_identity()?;
        let configured_limit = *CACHE_LIMIT.get_or_init(|| config.inactive_cache_limit_bytes);
        if configured_limit != config.inactive_cache_limit_bytes {
            return Err(MlxError::InvalidState(format!(
                "MLX cache limit is already configured to {configured_limit} bytes"
            )));
        }
        with_stream(&stream, || memory::set_cache_limit(configured_limit))
            .map_err(|error| operation_error("set_cache_limit", error))?;
        Ok(Self {
            config,
            version,
            toolchain_identity,
            bf16_qualified: AtomicBool::new(false),
        })
    }

    /// Runtime configuration in effect.
    #[must_use]
    pub fn config(&self) -> &MlxRuntimeConfig {
        &self.config
    }

    /// Version string reported by the linked MLX runtime.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Xcode/Metal toolchain identity used by the arithmetic state identity.
    #[must_use]
    pub fn toolchain_identity(&self) -> &str {
        &self.toolchain_identity
    }

    /// Run the required native-BF16 runtime preflight once.
    ///
    /// This is deliberately explicit because a native-BF16 model load must
    /// never silently become the qualification run. The full Gate A stress
    /// example remains the promotion check; this preflight verifies that the
    /// linked runtime can construct, evaluate, and read BF16 arithmetic on
    /// the current process before any model tensors are loaded.
    pub fn qualify_bf16(&self) -> Result<(), MlxError> {
        if self.bf16_qualified.load(Ordering::Acquire) {
            return Ok(());
        }
        self.execute(|| {
            let values = [
                half::bf16::from_f32(-1.0),
                half::bf16::from_f32(0.5),
                half::bf16::from_f32(2.0),
                half::bf16::from_f32(4.0),
            ];
            let input = Array::from_slice(&values, &[2, 2]);
            if input.dtype() != mlx_rs::Dtype::Bfloat16 {
                return Err(MlxError::InvalidState(
                    "native BF16 preflight constructed a non-BF16 input".to_owned(),
                ));
            }
            let product = input
                .clone()
                .matmul(&input)
                .map_err(|error| MlxError::Operation {
                    operation: "BF16 preflight matmul",
                    message: error.to_string(),
                })?;
            let normalized = mlx_rs::fast::rms_norm(&input, None, 1e-6).map_err(|error| {
                MlxError::Operation {
                    operation: "BF16 preflight rms norm",
                    message: error.to_string(),
                }
            })?;
            product.eval().map_err(|error| MlxError::Operation {
                operation: "BF16 preflight eval",
                message: error.to_string(),
            })?;
            normalized.eval().map_err(|error| MlxError::Operation {
                operation: "BF16 preflight norm eval",
                message: error.to_string(),
            })?;
            let product_values =
                product
                    .to_vec_cast::<f32>()
                    .map_err(|error| MlxError::Operation {
                        operation: "BF16 preflight read",
                        message: error.to_string(),
                    })?;
            let normalized_values =
                normalized
                    .to_vec_cast::<f32>()
                    .map_err(|error| MlxError::Operation {
                        operation: "BF16 preflight norm read",
                        message: error.to_string(),
                    })?;
            if product_values
                .iter()
                .chain(normalized_values.iter())
                .any(|value| !value.is_finite())
            {
                return Err(MlxError::InvalidState(
                    "native BF16 preflight produced a non-finite value".to_owned(),
                ));
            }
            Ok(())
        })??;
        self.bf16_qualified.store(true, Ordering::Release);
        Ok(())
    }

    /// Whether [`MlxRuntime::qualify_bf16`] has passed in this process.
    #[must_use]
    pub fn bf16_qualified(&self) -> bool {
        self.bf16_qualified.load(Ordering::Acquire)
    }

    /// Run `f` under the execution lock.
    ///
    /// Inside `f`, use ordinary `mlx_rs` operations: they resolve to the
    /// calling thread's default GPU stream, which is safe because the lock
    /// guarantees no other thread is executing MLX work concurrently.
    pub fn execute<T>(&self, f: impl FnOnce() -> T) -> Result<T, MlxError> {
        let _guard = self.lock_execution()?;
        std::panic::catch_unwind(AssertUnwindSafe(f)).map_err(|payload| MlxError::Operation {
            operation: "MLX execution",
            message: panic_message(payload),
        })
    }

    /// Block until all work already enqueued on the calling thread's stream
    /// completes.
    ///
    /// Meaningful when called from the thread that enqueued the work (the
    /// executor-boundary case); takes the execution lock so it cannot race
    /// with other MLX users.
    pub fn synchronize(&self) -> Result<(), MlxError> {
        let _guard = self.lock_execution()?;
        let stream = Stream::thread_local_or_default();
        let status = unsafe { mlx_sys::mlx_synchronize(stream.as_ptr()) };
        if status != 0 {
            return Err(MlxError::Operation {
                operation: "mlx_synchronize",
                message: format!("status {status}"),
            });
        }
        Ok(())
    }

    /// MLX active (allocated, referenced) memory in bytes.
    pub fn active_memory_bytes(&self) -> Result<usize, MlxError> {
        let _guard = self.lock_execution()?;
        memory::active_memory().map_err(|error| operation_error("active_memory", error))
    }

    /// MLX inactive allocator cache size in bytes.
    pub fn cache_memory_bytes(&self) -> Result<usize, MlxError> {
        let _guard = self.lock_execution()?;
        memory::cache_memory().map_err(|error| operation_error("cache_memory", error))
    }

    /// Peak active MLX memory in bytes since the last reset.
    pub fn peak_memory_bytes(&self) -> Result<usize, MlxError> {
        let _guard = self.lock_execution()?;
        memory::peak_memory().map_err(|error| operation_error("peak_memory", error))
    }

    /// Reset the peak-active-memory watermark.
    pub fn reset_peak_memory(&self) -> Result<(), MlxError> {
        let _guard = self.lock_execution()?;
        memory::reset_peak_memory().map_err(|error| operation_error("reset_peak_memory", error))
    }

    /// Snapshot of MLX memory telemetry at one instant.
    #[must_use]
    pub fn memory_snapshot(&self) -> MlxMemorySnapshot {
        let Ok(_guard) = self.lock_execution() else {
            return MlxMemorySnapshot::default();
        };
        MlxMemorySnapshot {
            active_bytes: memory::active_memory().ok(),
            cache_bytes: memory::cache_memory().ok(),
            peak_bytes: memory::peak_memory().ok(),
        }
    }

    fn lock_execution(&self) -> Result<MutexGuard<'static, ()>, MlxError> {
        lock_global_execution()
    }
}

fn lock_global_execution() -> Result<MutexGuard<'static, ()>, MlxError> {
    global_execution().lock().map_err(|_| MlxError::Operation {
        operation: "execution lock",
        message: "MLX execution mutex poisoned by a panicking critical section".into(),
    })
}

fn global_execution() -> &'static Mutex<()> {
    static EXECUTION: OnceLock<Mutex<()>> = OnceLock::new();
    EXECUTION.get_or_init(|| Mutex::new(()))
}

static CACHE_LIMIT: OnceLock<usize> = OnceLock::new();

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "MLX operation panicked".to_owned()
    }
}

/// Point-in-time MLX memory telemetry.
///
/// These are MLX allocator values, not total unified-memory pressure; pair
/// them with process RSS (reported separately) for admission decisions.
#[derive(Debug, Clone, Copy, Default)]
pub struct MlxMemorySnapshot {
    /// Active allocated MLX memory, if the query succeeded.
    pub active_bytes: Option<usize>,
    /// Inactive allocator cache size, if the query succeeded.
    pub cache_bytes: Option<usize>,
    /// Peak active MLX memory, if the query succeeded.
    pub peak_bytes: Option<usize>,
}

/// Shared runtime handle used by executors and branch states.
pub type SharedMlxRuntime = Arc<MlxRuntime>;

fn operation_error(operation: &'static str, error: mlx_rs::error::Exception) -> MlxError {
    MlxError::Operation {
        operation,
        message: error.to_string(),
    }
}

fn query_mlx_version(stream: &Stream) -> Result<String, MlxError> {
    with_stream(stream, || -> Result<String, MlxError> {
        // SAFETY: `c_string` is a freshly created handle owned by this scope;
        // every use below happens before `mlx_string_free` releases it.
        unsafe {
            let mut c_string = mlx_sys::mlx_string_new();
            let status = mlx_sys::mlx_version(&mut c_string as *mut _);
            if status != 0 {
                mlx_sys::mlx_string_free(c_string);
                return Err(MlxError::Operation {
                    operation: "mlx_version",
                    message: format!("status {status}"),
                });
            }
            let data = mlx_sys::mlx_string_data(c_string);
            let text = if data.is_null() {
                String::new()
            } else {
                CStr::from_ptr(data).to_string_lossy().into_owned()
            };
            mlx_sys::mlx_string_free(c_string);
            Ok(text)
        }
    })
}

fn query_toolchain_identity() -> Result<String, MlxError> {
    let xcode = command_identity("xcodebuild", &["-version"])?;
    let metal = command_identity("xcrun", &["--sdk", "macosx", "metal", "--version"])?;
    Ok(format!("xcode={xcode};metal={metal}"))
}

fn command_identity(program: &str, arguments: &[&str]) -> Result<String, MlxError> {
    let output = std::process::Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| MlxError::Operation {
            operation: "toolchain identity",
            message: format!("failed to run {program}: {error}"),
        })?;
    if !output.status.success() {
        return Err(MlxError::Operation {
            operation: "toolchain identity",
            message: format!("{program} exited with {}", output.status),
        });
    }
    let raw = if output.stdout.is_empty() {
        &output.stderr
    } else {
        &output.stdout
    };
    let identity = String::from_utf8_lossy(raw)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("|");
    if identity.is_empty() {
        return Err(MlxError::Operation {
            operation: "toolchain identity",
            message: format!("{program} returned no version information"),
        });
    }
    Ok(identity)
}
