//! ONNX Runtime execution for exported family artifacts (feature `onnx`).
//!
//! Every surveyed family that has an ONNX-representable readout ships a
//! family adapter that loads `model.onnx` from the profile root and executes
//! it through ONNX Runtime, reusing the family's pinned tokenizer and answer
//! mapping. Execution is optional at build time (`onnx` feature) and
//! fail-closed at load time: a missing runtime library, a missing artifact,
//! an unexpected input/output signature, or an unavailable CUDA execution
//! provider aborts the engine load instead of silently falling back.
//!
//! The runtime library itself is resolved with `load-dynamic` semantics:
//! an explicit path, the `ORT_DYLIB_PATH` environment variable, or the
//! system library search path. Builds stay offline — nothing downloads at
//! build time. See `docs/ONNX.md` for the artifact and runtime contract.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::value::{TensorElementType, ValueType};
use thiserror::Error;

mod artifact;

/// A tensor's name, element type, and compatible shape in a family export.
/// A dimension of `-1` accepts any nonnegative or dynamic dimension.
pub struct OnnxTensorSpec<'a> {
    name: &'a str,
    element_type: TensorElementType,
    dimensions: &'a [i64],
}

impl<'a> OnnxTensorSpec<'a> {
    /// Integer input contract used by token IDs, masks, and marker positions.
    #[must_use]
    pub fn i64(name: &'a str, dimensions: &'a [i64]) -> Self {
        Self {
            name,
            element_type: TensorElementType::Int64,
            dimensions,
        }
    }

    /// FP32 readout contract.
    #[must_use]
    pub fn f32(name: &'a str, dimensions: &'a [i64]) -> Self {
        Self {
            name,
            element_type: TensorElementType::Float32,
            dimensions,
        }
    }

    fn matches(&self, value_type: &ValueType) -> bool {
        let ValueType::Tensor { ty, shape, .. } = value_type else {
            return false;
        };
        *ty == self.element_type
            && shape.len() == self.dimensions.len()
            && shape
                .iter()
                .zip(self.dimensions)
                .all(|(&actual, &expected)| {
                    actual == -1 || (actual >= 0 && (expected == -1 || actual == expected))
                })
    }
}

/// Errors raised while loading or executing ONNX family artifacts.
#[derive(Debug, Error)]
pub enum OnnxError {
    /// The export manifest or a pinned graph/weight digest is invalid.
    #[error("ONNX artifact integrity failure for `{path}`: {message}")]
    Integrity {
        /// Manifest, graph, or weight file that failed verification.
        path: PathBuf,
        /// Reason verification failed.
        message: String,
    },
    /// The ONNX artifact is missing or unreadable.
    #[error("failed to read ONNX artifact `{path}`: {source}")]
    Io {
        /// Artifact path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },

    /// The ONNX Runtime library could not be loaded.
    #[error("failed to load the ONNX Runtime library: {0}")]
    RuntimeLibrary(String),

    /// ONNX Runtime reported an error during session setup or execution.
    #[error("ONNX Runtime failure: {0}")]
    Runtime(String),

    /// The loaded runtime cannot provide the requested CUDA execution
    /// provider, or the binary was built without `onnx-cuda`.
    #[error("ONNX CUDA execution provider unavailable: {0}")]
    CudaProvider(String),

    /// The artifact input/output signature does not match the family
    /// contract.
    #[error("ONNX signature mismatch for `{artifact}`: {message}")]
    Signature {
        /// Artifact path.
        artifact: PathBuf,
        /// What did not match.
        message: String,
    },
}

impl<R> From<ort::Error<R>> for OnnxError {
    fn from(error: ort::Error<R>) -> Self {
        Self::Runtime(error.to_string())
    }
}

/// Execution-provider selection for one ONNX session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxAcceleration {
    /// CPU execution provider (always available in every runtime build).
    Cpu,
    /// CUDA execution provider on the given device ordinal (feature
    /// `onnx-cuda`; the loaded runtime must ship CUDA support).
    #[cfg(feature = "onnx-cuda")]
    Cuda {
        /// Zero-based CUDA device ordinal.
        device_id: usize,
    },
}

impl OnnxAcceleration {
    /// Map from a [`FamilyExecution`](crate::device::FamilyExecution) ONNX
    /// selection.
    ///
    /// # Errors
    /// Fails closed when a CUDA ordinal is requested but the binary lacks
    /// the `onnx-cuda` feature.
    pub fn from_onnx_execution(device_id: Option<usize>) -> Result<Self, OnnxError> {
        match device_id {
            None => Ok(Self::Cpu),
            #[cfg(feature = "onnx-cuda")]
            Some(device_id) => Ok(Self::Cuda { device_id }),
            #[cfg(not(feature = "onnx-cuda"))]
            Some(_) => Err(OnnxError::CudaProvider(
                "this binary was built without the `onnx-cuda` feature; rebuild with it or select the CPU execution provider".to_owned(),
            )),
        }
    }

    /// Identifier fragment shared with [`FamilyExecution`](crate::device::FamilyExecution).
    #[must_use]
    pub fn id_fragment(self) -> String {
        match self {
            Self::Cpu => "onnx-cpu".to_owned(),
            #[cfg(feature = "onnx-cuda")]
            Self::Cuda { device_id } => format!("onnx-cuda:{device_id}"),
        }
    }
}

/// Runtime setup for ONNX sessions.
#[derive(Debug, Clone, Default)]
pub struct OnnxRuntimeSettings {
    /// Explicit path to the ONNX Runtime shared library. When `None`, ort
    /// resolves `ORT_DYLIB_PATH` or the default library names.
    pub dylib: Option<PathBuf>,
    /// Intra-op thread count for CPU execution. `None` lets ONNX Runtime
    /// choose its default.
    pub intra_threads: Option<usize>,
}

impl OnnxRuntimeSettings {
    /// Settings that defer library resolution to `ORT_DYLIB_PATH` or the
    /// system search path.
    #[must_use]
    pub fn system() -> Self {
        Self::default()
    }
}

/// Standard library names ONNX Runtime ships as, per platform.
#[cfg(target_os = "macos")]
const DYLIB_CANDIDATES: &[&str] = &["libonnxruntime.dylib", "libonnxruntime.2.dylib"];
#[cfg(target_os = "windows")]
const DYLIB_CANDIDATES: &[&str] = &["onnxruntime.dll"];
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const DYLIB_CANDIDATES: &[&str] = &["libonnxruntime.so"];

/// Standard directories probed when no explicit library path is configured.
/// Windows resolves `onnxruntime.dll` through the system search path
/// (executable directory, system directories, `PATH`) instead.
#[cfg(unix)]
fn dylib_search_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        directories.push(home.join(".local/lib"));
        directories.push(home.join("lib"));
    }
    for prefix in ["/opt/homebrew", "/usr/local", "/usr"] {
        let prefix = PathBuf::from(prefix);
        directories.push(prefix.join("lib"));
        if cfg!(target_os = "linux") {
            directories.push(prefix.join("lib/x86_64-linux-gnu"));
            directories.push(prefix.join("lib/aarch64-linux-gnu"));
        }
    }
    directories
}

#[cfg(not(unix))]
fn dylib_search_directories() -> Vec<PathBuf> {
    Vec::new()
}

/// Resolve the ONNX Runtime library path for detection and diagnostics.
///
/// Order: explicit setting, `ORT_DYLIB_PATH`, then standard library
/// directories. Returns `None` when nothing resolvable is found; this is a
/// detection helper and performs no loading.
#[must_use]
pub fn resolve_dylib_path(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH").filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path));
    }
    dylib_search_directories()
        .into_iter()
        .flat_map(|directory| {
            DYLIB_CANDIDATES
                .iter()
                .map(move |name| directory.join(name))
        })
        .find(|candidate| candidate.is_file())
}

/// Guard ensuring the process-global ONNX Runtime environment is committed
/// exactly once, before the first session is created.
static ENVIRONMENT: OnceLock<Result<(), String>> = OnceLock::new();

fn ensure_environment(settings: &OnnxRuntimeSettings) -> Result<(), OnnxError> {
    ENVIRONMENT
        .get_or_init(|| initialize_environment(settings).map_err(|error| error.to_string()))
        .as_ref()
        .copied()
        .map_err(|message| OnnxError::RuntimeLibrary(message.clone()))
}

fn initialize_environment(settings: &OnnxRuntimeSettings) -> Result<(), OnnxError> {
    let path = resolve_dylib_path(settings.dylib.as_deref())
        .unwrap_or_else(|| PathBuf::from(DYLIB_CANDIDATES[0]));
    // Explicit loading returns a typed error. Lazy ort initialization would
    // panic on a missing library, and it ignores our directory search.
    ort::init_from(&path)
        .map_err(|error| OnnxError::RuntimeLibrary(error.to_string()))?
        .commit();
    Ok(())
}

/// A committed ONNX Runtime session behind the family adapter contract.
///
/// Sessions are wrapped in a mutex because ONNX Runtime's `Run` call is
/// exclusive; family engines already bound evaluation concurrency with their
/// admission limits, so contention here is bounded by the same limits.
pub struct OnnxModel {
    session: Mutex<ort::session::Session>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    artifact_sha256: String,
}

impl OnnxModel {
    /// Commit the shared runtime environment (once) and load `model_path`.
    ///
    /// `expected_inputs` and `expected_outputs` are the family's pinned
    /// input/output names; the load fails closed when the artifact exports a
    /// different signature.
    ///
    /// # Errors
    /// Returns [`OnnxError`] for library load failures, artifact read
    /// failures, provider unavailability, and signature mismatches.
    pub fn load(
        model_path: &Path,
        settings: &OnnxRuntimeSettings,
        acceleration: OnnxAcceleration,
        expected_inputs: &[OnnxTensorSpec<'_>],
        expected_outputs: &[OnnxTensorSpec<'_>],
    ) -> Result<Self, OnnxError> {
        if !model_path.is_file() {
            return Err(OnnxError::Io {
                path: model_path.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "required ONNX artifact is missing; see docs/ONNX.md for the artifact contract",
                ),
            });
        }
        let artifact_sha256 = artifact::verify(model_path)?;
        ensure_environment(settings)?;

        let mut builder = ort::session::Session::builder()?;
        builder = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)?;
        if let Some(threads) = settings.intra_threads {
            builder = builder.with_intra_threads(threads)?;
        }
        match acceleration {
            OnnxAcceleration::Cpu => {
                // A downstream user may configure process-global providers.
                // Explicit CPU selection must not inherit a GPU provider.
                builder = builder.with_execution_providers([ort::ep::CPU::default()
                    .build()
                    .error_on_failure()])?;
            }
            #[cfg(feature = "onnx-cuda")]
            OnnxAcceleration::Cuda { device_id } => {
                use ort::ep::{ExecutionProvider, CUDA};
                if !CUDA::default().is_available().unwrap_or(false) {
                    return Err(OnnxError::CudaProvider(format!(
                        "the loaded ONNX Runtime does not ship CUDA support (device {device_id})"
                    )));
                }
                builder = builder.with_execution_providers([CUDA::default()
                    .with_device_id(i32::try_from(device_id).map_err(|_| {
                        OnnxError::CudaProvider(format!("device ordinal {device_id} exceeds i32"))
                    })?)
                    .build()
                    .error_on_failure()])?;
            }
        }
        let session = builder.commit_from_file(model_path)?;

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|outlet| outlet.name().to_owned())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|outlet| outlet.name().to_owned())
            .collect();
        if input_names.len() != expected_inputs.len()
            || output_names.len() != expected_outputs.len()
        {
            return Err(OnnxError::Signature {
                artifact: model_path.to_path_buf(),
                message: format!(
                    "expected {} inputs and {} outputs, got inputs {input_names:?} \
                     and outputs {output_names:?}",
                    expected_inputs.len(),
                    expected_outputs.len()
                ),
            });
        }
        for (outlets, expected) in [
            (session.inputs(), expected_inputs),
            (session.outputs(), expected_outputs),
        ] {
            for specification in expected {
                let outlet = outlets
                    .iter()
                    .find(|outlet| outlet.name() == specification.name);
                if !outlet.is_some_and(|outlet| specification.matches(outlet.dtype())) {
                    return Err(OnnxError::Signature {
                        artifact: model_path.to_path_buf(),
                        message: format!(
                            "`{}` must be {:?} {:?}, got {:?}",
                            specification.name,
                            specification.element_type,
                            specification.dimensions,
                            outlet.map(|outlet| outlet.dtype())
                        ),
                    });
                }
            }
        }

        Ok(Self {
            session: Mutex::new(session),
            input_names,
            output_names,
            artifact_sha256,
        })
    }

    /// Digest of the verified ONNX graph. External weights are pinned by
    /// the companion export manifest too.
    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    /// Exported input names, in artifact order.
    #[must_use]
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Exported output names, in artifact order.
    #[must_use]
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Run one session step and read the results through `read_outputs`
    /// while the outputs are live.
    ///
    /// `inputs` maps artifact input names to tensors; every required input
    /// must be present, and no extra names are tolerated.
    ///
    /// # Errors
    /// Returns [`OnnxError`] for ONNX Runtime failures and input-set
    /// mismatches.
    pub fn run<T>(
        &self,
        inputs: Vec<(String, ort::value::DynValue)>,
        read_outputs: impl FnOnce(&ort::session::SessionOutputs<'_>) -> Result<T, OnnxError>,
    ) -> Result<T, OnnxError> {
        let supplied: Vec<&str> = inputs.iter().map(|(name, _)| name.as_str()).collect();
        for name in &self.input_names {
            if !supplied.contains(&name.as_str()) {
                return Err(OnnxError::Signature {
                    artifact: PathBuf::new(),
                    message: format!("session input `{name}` was not supplied"),
                });
            }
        }
        if supplied.len() != self.input_names.len() {
            return Err(OnnxError::Signature {
                artifact: PathBuf::new(),
                message: format!(
                    "session inputs {supplied:?} do not match artifact inputs {:?}",
                    self.input_names
                ),
            });
        }
        let mut session = self
            .session
            .lock()
            .map_err(|_| OnnxError::Runtime("onnx session mutex poisoned".to_owned()))?;
        let outputs = session.run(inputs)?;
        read_outputs(&outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceleration_from_execution_maps_provider_selection() {
        assert!(matches!(
            OnnxAcceleration::from_onnx_execution(None).expect("cpu"),
            OnnxAcceleration::Cpu
        ));
        #[cfg(feature = "onnx-cuda")]
        assert!(matches!(
            OnnxAcceleration::from_onnx_execution(Some(1)).expect("cuda"),
            OnnxAcceleration::Cuda { device_id: 1 }
        ));
        #[cfg(not(feature = "onnx-cuda"))]
        {
            let error = OnnxAcceleration::from_onnx_execution(Some(1))
                .expect_err("cuda requires the onnx-cuda feature");
            assert!(matches!(error, OnnxError::CudaProvider(_)));
        }
    }

    #[test]
    fn acceleration_fragments_match_execution_fragments() {
        assert_eq!(OnnxAcceleration::Cpu.id_fragment(), "onnx-cpu");
        #[cfg(feature = "onnx-cuda")]
        assert_eq!(
            OnnxAcceleration::Cuda { device_id: 2 }.id_fragment(),
            "onnx-cuda:2"
        );
    }

    #[test]
    fn dylib_resolution_prefers_the_explicit_path() {
        let explicit = Path::new("/nonexistent/onnxruntime.dylib");
        assert_eq!(
            resolve_dylib_path(Some(explicit)),
            Some(explicit.to_path_buf())
        );
    }

    #[test]
    fn dylib_resolution_scans_standard_directories() {
        // Resolution must return either a found file or None; it must never
        // panic or invent a nonexistent path outside the search list.
        if let Some(path) = resolve_dylib_path(None) {
            assert!(path.is_absolute() || std::env::var_os("ORT_DYLIB_PATH").is_some());
        }
    }

    #[test]
    fn missing_runtime_returns_consistent_errors_for_concurrent_loads() {
        const CHILD: &str = "OPENKIND_ONNX_MISSING_RUNTIME_TEST";
        if std::env::var_os(CHILD).is_some() {
            let directory = tempfile::tempdir().expect("tempdir");
            let settings = OnnxRuntimeSettings {
                dylib: Some(directory.path().join("missing-runtime-library")),
                intra_threads: None,
            };
            let errors = std::thread::scope(|scope| {
                let threads: Vec<_> = (0..8)
                    .map(|_| scope.spawn(|| ensure_environment(&settings)))
                    .collect();
                threads
                    .into_iter()
                    .map(|thread| match thread.join().expect("load must not panic") {
                        Err(OnnxError::RuntimeLibrary(message)) => message,
                        _ => panic!("missing library must return a runtime-library error"),
                    })
                    .collect::<Vec<_>>()
            });
            assert!(errors.iter().all(|message| message == &errors[0]));
            assert!(errors[0].contains("missing-runtime-library"));
            return;
        }
        // ort's library and environment are process-global. Isolate a failed
        // first load so it cannot contaminate tests that use the real runtime.
        let result = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "onnx::tests::missing_runtime_returns_consistent_errors_for_concurrent_loads",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .expect("spawn isolated test");
        assert!(
            result.status.success(),
            "child failed: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }

    /// Minimal protobuf wire-format writer for the hand-encoded test graph.
    #[cfg(test)]
    mod proto {
        pub(crate) fn push_varint(buffer: &mut Vec<u8>, mut value: u64) {
            loop {
                let byte = (value & 0x7f) as u8;
                value >>= 7;
                if value == 0 {
                    buffer.push(byte);
                    break;
                }
                buffer.push(byte | 0x80);
            }
        }

        pub(crate) fn push_tag(buffer: &mut Vec<u8>, field: u32, wire_type: u8) {
            push_varint(buffer, ((u64::from(field)) << 3) | u64::from(wire_type));
        }

        pub(crate) fn push_len_delimited(buffer: &mut Vec<u8>, field: u32, payload: &[u8]) {
            push_tag(buffer, field, 2);
            push_varint(buffer, payload.len() as u64);
            buffer.extend_from_slice(payload);
        }

        pub(crate) fn push_string(buffer: &mut Vec<u8>, field: u32, value: &str) {
            push_len_delimited(buffer, field, value.as_bytes());
        }

        pub(crate) fn push_int64(buffer: &mut Vec<u8>, field: u32, value: i64) {
            push_tag(buffer, field, 0);
            push_varint(buffer, value as u64);
        }

        pub(crate) fn push_packed_int64(buffer: &mut Vec<u8>, field: u32, values: &[i64]) {
            let mut packed = Vec::new();
            for &value in values {
                push_varint(&mut packed, value as u64);
            }
            push_len_delimited(buffer, field, &packed);
        }

        pub(crate) fn push_packed_float(buffer: &mut Vec<u8>, field: u32, values: &[f32]) {
            let mut packed = Vec::new();
            for &value in values {
                packed.extend_from_slice(&value.to_le_bytes());
            }
            push_len_delimited(buffer, field, &packed);
        }

        fn tensor_type_message(elem_type: i32, shape: &[i64]) -> Vec<u8> {
            // TypeProto.tensor_type = 1
            let mut tensor = Vec::new();
            push_int64(&mut tensor, 1, i64::from(elem_type));
            // TensorShapeProto.dim = 1 (repeated Dimension{dim_value=1})
            let mut shape_message = Vec::new();
            for &dim in shape {
                let mut dimension = Vec::new();
                push_int64(&mut dimension, 1, dim);
                push_len_delimited(&mut shape_message, 1, &dimension);
            }
            push_len_delimited(&mut tensor, 2, &shape_message);
            let mut type_proto = Vec::new();
            push_len_delimited(&mut type_proto, 1, &tensor);
            type_proto
        }

        fn value_info_message(name: &str, elem_type: i32, shape: &[i64]) -> Vec<u8> {
            let mut message = Vec::new();
            push_string(&mut message, 1, name);
            push_len_delimited(&mut message, 2, &tensor_type_message(elem_type, shape));
            message
        }

        fn initializer_message(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
            let mut message = Vec::new();
            push_packed_int64(&mut message, 1, dims);
            // data_type = 1 (FLOAT)
            push_int64(&mut message, 2, 1);
            push_packed_float(&mut message, 4, data);
            push_string(&mut message, 8, name);
            message
        }

        /// Encode `y = x @ w + b` over float32 tensors:
        /// `x` input `[1, 2]`, `w` initializer `[2, 3]`, `b` initializer
        /// `[3]`, `y` output `[1, 3]`.
        pub(crate) fn tiny_gemm_model() -> Vec<u8> {
            tiny_gemm_model_with_extra_input(false)
        }

        pub(crate) fn tiny_gemm_model_with_extra_input(extra_input: bool) -> Vec<u8> {
            let mut graph = Vec::new();
            // GraphProto.node = 1: Gemm(x, w, b) -> y
            let mut node = Vec::new();
            push_string(&mut node, 1, "x");
            push_string(&mut node, 1, "w");
            push_string(&mut node, 1, "b");
            push_string(&mut node, 2, "y");
            push_string(&mut node, 4, "Gemm");
            push_string(&mut node, 3, "gemm_node");
            push_len_delimited(&mut graph, 1, &node);
            // GraphProto.name = 2
            push_string(&mut graph, 2, "tiny_gemm");
            // GraphProto.initializer = 5
            push_len_delimited(
                &mut graph,
                5,
                &initializer_message("w", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            );
            push_len_delimited(
                &mut graph,
                5,
                &initializer_message("b", &[3], &[0.0, 0.0, 0.0]),
            );
            // GraphProto.input = 11
            push_len_delimited(&mut graph, 11, &value_info_message("x", 1, &[1, 2]));
            if extra_input {
                push_len_delimited(&mut graph, 11, &value_info_message("extra", 1, &[1]));
            }
            // GraphProto.output = 12
            push_len_delimited(&mut graph, 12, &value_info_message("y", 1, &[1, 3]));

            let mut model = Vec::new();
            // ModelProto.ir_version = 1 (IR 8 = ONNX 1.13)
            push_int64(&mut model, 1, 8);
            // ModelProto.graph = 7
            push_len_delimited(&mut model, 7, &graph);
            // ModelProto.opset_import = 8: {domain: "", version: 13}
            let mut opset = Vec::new();
            push_string(&mut opset, 1, "");
            push_int64(&mut opset, 2, 13);
            push_len_delimited(&mut model, 8, &opset);
            model
        }
    }

    /// End-to-end ONNX Runtime execution over a hand-encoded Gemm graph.
    ///
    /// Skips when no ONNX Runtime library is discoverable — the same
    /// fail-soft pattern as checkpoint-gated parity replays.
    #[test]
    fn tiny_model_loads_runs_and_validates_signatures() {
        let Some(dylib) = resolve_dylib_path(None) else {
            eprintln!(
                "skipping: no ONNX Runtime library found (set ORT_DYLIB_PATH); \
                 see docs/ONNX.md"
            );
            return;
        };

        let directory = tempfile::tempdir().expect("tempdir");
        let model_path = directory.path().join("model.onnx");
        std::fs::write(&model_path, proto::tiny_gemm_model()).expect("write tiny model");
        artifact::write_test_manifest(&model_path, &[]);

        let settings = OnnxRuntimeSettings {
            dylib: Some(dylib),
            intra_threads: Some(1),
        };
        let model = OnnxModel::load(
            &model_path,
            &settings,
            OnnxAcceleration::Cpu,
            &[OnnxTensorSpec::f32("x", &[1, 2])],
            &[OnnxTensorSpec::f32("y", &[1, 3])],
        )
        .expect("load tiny gemm model");
        assert_eq!(model.input_names(), ["x"]);
        assert_eq!(model.output_names(), ["y"]);

        // Names alone cannot establish the adapter contract: wrong types,
        // dimensions, or an unsupported extra input must fail at load time.
        for inputs in [
            vec![OnnxTensorSpec::i64("x", &[1, 2])],
            vec![OnnxTensorSpec::f32("x", &[1, 4])],
            vec![
                OnnxTensorSpec::f32("x", &[1, 2]),
                OnnxTensorSpec::f32("extra", &[1]),
            ],
        ] {
            assert!(matches!(
                OnnxModel::load(
                    &model_path,
                    &settings,
                    OnnxAcceleration::Cpu,
                    &inputs,
                    &[OnnxTensorSpec::f32("y", &[1, 3])]
                ),
                Err(OnnxError::Signature { .. })
            ));
        }

        let extra_model_path = directory.path().join("extra.onnx");
        std::fs::write(
            &extra_model_path,
            proto::tiny_gemm_model_with_extra_input(true),
        )
        .expect("write model with an unsupported input");
        artifact::write_test_manifest(&extra_model_path, &[]);
        assert!(matches!(
            OnnxModel::load(
                &extra_model_path,
                &settings,
                OnnxAcceleration::Cpu,
                &[OnnxTensorSpec::f32("x", &[1, 2])],
                &[OnnxTensorSpec::f32("y", &[1, 3])]
            ),
            Err(OnnxError::Signature { .. })
        ));

        let observed = model
            .run(
                vec![(
                    "x".to_owned(),
                    ort::value::Tensor::from_array((vec![1_usize, 2], vec![1.0_f32, 2.0_f32]))
                        .expect("input tensor")
                        .into_dyn(),
                )],
                |outputs| {
                    let (shape, data) = outputs["y"].try_extract_tensor::<f32>()?;
                    assert_eq!(shape.len(), 2);
                    Ok([data[0], data[1], data[2]])
                },
            )
            .expect("run tiny gemm");
        // y = [1, 2] @ [[1, 2, 3], [4, 5, 6]]
        assert!(
            observed
                .iter()
                .zip([9.0_f32, 12.0, 15.0])
                .all(|(observed, expected)| (observed - expected).abs() < 1e-5),
            "unexpected gemm output {observed:?}"
        );

        // A missing required input fails closed with a signature error.
        let error = model
            .run(vec![], |outputs| {
                let _ = outputs;
                Ok::<(), OnnxError>(())
            })
            .expect_err("empty inputs must fail closed");
        assert!(matches!(error, OnnxError::Signature { .. },));
    }

    /// A missing artifact fails closed before touching the runtime library.
    #[test]
    fn missing_artifact_fails_closed_without_a_runtime() {
        let directory = tempfile::tempdir().expect("tempdir");
        let error = match OnnxModel::load(
            &directory.path().join("model.onnx"),
            &OnnxRuntimeSettings::system(),
            OnnxAcceleration::Cpu,
            &[],
            &[],
        ) {
            Err(error) => error,
            Ok(_) => panic!("missing artifact must fail closed"),
        };
        assert!(matches!(error, OnnxError::Io { .. }));
    }
}
