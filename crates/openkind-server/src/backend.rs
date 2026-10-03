//! Backend readiness and load-time policy. Runtime probes never load checkpoints.

use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::args::DeviceOrdinals;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Backend {
    Auto,
    NativeCpu,
    Cuda,
    MlxFp32,
    Onnx,
    OnnxCuda,
    OnnxRocm,
}

impl Backend {
    fn advice(self) -> &'static str {
        match self {
            Self::Cuda => "Install a compatible NVIDIA driver and CUDA 12 user libraries (cuBLAS, cuRAND, NVRTC); check --cuda-device and CUDA_VISIBLE_DEVICES.",
            Self::OnnxCuda => "Use a GPU ONNX Runtime with CUDA 12.8 or newer and cuDNN 9; check --cuda-device and CUDA_VISIBLE_DEVICES.",
            Self::OnnxRocm => "Use a ROCm-enabled ONNX Runtime and check --rocm-device.",
            Self::Onnx => "Check --onnx-runtime, ORT_DYLIB_PATH, and the bundled runtime libraries.",
            Self::MlxFp32 => "Check Apple Silicon GPU access and the colocated mlx.metallib.",
            _ => "Check backend configuration.",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::NativeCpu => "native-cpu",
            Self::Cuda => "cuda",
            Self::MlxFp32 => "mlx-fp32",
            Self::Onnx => "onnx",
            Self::OnnxCuda => "onnx-cuda",
            Self::OnnxRocm => "onnx-rocm",
        }
    }
}

pub(crate) trait Selector: Copy {
    fn backend(self) -> Backend;
    fn from_backend(backend: Backend) -> Option<Self>;
    // Loader compatibility is separate from compiled runtime support.
    fn supports_onnx_artifact() -> bool {
        false
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Readiness {
    pub backend: Backend,
    pub compiled: bool,
    pub ready: bool,
    pub execution_ordinal: Option<usize>,
    pub reason: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct Report {
    schema_version: u32,
    hardware: Vec<openkind_runtime::Accelerator>,
    backends: Vec<Readiness>,
    onnx_runtime: Option<std::path::PathBuf>,
}

pub(crate) fn diagnose(devices: DeviceOrdinals, json: bool) -> Result<()> {
    let report = Report {
        schema_version: 1,
        hardware: openkind_runtime::detect_accelerators(),
        backends: readiness(devices),
        onnx_runtime: {
            #[cfg(feature = "onnx")]
            {
                openkind_backends::onnx::resolve_dylib_path(None)
            }
            #[cfg(not(feature = "onnx"))]
            {
                None
            }
        },
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        for hardware in &report.hardware {
            println!("{}: {}", hardware.device, hardware.name);
        }
        for status in &report.backends {
            let state = if status.ready {
                "ready"
            } else if status.compiled {
                "unavailable"
            } else {
                "not compiled"
            };
            println!(
                "{}: {}{}",
                status.backend.name(),
                state,
                status
                    .reason
                    .as_ref()
                    .map(|reason| format!(" ({reason})"))
                    .unwrap_or_default()
            );
        }
        println!("Runtime readiness does not establish model parity or qualification.");
    }
    Ok(())
}

type ProbeCache = Mutex<HashMap<(usize, usize), Vec<Readiness>>>;
static PROBES: OnceLock<ProbeCache> = OnceLock::new();

fn readiness(devices: DeviceOrdinals) -> Vec<Readiness> {
    let mut cache = PROBES
        .get_or_init(Mutex::default)
        .lock()
        .expect("probe cache");
    cache
        .entry((devices.cuda, devices.rocm))
        .or_insert_with(|| {
            let support = openkind_backends::device::execution_support();
            [
                (Backend::NativeCpu, true, None),
                (Backend::Cuda, support.candle_cuda, Some(devices.cuda)),
                (Backend::MlxFp32, support.mlx, Some(0)),
                (Backend::Onnx, support.onnx, None),
                (Backend::OnnxCuda, support.onnx_cuda, Some(devices.cuda)),
                (Backend::OnnxRocm, support.onnx_rocm, Some(devices.rocm)),
            ]
            .into_iter()
            .map(|(backend, compiled, execution_ordinal)| {
                let result = if backend == Backend::NativeCpu {
                    Ok(())
                } else if !compiled {
                    Err("backend is not compiled into this daemon".into())
                } else {
                    isolated_probe(backend, devices)
                };
                Readiness {
                    backend,
                    compiled,
                    ready: result.is_ok(),
                    execution_ordinal,
                    reason: result
                        .err()
                        .map(|reason| format!("{reason}. {}", backend.advice())),
                }
            })
            .collect()
        })
        .clone()
}

fn isolated_probe(backend: Backend, devices: DeviceOrdinals) -> std::result::Result<(), String> {
    // A vendor runtime can panic or abort while loading libraries. Keep that
    // failure out of the serving process and bound driver initialization time.
    let mut child = Command::new(std::env::current_exe().map_err(|error| error.to_string())?)
        .args([
            "--probe-backend",
            backend.name(),
            "--cuda-device",
            &devices.cuda.to_string(),
            "--rocm-device",
            &devices.rocm.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        // main has already applied the explicit setting to ORT_DYLIB_PATH.
        // An inherited CLI-setting alias must not override it in the child.
        .env_remove("OPENKIND_ONNX_RUNTIME")
        .env_remove("OPENKIND_DIAGNOSE_BACKENDS")
        .env_remove("OPENKIND_DIAGNOSTICS_JSON")
        .spawn()
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() < Duration::from_secs(20) => {
                std::thread::sleep(Duration::from_millis(20))
            }
            state => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(match state {
                    Err(error) => error.to_string(),
                    _ => "runtime probe timed out".into(),
                });
            }
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(
            "runtime probe failed; check driver, shared libraries, and execution ordinal".into(),
        );
    }
    serde_json::from_slice::<std::result::Result<(), String>>(&output.stdout)
        .map_err(|_| "runtime probe returned an invalid response".to_owned())?
}

pub(crate) fn probe(backend: Backend, devices: DeviceOrdinals) -> Result<()> {
    let result = probe_runtime(backend, devices).map_err(|error| error.to_string());
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

fn probe_runtime(backend: Backend, devices: DeviceOrdinals) -> Result<()> {
    let _ = devices;
    match backend {
        Backend::NativeCpu => Ok(()),
        #[cfg(feature = "cuda")]
        Backend::Cuda => openkind_backends::device::probe_cuda(devices.cuda).map_err(Into::into),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        Backend::MlxFp32 => openkind_backends::device::probe_mlx().map_err(Into::into),
        #[cfg(feature = "onnx")]
        Backend::Onnx => {
            openkind_backends::onnx::probe_runtime(openkind_backends::onnx::OnnxAcceleration::Cpu)
                .map_err(Into::into)
        }
        #[cfg(feature = "onnx-cuda")]
        Backend::OnnxCuda => openkind_backends::onnx::probe_runtime(
            openkind_backends::onnx::OnnxAcceleration::Cuda {
                device_id: devices.cuda,
            },
        )
        .map_err(Into::into),
        #[cfg(feature = "onnx-rocm")]
        Backend::OnnxRocm => openkind_backends::onnx::probe_runtime(
            openkind_backends::onnx::OnnxAcceleration::Rocm {
                device_id: devices.rocm,
            },
        )
        .map_err(Into::into),
        _ => anyhow::bail!("backend is not compiled for this platform"),
    }
}

fn candidates<S: Selector>(selection: S, root: &Path, statuses: &[Readiness]) -> Vec<S> {
    if selection.backend() != Backend::Auto {
        return vec![selection];
    }
    [
        Backend::MlxFp32,
        Backend::Cuda,
        Backend::OnnxCuda,
        Backend::NativeCpu,
        Backend::Onnx,
    ]
    .into_iter()
    .filter(|backend| {
        let export_exists = S::supports_onnx_artifact() && root.join("model.onnx").is_file();
        let export = !matches!(backend, Backend::Onnx | Backend::OnnxCuda) || export_exists;
        // An export-only directory can execute ONNX without attempting
        // a native weight loader first. Present native weights must still
        // pass their loader's full digest and profile checks.
        let native = !export_exists
            || !matches!(
                backend,
                Backend::Cuda | Backend::NativeCpu | Backend::MlxFp32
            )
            || native_weights_exist(root);
        export
            && native
            && statuses
                .iter()
                .any(|status| status.backend == *backend && status.ready)
    })
    .filter_map(S::from_backend)
    .collect()
}

fn native_weights_exist(root: &Path) -> bool {
    [root.to_path_buf(), root.join("checkpoint")]
        .iter()
        .any(|directory| {
            std::fs::read_dir(directory).is_ok_and(|entries| {
                entries.flatten().any(|entry| {
                    matches!(
                        entry.path().extension().and_then(|ext| ext.to_str()),
                        Some("safetensors")
                    ) || entry.file_name() == "option_marker.pt"
                })
            })
        })
}

pub(crate) fn load<S: Selector, T>(
    selection: S,
    root: &Path,
    devices: DeviceOrdinals,
    loader: impl FnMut(S) -> Result<T>,
) -> Result<T> {
    // Unit tests inject probes into the policy seam. Never run a test harness
    // as though it were a daemon subprocess.
    #[cfg(not(test))]
    let statuses = readiness(devices);
    #[cfg(test)]
    let statuses = {
        let _ = devices;
        vec![Readiness {
            backend: Backend::NativeCpu,
            compiled: true,
            ready: true,
            execution_ordinal: None,
            reason: None,
        }]
    };
    load_with_readiness(selection, root, &statuses, loader)
}

fn load_with_readiness<S: Selector, T>(
    selection: S,
    root: &Path,
    statuses: &[Readiness],
    mut loader: impl FnMut(S) -> Result<T>,
) -> Result<T> {
    let auto = selection.backend() == Backend::Auto;
    if !auto {
        if let Some(status) = statuses
            .iter()
            .find(|status| status.backend == selection.backend() && !status.ready)
        {
            return Err(openkind_backends::device::AcceleratorUnavailable(format!(
                "requested {}: {}",
                selection.backend().name(),
                status
                    .reason
                    .as_deref()
                    .unwrap_or("runtime initialization failed")
            ))
            .into());
        }
    }
    if auto {
        for status in statuses.iter().filter(|status| {
            status.compiled && !status.ready && S::from_backend(status.backend).is_some()
        }) {
            tracing::info!(backend = status.backend.name(), reason = ?status.reason, "backend skipped during automatic selection");
        }
    }
    let mut last_error = None;
    for candidate in candidates(selection, root, statuses) {
        match loader(candidate) {
            Ok(value) => {
                let execution_ordinal = statuses
                    .iter()
                    .find(|status| status.backend == candidate.backend())
                    .and_then(|status| status.execution_ordinal);
                tracing::info!(
                    backend = candidate.backend().name(),
                    ?execution_ordinal,
                    automatic = auto,
                    "selected execution backend"
                );
                return Ok(value);
            }
            Err(error) if auto && recoverable(&error) => {
                tracing::warn!(backend = candidate.backend().name(), reason = %format!("{error:#}"), "backend load unavailable; trying fallback");
                last_error = Some(error);
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no compatible execution backend is ready")))
        .context("automatic backend selection exhausted")
}

fn recoverable(error: &anyhow::Error) -> bool {
    use openkind_backends::families::support::FamilyError;
    error.chain().any(|cause| {
        if matches!(
            cause.downcast_ref::<FamilyError>(),
            Some(FamilyError::ExecutionUnavailable(_))
        ) {
            return true;
        }
        #[cfg(feature = "onnx")]
        if matches!(
            cause.downcast_ref::<openkind_backends::onnx::OnnxError>(),
            Some(
                openkind_backends::onnx::OnnxError::RuntimeLibrary(_)
                    | openkind_backends::onnx::OnnxError::CudaProvider(_)
                    | openkind_backends::onnx::OnnxError::RocmProvider(_)
                    | openkind_backends::onnx::OnnxError::AcceleratorMemory(_)
            )
        ) {
            return true;
        }
        // CUDA failures are typed; only allocation failures during model
        // loading are eligible. Shape, tokenizer, and digest errors are fatal.
        if cause
            .downcast_ref::<openkind_backends::device::AcceleratorUnavailable>()
            .is_some()
            || openkind_backends::device::is_accelerator_allocation_failure(cause)
        {
            return true;
        }
        false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone, Copy)]
    struct Selection(Backend);
    impl Selector for Selection {
        fn backend(self) -> Backend {
            self.0
        }
        fn from_backend(backend: Backend) -> Option<Self> {
            Some(Self(backend))
        }
        fn supports_onnx_artifact() -> bool {
            true
        }
    }
    fn ready(backends: &[Backend]) -> Vec<Readiness> {
        backends
            .iter()
            .map(|&backend| Readiness {
                backend,
                compiled: true,
                ready: true,
                execution_ordinal: Some(2),
                reason: None,
            })
            .collect()
    }
    #[test]
    fn accelerator_order_and_cpu_fallback() {
        let statuses = ready(&[Backend::Cuda, Backend::MlxFp32, Backend::NativeCpu]);
        let mut attempts = Vec::new();
        let selected = load_with_readiness(
            Selection(Backend::Auto),
            Path::new("missing-root"),
            &statuses,
            |selection| {
                attempts.push(selection.0);
                if selection.0 == Backend::NativeCpu {
                    Ok(selection.0)
                } else {
                    Err(openkind_backends::device::AcceleratorUnavailable(
                        "runtime unavailable".into(),
                    )
                    .into())
                }
            },
        )
        .unwrap();
        assert_eq!(selected, Backend::NativeCpu);
        assert_eq!(
            attempts,
            [Backend::MlxFp32, Backend::Cuda, Backend::NativeCpu]
        );
    }
    #[test]
    fn explicit_override_never_falls_back() {
        let mut attempts = Vec::new();
        let result: Result<()> = load_with_readiness(
            Selection(Backend::Cuda),
            Path::new("missing-root"),
            &ready(&[Backend::Cuda, Backend::NativeCpu]),
            |selection| {
                attempts.push(selection.0);
                Err(
                    openkind_backends::device::AcceleratorUnavailable("bad device ordinal".into())
                        .into(),
                )
            },
        );
        assert!(result.is_err());
        assert_eq!(attempts, [Backend::Cuda]);
    }

    #[test]
    fn proxy_encoder_preserves_allocation_and_integrity_failures() {
        use openkind_backends::families::support::FamilyError;
        use openkind_backends::proxy_cache::ProxyCacheError;
        let root = Path::new("missing-root");
        let statuses = ready(&[Backend::Cuda, Backend::NativeCpu]);
        let mut attempts = Vec::new();
        let selected =
            load_with_readiness(Selection(Backend::Auto), root, &statuses, |selection| {
                attempts.push(selection.0);
                if selection.0 == Backend::NativeCpu {
                    Ok(selection.0)
                } else {
                    Err(ProxyCacheError::from(candle_core::Error::Cuda(Box::new(
                        std::io::Error::other("CUDA_ERROR_OUT_OF_MEMORY"),
                    )))
                    .into())
                }
            })
            .unwrap();
        assert_eq!(selected, Backend::NativeCpu);
        assert_eq!(attempts, [Backend::Cuda, Backend::NativeCpu]);
        let corrupt = ProxyCacheError::from(FamilyError::DigestMismatch {
            path: "encoder weights".into(),
            expected: "a".into(),
            actual: "b".into(),
        });
        assert!(!recoverable(&corrupt.into()));
        let invalid = ProxyCacheError::Encoder("out of memory in invalid configuration".into());
        assert!(!recoverable(&invalid.into()));
    }
    #[test]
    fn unavailable_probe_excludes_backend_and_rejects_explicit_request() {
        let mut statuses = ready(&[Backend::Cuda, Backend::NativeCpu]);
        statuses[0].ready = false;
        statuses[0].reason = Some("CUDA libraries missing or invalid ordinal".into());
        assert_eq!(
            candidates(
                Selection(Backend::Auto),
                Path::new("missing-root"),
                &statuses
            )
            .iter()
            .map(|s| s.0)
            .collect::<Vec<_>>(),
            [Backend::NativeCpu]
        );
        let result: Result<()> = load_with_readiness(
            Selection(Backend::Cuda),
            Path::new("missing-root"),
            &statuses,
            |_| panic!("must not load"),
        );
        assert!(result.unwrap_err().to_string().contains("invalid ordinal"));
    }
    #[test]
    fn corrupted_artifact_and_configuration_errors_never_fall_back() {
        use openkind_backends::families::support::FamilyError;
        for error in [
            FamilyError::DigestMismatch {
                path: "weights".into(),
                expected: "a".into(),
                actual: "b".into(),
            },
            FamilyError::InvalidInput("invalid configuration".into()),
            FamilyError::ContractMismatch {
                field: "profile",
                expected: "a".into(),
                actual: "b".into(),
            },
        ] {
            let mut calls = 0;
            let mut error = Some(error);
            let result: Result<()> = load_with_readiness(
                Selection(Backend::Auto),
                Path::new("missing-root"),
                &ready(&[Backend::Cuda, Backend::NativeCpu]),
                |_| {
                    calls += 1;
                    Err(anyhow::Error::new(error.take().unwrap()).context("load pinned profile"))
                },
            );
            assert!(result.is_err());
            assert_eq!(calls, 1);
        }
    }
    #[test]
    fn export_only_directory_selects_onnx_and_requires_export() {
        let root = tempfile::tempdir().unwrap();
        let statuses = ready(&[
            Backend::Cuda,
            Backend::OnnxCuda,
            Backend::NativeCpu,
            Backend::Onnx,
        ]);
        assert!(
            !candidates(Selection(Backend::Auto), root.path(), &statuses)
                .iter()
                .any(|s| matches!(s.0, Backend::Onnx | Backend::OnnxCuda))
        );
        std::fs::write(root.path().join("model.onnx"), []).unwrap();
        // Raw external ONNX weights are not a native checkpoint.
        std::fs::write(root.path().join("weights.bin"), []).unwrap();
        assert_eq!(
            candidates(Selection(Backend::Auto), root.path(), &statuses)
                .iter()
                .map(|s| s.0)
                .collect::<Vec<_>>(),
            [Backend::OnnxCuda, Backend::Onnx]
        );
    }
    #[test]
    fn cuda_then_onnx_cuda_then_cpu_and_oom_fallback() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("model.onnx"), []).unwrap();
        std::fs::write(root.path().join("model.safetensors"), []).unwrap();
        let statuses = ready(&[
            Backend::Cuda,
            Backend::OnnxCuda,
            Backend::NativeCpu,
            Backend::Onnx,
        ]);
        assert_eq!(
            candidates(Selection(Backend::Auto), root.path(), &statuses)
                .iter()
                .map(|s| s.0)
                .collect::<Vec<_>>(),
            [
                Backend::Cuda,
                Backend::OnnxCuda,
                Backend::NativeCpu,
                Backend::Onnx
            ]
        );
        #[cfg(feature = "onnx")]
        {
            let mut calls = Vec::new();
            let selected = load_with_readiness(
                Selection(Backend::Auto),
                root.path(),
                &statuses,
                |selection| {
                    calls.push(selection.0);
                    if selection.0 == Backend::NativeCpu {
                        Ok(selection.0)
                    } else {
                        Err(openkind_backends::onnx::OnnxError::AcceleratorMemory(
                            "CUDA_ERROR_OUT_OF_MEMORY".into(),
                        )
                        .into())
                    }
                },
            )
            .unwrap();
            assert_eq!(selected, Backend::NativeCpu);
            assert_eq!(
                calls,
                [Backend::Cuda, Backend::OnnxCuda, Backend::NativeCpu]
            );
            let error = openkind_backends::onnx::OnnxError::Integrity {
                path: "model.onnx".into(),
                message: "digest mismatch".into(),
            };
            assert!(!recoverable(&error.into()));
        }
        assert!(!recoverable(&anyhow::anyhow!(
            "out of memory in invalid input"
        )));
    }

    #[test]
    fn unsupported_family_candidates_are_excluded() {
        use crate::args::CudaOnlyBackendArg;
        let statuses = ready(&[Backend::MlxFp32, Backend::Onnx, Backend::NativeCpu]);
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("model.onnx"), []).unwrap();
        assert_eq!(
            candidates(CudaOnlyBackendArg::Auto, root.path(), &statuses)
                .iter()
                .map(|s| s.backend())
                .collect::<Vec<_>>(),
            [Backend::NativeCpu]
        );
        #[cfg(not(feature = "onnx"))]
        assert!(
            candidates(crate::args::FamilyBackendArg::Auto, root.path(), &statuses,).is_empty()
        );
    }
}
