//! Execution-device and backend-selection vocabulary shared by the native
//! backbone and every surveyed family.
//!
//! Candle model code is device-agnostic: loaders resolve a
//! [`FamilyExecution`] into a [`candle_core::Device`] once, and every tensor
//! afterwards follows the weights' device. The CPU reference path resolves to
//! [`Device::Cpu`] exactly as before; the CUDA path requires the `cuda`
//! feature and a working CUDA driver at load time and fails closed otherwise.
//! ONNX execution is selected with the same enum but runs through ONNX
//! Runtime (see [`crate::onnx`]) instead of candle, on CPU, CUDA, or — with
//! the `onnx-rocm` feature — an AMD ROCm execution provider.

/// Execution selection for one family engine load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FamilyExecution {
    /// Candle FP32 CPU reference execution — the frozen parity oracle.
    Cpu,
    /// Candle FP32 execution on a CUDA device (feature `cuda`).
    #[cfg(feature = "cuda")]
    Cuda {
        /// Zero-based CUDA device ordinal.
        device_id: usize,
    },
    /// ONNX Runtime execution (feature `onnx`). `device_id: None` runs the
    /// CPU execution provider; `Some(id)` registers the CUDA execution
    /// provider for that ordinal and fails closed when the loaded ONNX
    /// Runtime cannot provide it.
    #[cfg(feature = "onnx")]
    Onnx {
        /// Optional CUDA device ordinal for the CUDA execution provider.
        device_id: Option<usize>,
    },
    /// ONNX Runtime execution on the AMD ROCm execution provider (feature
    /// `onnx-rocm`, Linux only). Fails closed when the loaded ONNX Runtime
    /// cannot provide ROCm support.
    #[cfg(feature = "onnx-rocm")]
    OnnxRocm {
        /// Zero-based ROCm (HIP) device ordinal.
        device_id: usize,
    },
}

impl FamilyExecution {
    /// Stable identifier fragment used in family `backend_id` strings.
    #[must_use]
    pub fn id_fragment(self) -> String {
        match self {
            Self::Cpu => "cpu-fp32".to_owned(),
            #[cfg(feature = "cuda")]
            Self::Cuda { device_id } => format!("cuda-fp32:{device_id}"),
            #[cfg(feature = "onnx")]
            Self::Onnx { device_id: None } => "onnx-cpu".to_owned(),
            #[cfg(feature = "onnx")]
            Self::Onnx {
                device_id: Some(device_id),
            } => format!("onnx-cuda:{device_id}"),
            #[cfg(feature = "onnx-rocm")]
            Self::OnnxRocm { device_id } => format!("onnx-rocm:{device_id}"),
        }
    }

    /// Resolve the candle device for candle-backed execution.
    ///
    /// ONNX selection has no candle device; callers must route it to
    /// [`crate::onnx`] before reaching for a device.
    ///
    /// # Errors
    /// Returns [`candle_core::Error`] when CUDA device creation fails —
    /// missing `cuda` feature support in the binary, no driver, or a bad
    /// ordinal — so loads fail closed instead of silently falling back.
    pub fn candle_device(self) -> candle_core::Result<candle_core::Device> {
        match self {
            Self::Cpu => Ok(candle_core::Device::Cpu),
            #[cfg(feature = "cuda")]
            Self::Cuda { device_id } => candle_core::Device::new_cuda(device_id),
            #[cfg(feature = "onnx")]
            Self::Onnx { .. } => Err(candle_core::Error::Msg(
                "onnx execution does not resolve to a candle device".to_owned(),
            )),
            #[cfg(feature = "onnx-rocm")]
            Self::OnnxRocm { .. } => Err(candle_core::Error::Msg(
                "onnx-rocm execution does not resolve to a candle device".to_owned(),
            )),
        }
    }

    /// True when this selection executes through ONNX Runtime.
    #[must_use]
    pub fn is_onnx(self) -> bool {
        #[cfg(feature = "onnx")]
        {
            #[cfg(feature = "onnx-rocm")]
            if matches!(self, Self::OnnxRocm { .. }) {
                return true;
            }
            matches!(self, Self::Onnx { .. })
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }
}

/// Compile-time report of which accelerated executions this binary contains.
///
/// Detection ([`openkind_runtime::detect_accelerators`]) reports hardware;
/// this reports what the loaded build can actually execute, so operators can
/// distinguish "GPU present but this binary lacks CUDA" from "no GPU".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionSupport {
    /// Native candle CPU execution (always available).
    pub candle_cpu: bool,
    /// Native candle CUDA execution compiled in (feature `cuda`).
    pub candle_cuda: bool,
    /// ONNX Runtime execution compiled in (feature `onnx`).
    pub onnx: bool,
    /// ONNX CUDA execution-provider registration compiled in (feature
    /// `onnx-cuda`). The runtime library must also ship CUDA support.
    pub onnx_cuda: bool,
    /// ONNX ROCm execution-provider registration compiled in (feature
    /// `onnx-rocm`, Linux only). The runtime library must also ship ROCm
    /// support.
    pub onnx_rocm: bool,
    /// MLX execution compiled in (feature `mlx`, macOS arm64).
    pub mlx: bool,
}

/// Report the execution backends compiled into this binary.
#[must_use]
pub fn execution_support() -> ExecutionSupport {
    ExecutionSupport {
        candle_cpu: true,
        candle_cuda: cfg!(feature = "cuda"),
        onnx: cfg!(feature = "onnx"),
        onnx_cuda: cfg!(feature = "onnx-cuda"),
        onnx_rocm: cfg!(feature = "onnx-rocm"),
        mlx: cfg!(all(
            feature = "mlx",
            target_os = "macos",
            target_arch = "aarch64"
        )),
    }
}

/// A runtime failed initialization before any model artifacts were loaded.
#[derive(Debug, thiserror::Error)]
#[error("accelerator unavailable: {0}")]
pub struct AcceleratorUnavailable(pub String);

/// Test CUDA driver, matrix multiplication, and compiled tensor kernels.
///
/// # Errors
/// Returns an initialization failure; run this in an isolated process because
/// vendor libraries may panic while resolving missing dependencies.
#[cfg(feature = "cuda")]
pub fn probe_cuda(device_id: usize) -> Result<(), AcceleratorUnavailable> {
    let run = || -> candle_core::Result<()> {
        let device = candle_core::Device::new_cuda(device_id)?;
        let tensor = candle_core::Tensor::ones((2, 2), candle_core::DType::F32, &device)?;
        let values = tensor.matmul(&tensor)?.affine(1.0, 1.0)?.to_vec2::<f32>()?;
        if values != vec![vec![3.0; 2]; 2] {
            candle_core::bail!("CUDA probe produced invalid arithmetic");
        }
        Ok(())
    };
    run().map_err(|error| AcceleratorUnavailable(error.to_string()))
}

/// Test the pinned MLX runtime and a synchronized GPU operation.
///
/// # Errors
/// Returns an initialization failure. This belongs in an isolated probe.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub fn probe_mlx() -> Result<(), AcceleratorUnavailable> {
    use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default())
        .map_err(|error| AcceleratorUnavailable(error.to_string()))?;
    runtime
        .execute(|| {
            let input = mlx_rs::Array::from_slice(&[1.0f32, 2.0], &[2]);
            let output = input.add(&input)?;
            output.eval()?;
            if output.as_slice::<f32>() != [2.0, 4.0] {
                return Err(mlx_rs::error::Exception::from(
                    "invalid MLX probe arithmetic",
                ));
            }
            Ok(())
        })
        .map_err(|error| AcceleratorUnavailable(error.to_string()))?
        .map_err(|error: mlx_rs::error::Exception| AcceleratorUnavailable(error.to_string()))
}

/// Identify accelerator allocation failures without treating malformed
/// artifacts or tensor shapes as reasons to switch arithmetic backends.
#[must_use]
pub fn is_accelerator_allocation_failure(error: &(dyn std::error::Error + 'static)) -> bool {
    fn allocation(message: &str) -> bool {
        let message = message.to_ascii_lowercase();
        message.contains("out_of_memory")
            || message.contains("out of memory")
            || message.contains("failed to allocate")
            || message.contains("memory allocation failed")
    }
    match error.downcast_ref::<candle_core::Error>() {
        Some(candle_core::Error::Cuda(error)) => return allocation(&error.to_string()),
        Some(candle_core::Error::WithBacktrace { inner, .. }) => {
            return is_accelerator_allocation_failure(inner.as_ref())
        }
        _ => {}
    }
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    {
        if let Some(crate::qwen35::mlx::MlxError::Operation { message, .. }) = error.downcast_ref()
        {
            return allocation(message);
        }
        if let Some(crate::families::support::FamilyError::Mlx(message)) = error.downcast_ref() {
            return allocation(message);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_typed_accelerator_allocation_errors_allow_fallback() {
        let allocation =
            candle_core::Error::Cuda(Box::new(std::io::Error::other("CUDA_ERROR_OUT_OF_MEMORY")));
        assert!(is_accelerator_allocation_failure(&allocation));
        let device_error = candle_core::Error::Cuda(Box::new(std::io::Error::other(
            "CUDA_ERROR_ILLEGAL_ADDRESS",
        )));
        assert!(!is_accelerator_allocation_failure(&device_error));
        let input_error = std::io::Error::other("out of memory in invalid model metadata");
        assert!(!is_accelerator_allocation_failure(&input_error));
    }

    #[test]
    fn cpu_execution_resolves_to_the_candle_cpu_device() {
        let device = FamilyExecution::Cpu
            .candle_device()
            .expect("cpu device resolves");
        assert!(
            matches!(device, candle_core::Device::Cpu),
            "cpu execution must resolve to the candle CPU device"
        );
    }

    #[test]
    fn cpu_id_fragment_is_stable() {
        assert_eq!(FamilyExecution::Cpu.id_fragment(), "cpu-fp32");
    }

    #[test]
    fn onnx_execution_never_resolves_to_a_candle_device() {
        #[cfg(feature = "onnx")]
        {
            let execution = FamilyExecution::Onnx { device_id: None };
            assert!(execution.is_onnx());
            assert!(execution.candle_device().is_err());
            assert_eq!(execution.id_fragment(), "onnx-cpu");
        }
        #[cfg(not(feature = "onnx"))]
        {
            // Without the feature the ONNX variant cannot be constructed;
            // the flag must stay false for the remaining selections.
            assert!(!FamilyExecution::Cpu.is_onnx());
        }
    }

    #[test]
    fn execution_support_reports_compile_time_features() {
        let support = execution_support();
        assert!(support.candle_cpu);
        assert_eq!(support.candle_cuda, cfg!(feature = "cuda"));
        assert_eq!(support.onnx, cfg!(feature = "onnx"));
        assert_eq!(support.onnx_rocm, cfg!(feature = "onnx-rocm"));
    }

    #[test]
    fn onnx_rocm_execution_never_resolves_to_a_candle_device() {
        #[cfg(feature = "onnx-rocm")]
        {
            // `onnx-rocm` implies `onnx`, so ONNX Runtime execution must
            // report through `is_onnx` as well.
            let execution = FamilyExecution::OnnxRocm { device_id: 0 };
            assert!(execution.is_onnx());
            assert!(execution.candle_device().is_err());
            assert_eq!(execution.id_fragment(), "onnx-rocm:0");
        }
        #[cfg(not(feature = "onnx-rocm"))]
        {
            assert!(!FamilyExecution::Cpu.is_onnx());
        }
    }
}
