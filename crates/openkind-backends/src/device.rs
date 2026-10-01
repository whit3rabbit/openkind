//! Execution-device and backend-selection vocabulary shared by the native
//! backbone and every surveyed family.
//!
//! Candle model code is device-agnostic: loaders resolve a
//! [`FamilyExecution`] into a [`candle_core::Device`] once, and every tensor
//! afterwards follows the weights' device. The CPU reference path resolves to
//! [`Device::Cpu`] exactly as before; the CUDA path requires the `cuda`
//! feature and a working CUDA driver at load time and fails closed otherwise.
//! ONNX execution is selected with the same enum but runs through ONNX
//! Runtime (see [`crate::onnx`]) instead of candle.

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
        }
    }

    /// True when this selection executes through ONNX Runtime.
    #[must_use]
    pub fn is_onnx(self) -> bool {
        #[cfg(feature = "onnx")]
        {
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
        mlx: cfg!(all(
            feature = "mlx",
            target_os = "macos",
            target_arch = "aarch64"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
