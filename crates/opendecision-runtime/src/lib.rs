//! `opendecision-runtime`: Hardware abstraction, device discovery, and execution management.
//!
//! # Purpose & Integration
//! As documented in `docs/ARCHITECTURE.md` and `docs/ROADMAP.md`, `opendecision-runtime`
//! provides hardware device detection (CPU, Apple Silicon Metal Performance Shaders, NVIDIA CUDA),
//! memory limit accounting, worker pool sizing, backend-neutral execution capabilities, and the
//! complete branchable-state contract. Qwen3.5 state captures attention KV, DeltaNet recurrent
//! state, and convolution state together.
//!
//! Real model backends defined in `opendecision-backends` query this crate to discover available compute
//! targets and configure thread parallelism.

#![warn(missing_docs)]

/// Backend-neutral branchable continuation-state contracts.
pub mod branch;
/// Typed digests over finalized execution-input token sequences.
pub mod digest;
/// Backend-neutral native-run evidence artifacts.
pub mod evidence;
/// Backend capabilities and generic execution-plan vocabulary.
pub mod execution;
/// Process-memory observations for admission calibration.
pub mod memory;

pub use digest::{
    CandidateTokenDigest, ExecutionInputDigest, QuestionTokenDigest, SemanticSetDigest,
    StateTokenDigest,
};
pub use evidence::{
    generate_run_id, BackendRecord, ChecksumsRecord, EvidenceError, ExecutionRecord,
    NativeRunWriter, ProfileRecord, RunEnvironment, RunRecord, SanitizedInvocation,
    CHECKSUMS_SCHEMA, NATIVE_RUN_SCHEMA,
};
pub use execution::{BackendCapabilities, BatchForwardMode, ExecutionPlan};
pub use memory::peak_resident_bytes;

use std::fmt;

/// Compute device types supported by the opendecision runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Host CPU execution fallback.
    Cpu,
    /// Apple Silicon Metal Performance Shaders (MPS).
    Metal {
        /// Zero-based Metal device index.
        device_id: usize,
    },
    /// NVIDIA CUDA execution device.
    Cuda {
        /// Zero-based CUDA device ordinal.
        device_id: usize,
    },
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Metal { device_id } => write!(f, "metal:{device_id}"),
            DeviceType::Cuda { device_id } => write!(f, "cuda:{device_id}"),
        }
    }
}

/// Runtime configuration for device allocation and execution limits.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeConfig {
    /// Target execution device.
    pub device: DeviceType,
    /// Maximum memory allocation budget in bytes (None = unbounded).
    pub memory_limit_bytes: Option<usize>,
    /// Number of worker threads for CPU/runtime parallel processing.
    pub worker_threads: usize,
}

impl RuntimeConfig {
    /// Construct a new `RuntimeConfig`, ensuring `worker_threads` is at least 1 to avoid thread pool panics.
    pub fn new(
        device: DeviceType,
        worker_threads: usize,
        memory_limit_bytes: Option<usize>,
    ) -> Self {
        Self {
            device,
            memory_limit_bytes: memory_limit_bytes.filter(|&b| b > 0),
            worker_threads: worker_threads.max(1),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: DeviceType::Cpu,
            memory_limit_bytes: None,
            worker_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
                .max(1),
        }
    }
}

/// Detect available acceleration devices on the current host.
pub fn detect_available_devices() -> Vec<DeviceType> {
    let mut devices = vec![DeviceType::Cpu];

    #[cfg(target_os = "macos")]
    {
        // Apple Silicon macOS platforms support Metal (MPS).
        #[cfg(target_arch = "aarch64")]
        {
            devices.push(DeviceType::Metal { device_id: 0 });
        }
    }

    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_config_uses_cpu_and_available_parallelism() {
        let config = RuntimeConfig::default();
        assert_eq!(config.device, DeviceType::Cpu);
        assert!(config.worker_threads >= 1);
        assert_eq!(config.memory_limit_bytes, None);
    }

    #[test]
    fn runtime_config_clamps_zero_worker_threads_to_one() {
        let config = RuntimeConfig::new(DeviceType::Cpu, 0, None);
        assert_eq!(
            config.worker_threads, 1,
            "zero threads would panic the pool"
        );
    }

    #[test]
    fn runtime_config_treats_zero_memory_limit_as_unbounded() {
        let config = RuntimeConfig::new(DeviceType::Cpu, 4, Some(0));
        assert_eq!(config.memory_limit_bytes, None);
        let bounded = RuntimeConfig::new(DeviceType::Cpu, 4, Some(1024));
        assert_eq!(bounded.memory_limit_bytes, Some(1024));
    }

    #[test]
    fn runtime_config_preserves_explicit_device() {
        let config = RuntimeConfig::new(DeviceType::Cuda { device_id: 2 }, 8, None);
        assert_eq!(config.device, DeviceType::Cuda { device_id: 2 });
        assert_eq!(config.worker_threads, 8);
    }

    #[test]
    fn device_type_display_formatting() {
        assert_eq!(DeviceType::Cpu.to_string(), "cpu");
        assert_eq!(DeviceType::Metal { device_id: 0 }.to_string(), "metal:0");
        assert_eq!(DeviceType::Cuda { device_id: 1 }.to_string(), "cuda:1");
    }

    #[test]
    fn detect_available_devices_includes_cpu() {
        let devices = detect_available_devices();
        assert!(devices.contains(&DeviceType::Cpu));
    }
}
