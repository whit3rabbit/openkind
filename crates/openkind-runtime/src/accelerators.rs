//! Real accelerator discovery backing backend selection and daemon startup
//! reporting.
//!
//! Detection is hardware-only: it reports what the host physically exposes
//! without claiming anything about which execution backends this particular
//! binary can use. A CUDA device reported here still requires a build with
//! CUDA support in `openkind-backends` (native candle CUDA or the ONNX CUDA
//! execution provider) before it can execute work.

use serde::Serialize;

use crate::hardware::host_hardware;

/// One detected accelerator on the current host.
///
/// The CPU is always reported. CUDA devices are enumerated through the NVIDIA
/// Management Library when the driver is loadable; Metal is reported on
/// Apple Silicon macs. Devices the probe cannot describe are omitted rather
/// than guessed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accelerator {
    /// Compute device kind and index. CUDA indices are physical NVML
    /// indices, which can differ from execution ordinals under CUDA masks.
    pub device: crate::DeviceType,
    /// Hardware name as reported by the platform probe.
    pub name: String,
    /// Total device memory in bytes, when the platform reports it.
    pub total_memory_bytes: Option<u64>,
}

impl Serialize for Accelerator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let mut state = serializer.serialize_struct("Accelerator", 3)?;
        state.serialize_field("device", &self.device.to_string())?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("total_memory_bytes", &self.total_memory_bytes)?;
        state.end()
    }
}

/// Detect every accelerator exposed by the current host.
///
/// The probe fails soft per device: a single unreadable CUDA device is
/// skipped instead of failing detection, and a missing NVIDIA driver simply
/// yields no CUDA entries.
#[must_use]
pub fn detect_accelerators() -> Vec<Accelerator> {
    let hardware = host_hardware();
    let mut accelerators = vec![Accelerator {
        device: crate::DeviceType::Cpu,
        name: hardware.cpu_brand.unwrap_or_else(|| "CPU".to_owned()),
        total_memory_bytes: hardware.total_memory_bytes,
    }];

    accelerators.extend(cuda_accelerators());

    #[cfg(target_os = "macos")]
    #[cfg(target_arch = "aarch64")]
    accelerators.push(Accelerator {
        device: crate::DeviceType::Metal { device_id: 0 },
        name: "Apple Silicon GPU".to_owned(),
        total_memory_bytes: None,
    });

    accelerators
}

/// Enumerate CUDA devices through NVML, returning an empty set when the
/// NVIDIA driver is not loadable.
pub(crate) fn cuda_accelerators() -> Vec<Accelerator> {
    let nvml = match nvml_wrapper::Nvml::init() {
        Ok(nvml) => nvml,
        // No driver library, no driver-backed CUDA devices.
        Err(_) => return Vec::new(),
    };
    let count = match nvml.device_count() {
        Ok(count) => count,
        Err(_) => return Vec::new(),
    };
    (0..count)
        .filter_map(|index| {
            let cuda = nvml.device_by_index(index).ok()?;
            let name = cuda
                .name()
                .unwrap_or_else(|_| format!("CUDA device {index}"));
            let total_memory_bytes = cuda.memory_info().ok().map(|memory| memory.total);
            Some(Accelerator {
                device: crate::DeviceType::Cuda {
                    device_id: index as usize,
                },
                name,
                total_memory_bytes,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceType;

    #[test]
    fn detection_always_reports_the_cpu() {
        let accelerators = detect_accelerators();
        assert!(
            accelerators
                .iter()
                .any(|accelerator| accelerator.device == DeviceType::Cpu),
            "CPU must always be detected"
        );
    }

    #[test]
    fn cpu_entry_carries_hardware_name_when_reported() {
        let accelerators = detect_accelerators();
        let cpu = accelerators
            .iter()
            .find(|accelerator| accelerator.device == DeviceType::Cpu)
            .expect("CPU entry");
        assert!(!cpu.name.is_empty());
    }

    #[test]
    fn cuda_entries_keep_unique_physical_indices() {
        let mut previous = None;
        for accelerator in detect_accelerators()
            .into_iter()
            .filter(|accelerator| matches!(accelerator.device, DeviceType::Cuda { .. }))
        {
            let DeviceType::Cuda { device_id } = accelerator.device else {
                unreachable!()
            };
            // A failed per-device probe leaves a gap. Renumbering would
            // identify a different physical GPU.
            assert!(previous.is_none_or(|index| device_id > index));
            previous = Some(device_id);
            assert!(!accelerator.name.is_empty());
        }
    }

    #[cfg(target_os = "macos")]
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn apple_silicon_reports_metal() {
        let accelerators = detect_accelerators();
        assert!(
            accelerators
                .iter()
                .any(|accelerator| accelerator.device == DeviceType::Metal { device_id: 0 }),
            "Apple Silicon must report the Metal device"
        );
    }
}
