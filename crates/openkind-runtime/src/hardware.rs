//! Host hardware discovery and process CPU-time observations used with
//! benchmark evidence and admission sizing.

use std::io;

use serde::Serialize;

/// Static host hardware observation recorded with benchmark evidence.
///
/// Values come from `sysctl` on macOS. Keys the host does not report stay
/// `None` instead of being guessed.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HostHardware {
    /// Platform model identifier, such as `Mac16,5`.
    pub model: Option<String>,
    /// CPU marketing string, such as `Apple M4 Max`.
    pub cpu_brand: Option<String>,
    /// Logical CPU cores visible to the operating system.
    pub logical_cores: Option<u32>,
    /// Total physical memory in bytes.
    pub total_memory_bytes: Option<u64>,
}

/// Observe the host hardware for attribution in benchmark summaries.
pub fn host_hardware() -> HostHardware {
    #[cfg(target_os = "macos")]
    {
        HostHardware {
            model: sysctl_string("hw.model"),
            cpu_brand: sysctl_string("machdep.cpu.brand_string"),
            logical_cores: sysctl_int("hw.ncpu").and_then(|value| u32::try_from(value).ok()),
            total_memory_bytes: sysctl_int("hw.memsize"),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        HostHardware {
            model: None,
            cpu_brand: None,
            logical_cores: std::thread::available_parallelism()
                .ok()
                .map(|value| value.get() as u32),
            total_memory_bytes: None,
        }
    }
}

/// Return cumulative user plus system CPU time for this process in seconds.
///
/// The total covers all threads of the process. Callers diff two observations
/// around a measured region, the same way `peak_resident_bytes` is used for
/// memory high-water marks.
#[cfg(unix)]
pub fn cpu_time_seconds() -> io::Result<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` initializes the provided `rusage` on success, and
    // the pointer is valid for the duration of the call.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the successful call above initialized the structure.
    let usage = unsafe { usage.assume_init() };
    Ok(timeval_seconds(usage.ru_utime) + timeval_seconds(usage.ru_stime))
}

/// Return an unsupported-platform error when the host has no Unix `getrusage`.
#[cfg(not(unix))]
pub fn cpu_time_seconds() -> io::Result<f64> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "process CPU-time observation is unavailable on this platform",
    ))
}

#[cfg(unix)]
fn timeval_seconds(value: libc::timeval) -> f64 {
    value.tv_sec as f64 + f64::from(value.tv_usec) / 1_000_000.0
}

#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    let cname = std::ffi::CString::new(name).ok()?;
    let mut size: libc::size_t = 0;
    // SAFETY: the null output buffer query only writes the required size.
    if unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
        || size == 0
    {
        return None;
    }
    let mut buffer = vec![0_u8; size];
    // SAFETY: `buffer` is sized to the value reported by the query above.
    if unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            buffer.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    let end = buffer.iter().position(|&byte| byte == 0).unwrap_or(size);
    String::from_utf8(buffer[..end].to_vec()).ok()
}

#[cfg(target_os = "macos")]
fn sysctl_int(name: &str) -> Option<u64> {
    let cname = std::ffi::CString::new(name).ok()?;
    let mut size: libc::size_t = 0;
    // SAFETY: the null output buffer query only writes the required size.
    if unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
        || size == 0
        || size > 8
    {
        return None;
    }
    let mut buffer = [0_u8; 8];
    // SAFETY: `buffer` is at least the size reported by the query above.
    if unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            buffer.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    // `sysctlbyname` returns native-endian integers; every supported target
    // is little-endian, and 4-byte values zero-extend into the u64 read.
    Some(u64::from_le_bytes(buffer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn host_hardware_reports_this_mac() {
        let hardware = host_hardware();
        assert!(hardware.model.is_some(), "hw.model should be readable");
        assert!(hardware.cpu_brand.is_some(), "cpu brand should be readable");
        let cores = hardware.logical_cores.expect("hw.ncpu should be readable");
        assert!(cores > 0);
        let memory = hardware
            .total_memory_bytes
            .expect("hw.memsize should be readable");
        assert!(memory > 0);
    }

    #[cfg(unix)]
    #[test]
    fn cpu_time_observation_is_positive_after_work() {
        let start = cpu_time_seconds().expect("read CPU time");
        let mut sink = 0_u64;
        for value in 0..200_000_u64 {
            sink = sink.wrapping_add(value.wrapping_mul(value));
        }
        std::hint::black_box(sink);
        let end = cpu_time_seconds().expect("read CPU time");
        assert!(end >= start);
    }
}
