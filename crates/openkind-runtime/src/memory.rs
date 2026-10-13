//! Process-memory observations used to calibrate scheduler admission.

use std::io;

/// Return the operating system's peak resident-set observation for this process.
///
/// On macOS `ru_maxrss` is reported in bytes. Other supported Unix targets
/// report KiB, which is converted to bytes. On Windows this is the peak
/// working set from `GetProcessMemoryInfo`. This is an observed high-water
/// mark, not the current RSS and not a prediction of the next forward pass.
#[cfg(unix)]
pub fn peak_resident_bytes() -> io::Result<usize> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` initializes the provided `rusage` on success, and
    // the pointer is valid for the duration of the call.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the successful call above initialized the structure.
    let usage = unsafe { usage.assume_init() };
    let raw = usize::try_from(usage.ru_maxrss)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "negative peak RSS"))?;
    #[cfg(target_os = "macos")]
    let bytes = raw;
    #[cfg(not(target_os = "macos"))]
    let bytes = raw.saturating_mul(1024);
    Ok(bytes)
}

/// Return the peak working set reported by `GetProcessMemoryInfo`.
#[cfg(windows)]
pub fn peak_resident_bytes() -> io::Result<usize> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters = unsafe { std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: `counters` is a valid, `cb`-sized destination buffer, and the
    // pseudo-handle from `GetCurrentProcess` requires no cleanup.
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    // `PeakWorkingSetSize` is a `SIZE_T` (usize); it cannot be negative.
    Ok(counters.PeakWorkingSetSize)
}

/// Return an unsupported-platform error when the host has neither a Unix
/// `getrusage` nor the Windows `GetProcessMemoryInfo` observation.
#[cfg(not(any(unix, windows)))]
pub fn peak_resident_bytes() -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "peak resident-set observation is unavailable on this platform",
    ))
}

/// Memory the host can give a new allocation without swapping, in bytes, or
/// `None` when the platform offers no cheap, trustworthy reading.
///
/// This is deliberately conservative. On macOS it counts free, purgeable, and
/// file-backed (cache) pages, and leaves out inactive anonymous pages, which
/// would have to be swapped out first. On Linux it is `MemAvailable`. It is a
/// point-in-time reading for an admission-style decision, not a reservation.
#[cfg(target_os = "macos")]
#[must_use]
pub fn available_memory_bytes() -> Option<u64> {
    use crate::hardware::sysctl_int;
    let page = sysctl_int("hw.pagesize")?;
    let pages = sysctl_int("vm.page_free_count")?
        .saturating_add(sysctl_int("vm.page_purgeable_count")?)
        .saturating_add(sysctl_int("vm.page_pageable_external_count")?);
    Some(pages.saturating_mul(page))
}

/// Linux: `MemAvailable` from `/proc/meminfo`.
#[cfg(target_os = "linux")]
#[must_use]
pub fn available_memory_bytes() -> Option<u64> {
    parse_mem_available(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// No reading on this platform.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
#[must_use]
pub fn available_memory_bytes() -> Option<u64> {
    None
}

/// Extract `MemAvailable` (kB in the file) as bytes.
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
fn parse_mem_available(meminfo: &str) -> Option<u64> {
    let line = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?;
    let kilobytes: u64 = line.split_whitespace().next()?.parse().ok()?;
    kilobytes.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(unix, windows))]
    fn peak_resident_observation_is_nonzero() {
        assert!(peak_resident_bytes().expect("read peak RSS") > 0);
    }

    #[test]
    fn meminfo_available_is_parsed_from_kilobytes() {
        let text = "MemTotal:       16384000 kB\nMemFree:          100000 kB\nMemAvailable:    8192000 kB\n";
        assert_eq!(parse_mem_available(text), Some(8_192_000 * 1024));
        assert_eq!(parse_mem_available("MemTotal: 1 kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: abc kB\n"), None);
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn available_memory_is_reported_and_bounded_by_the_host() {
        let available = available_memory_bytes().expect("read available memory");
        assert!(available > 0);
        // No host this runs on has an exabyte free.
        assert!(available < (1 << 60));
    }
}
