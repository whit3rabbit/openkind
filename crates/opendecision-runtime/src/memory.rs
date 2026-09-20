//! Process-memory observations used to calibrate scheduler admission.

use std::io;

/// Return the operating system's peak resident-set observation for this process.
///
/// On macOS `ru_maxrss` is reported in bytes. Other supported Unix targets
/// report KiB, which is converted to bytes. This is an observed high-water
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

/// Return an unsupported-platform error when the host has no Unix `getrusage`.
#[cfg(not(unix))]
pub fn peak_resident_bytes() -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "peak resident-set observation is unavailable on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(unix)]
    fn peak_resident_observation_is_nonzero() {
        assert!(peak_resident_bytes().expect("read peak RSS") > 0);
    }
}
