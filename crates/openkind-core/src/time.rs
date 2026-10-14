//! Small time helpers shared by the transport and server layers.

use std::time::Duration;

/// Whole milliseconds in `duration`, saturating at `u64::MAX`.
///
/// `Duration::as_millis` returns `u128`; a bare `as u64` cast would wrap for
/// absurd durations and turn a long backoff into a short one.
pub fn duration_millis_saturating(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_and_saturates() {
        assert_eq!(
            duration_millis_saturating(Duration::from_millis(2500)),
            2500
        );
        assert_eq!(duration_millis_saturating(Duration::ZERO), 0);
        assert_eq!(duration_millis_saturating(Duration::MAX), u64::MAX);
    }
}
