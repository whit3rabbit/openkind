use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static RUN_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Format Unix seconds as ISO-8601 UTC, e.g. `2026-09-20T15:22:06Z`.
#[must_use]
pub fn format_utc_timestamp(unix_seconds: u64) -> String {
    let (year, month, day, hour, minute, second) = civil_from_unix(unix_seconds);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Generate a UTC run identifier with a process and invocation suffix.
#[must_use]
pub fn generate_run_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    // Evidence directories must be fresh, including concurrent runs in one second.
    format!(
        "{}-{}-{:09}-{}",
        generate_run_id_at(now.as_secs()),
        std::process::id(),
        now.subsec_nanos(),
        RUN_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn generate_run_id_at(unix_seconds: u64) -> String {
    let (year, month, day, hour, minute, second) = civil_from_unix(unix_seconds);
    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

/// Civil date-time from Unix seconds (Howard Hinnant's `civil_from_days`).
fn civil_from_unix(unix_seconds: u64) -> (u64, u64, u64, u64, u64, u64) {
    let days = (unix_seconds / 86_400) as i64;
    let seconds_of_day = unix_seconds % 86_400;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    (
        year.unsigned_abs(),
        month.unsigned_abs(),
        day.unsigned_abs(),
        seconds_of_day / 3_600,
        (seconds_of_day % 3_600) / 60,
        seconds_of_day % 60,
    )
}
