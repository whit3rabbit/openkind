//! Parsing logic for `Retry-After` and `retry-after-ms` HTTP headers.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::HeaderMap;

pub(crate) const REQUEST_ID_HEADER: &str = "x-typesafe-request-id";
pub(crate) const RETRY_AFTER_HEADER: &str = "retry-after";
pub(crate) const RETRY_AFTER_MS_HEADER: &str = "retry-after-ms";

/// Parse the retry delay the server asked for, in order of preference:
///
/// 1. `retry-after-ms` — milliseconds (the openkind server sends this).
/// 2. `retry-after` — seconds (integer or decimal), or an RFC 7231 HTTP-date
///    such as `Sun, 06 Nov 1994 08:49:37 GMT`.
///
/// Returns `None` when neither header is present, or their values are
/// negative, non-finite, unrepresentable in milliseconds, or unparseable.
pub fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    parse_retry_after_with(|name| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
    })
}

/// Header-source-agnostic core, so the direct HTTP/1.1 transport (which
/// keeps headers in its own structure) reuses the exact same precedence
/// and parsing rules.
pub(crate) fn parse_retry_after_with<'h, F>(lookup: F) -> Option<Duration>
where
    F: Fn(&str) -> Option<&'h str>,
{
    if let Some(raw) = lookup(RETRY_AFTER_MS_HEADER) {
        if let Some(delay) = finite_millis(raw, 1.0) {
            return Some(delay);
        }
    }
    if let Some(raw) = lookup(RETRY_AFTER_HEADER) {
        if let Some(delay) = finite_millis(raw, 1000.0) {
            return Some(delay);
        }
        if let Some(delay) = parse_http_date(raw) {
            return Some(delay);
        }
    }
    None
}

/// Parse a header value as a finite, non-negative number and scale it to
/// milliseconds (`multiplier` is the value's unit in milliseconds). An empty
/// or unparseable value returns `None`.
fn finite_millis(raw: &str, multiplier: f64) -> Option<Duration> {
    if raw.is_empty() {
        return None;
    }
    // Keep exact integer boundaries, which f64 cannot represent near u64::MAX.
    if let Ok(value) = raw.parse::<u64>() {
        let millis = value.checked_mul(multiplier as u64)?;
        return Some(Duration::from_millis(millis));
    }
    let value: f64 = raw.parse().ok()?;
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let millis = value * multiplier;
    // Reject an unrepresentable delay instead of saturating the float cast
    // into a practically infinite sleep when the total timeout is disabled.
    if !millis.is_finite() || millis >= u64::MAX as f64 {
        return None;
    }
    Some(Duration::from_millis(millis as u64))
}

/// Parse an IMF-fixdate HTTP-date (`Sun, 06 Nov 1994 08:49:37 GMT`) into a
/// delay from now. Obsolete RFC 850 / asctime forms are not supported; the
/// openkind server never emits them.
fn parse_http_date(raw: &str) -> Option<Duration> {
    // Strip the leading weekday name ("Sun, ") if present.
    let rest = raw.split_once(", ").map(|(_, r)| r).unwrap_or(raw);
    let mut fields = rest.split_ascii_whitespace();
    let day: i64 = fields.next()?.parse().ok()?;
    let month = month_number(fields.next()?)?;
    let raw_year = fields.next()?;
    if raw_year.len() != 4 || !raw_year.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let year: i64 = raw_year.parse().ok()?;
    let mut clock = fields.next()?.split(':');
    let hour: i64 = clock.next()?.parse().ok()?;
    let minute: i64 = clock.next()?.parse().ok()?;
    let second: i64 = clock.next()?.parse().ok()?;
    let max_day = match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if !(1..=max_day).contains(&day)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0..=60).contains(&second)
        || clock.next().is_some()
        || fields.next()? != "GMT"
        || fields.next().is_some()
    {
        return None;
    }
    let target = days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second;
    let now = i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs()).ok()?;
    // Dates in the past mean "retry now" (clamped to zero), matching the
    // Python SDK's `max(0.0, delta)`.
    let delta = target.saturating_sub(now).max(0);
    Some(Duration::from_secs(delta as u64))
}

fn month_number(name: &str) -> Option<i64> {
    let month = match name {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    Some(month)
}

/// Days since 1970-01-01 from a civil date (Howard Hinnant's algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}
