//! Fixed-window per-client-IP rate limiting middleware and state tracking.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::Request,
    middleware::Next,
    response::{IntoResponse, Response},
};

/// Fixed-window per-client-IP rate limit configuration.
///
/// `max_requests` of `0` disables limiting. The window is a plain fixed
/// window (not sliding): counters reset every `window` interval per IP.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests allowed per client IP within `window`. `0` disables rate limiting.
    pub max_requests: u32,
    /// Length of the counting window.
    pub window: std::time::Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        // Generous for SDK clients, but caps runaway loops and brute force.
        Self {
            max_requests: 120,
            window: std::time::Duration::from_secs(60),
        }
    }
}

/// Sweep the bucket map once it grows past this many entries so a large,
/// rotating client population cannot grow state without bound.
pub(crate) const RATE_LIMIT_SWEEP_THRESHOLD: usize = 4096;

/// Upper bound on how often the full sweep may run. A sweep is O(entries)
/// under the limiter lock, so running it on every call once the map is large
/// would turn each request into a table scan.
const RATE_LIMIT_SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// Per-IP window counters plus the time of the last sweep.
#[derive(Debug)]
struct Buckets {
    map: std::collections::HashMap<std::net::IpAddr, (u32, std::time::Instant)>,
    last_sweep: std::time::Instant,
    /// Sweeps performed, so tests can prove they are rate limited.
    #[cfg(test)]
    sweeps: u32,
}

/// Shared fixed-window counter state for [`rate_limit_layer`].
#[derive(Debug, Clone)]
pub struct RateLimiter {
    config: RateLimitConfig,
    buckets: Arc<std::sync::Mutex<Buckets>>,
}

/// Shared transport budgets with an independent failed-authentication allowance.
#[derive(Debug, Clone)]
pub struct RequestLimits {
    /// Budget for authenticated or anonymous API work.
    pub evaluation: RateLimiter,
    /// Budget charged only when credentials fail verification.
    pub failed_auth: RateLimiter,
}

impl RequestLimits {
    /// Construct independent budgets with the same configured limit and window.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            evaluation: RateLimiter::new(config.clone()),
            failed_auth: RateLimiter::new(config),
        }
    }
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

impl From<RateLimiter> for RequestLimits {
    fn from(evaluation: RateLimiter) -> Self {
        Self {
            failed_auth: RateLimiter::new(evaluation.config.clone()),
            evaluation,
        }
    }
}

/// Per-request handle used by bulk handlers to charge work beyond the one
/// unit already recorded by [`rate_limit_layer`].
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct RateLimitContext {
    limiter: RateLimiter,
    ip: std::net::IpAddr,
}

impl RateLimitContext {
    /// Atomically charge additional units to the current client's window.
    pub(crate) fn charge(&self, units: u32) -> Result<(), u64> {
        self.limiter.check_n(self.ip, units)
    }
}

impl RateLimiter {
    /// Construct a `RateLimiter` with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: Arc::new(std::sync::Mutex::new(Buckets {
                map: std::collections::HashMap::default(),
                last_sweep: std::time::Instant::now(),
                #[cfg(test)]
                sweeps: 0,
            })),
        }
    }

    /// Construct a disabled `RateLimiter` (all requests pass).
    pub fn disabled() -> Self {
        Self::new(RateLimitConfig {
            max_requests: 0,
            window: std::time::Duration::from_secs(60),
        })
    }

    /// Whether this limiter enforces anything.
    pub fn is_enabled(&self) -> bool {
        self.config.max_requests > 0
    }

    /// Record one request for `ip`. Returns `Ok(())` when under the limit,
    /// or `Err(retry_after_ms)` when the client has exhausted its window.
    pub(crate) fn check(&self, ip: std::net::IpAddr) -> Result<(), u64> {
        self.check_n(ip, 1)
    }

    /// Atomically record `units` for `ip` without partially spending a
    /// rejected charge.
    fn check_n(&self, ip: std::net::IpAddr, units: u32) -> Result<(), u64> {
        self.check_n_at(ip, units, std::time::Instant::now())
    }

    /// `check_n` with an explicit clock so sweep timing is testable.
    fn check_n_at(
        &self,
        ip: std::net::IpAddr,
        units: u32,
        now: std::time::Instant,
    ) -> Result<(), u64> {
        if !self.is_enabled() {
            return Ok(());
        }
        let mut guard = self
            .buckets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = &mut *guard;
        let window = self.config.window;
        // Entries cannot expire faster than `window`, so sweeping more often
        // than that (or than the fixed cap) only burns time. Expired entries
        // that survive until the next sweep are reset on access, so limits
        // are unaffected; only memory is bounded slightly more loosely.
        let sweep_interval = window.min(RATE_LIMIT_SWEEP_INTERVAL);
        if state.map.len() >= RATE_LIMIT_SWEEP_THRESHOLD
            && now.saturating_duration_since(state.last_sweep) >= sweep_interval
        {
            state
                .map
                .retain(|_, (_, start)| now.saturating_duration_since(*start) < window);
            state.last_sweep = now;
            #[cfg(test)]
            {
                state.sweeps += 1;
            }
        }
        let entry = state.map.entry(ip).or_insert((0, now));
        if now.saturating_duration_since(entry.1) >= window {
            *entry = (0, now);
        }
        if units > self.config.max_requests.saturating_sub(entry.0) {
            let elapsed = now.saturating_duration_since(entry.1);
            let remaining_ms =
                openkind_core::duration_millis_saturating(window.saturating_sub(elapsed));
            return Err(remaining_ms.max(1));
        }
        entry.0 += units;
        Ok(())
    }
}

/// Stackable middleware function: fixed-window rate limit on `/v1/*` per
/// client IP (taken from the `ConnectInfo` extension, which `axum::serve`
/// provides when the router is served via
/// `into_make_service_with_connect_info`). Requests without connect info
/// (unit tests, or any future non-TCP transport such as a Unix socket or a
/// Windows named pipe) are passed through — per-IP limiting is only
/// enforceable when the peer address is known.
pub async fn rate_limit_layer(
    State(limiter): State<RateLimiter>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if !limiter.is_enabled()
        || !req.uri().path().starts_with("/v1/")
        || req.method() == axum::http::Method::OPTIONS
    {
        return next.run(req).await;
    }
    let peer_ip = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|c| c.0.ip());
    match peer_ip {
        Some(ip) => match limiter.check(ip) {
            Ok(()) => {
                let mut req = req;
                req.extensions_mut().insert(RateLimitContext {
                    limiter: limiter.clone(),
                    ip,
                });
                next.run(req).await
            }
            Err(retry_after_ms) => {
                tracing::debug!(%ip, retry_after_ms, "rate limited");
                crate::error::ApiError::RateLimited { retry_after_ms }.into_response()
            }
        },
        None => next.run(req).await,
    }
}

#[cfg(test)]
mod sweep_tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::{Duration, Instant};

    fn ip(n: u32) -> IpAddr {
        IpAddr::V4(Ipv4Addr::from(n))
    }

    fn limiter(window: Duration) -> RateLimiter {
        RateLimiter::new(RateLimitConfig {
            max_requests: 2,
            window,
        })
    }

    fn sweeps(limiter: &RateLimiter) -> u32 {
        limiter.buckets.lock().unwrap().sweeps
    }

    fn len(limiter: &RateLimiter) -> usize {
        limiter.buckets.lock().unwrap().map.len()
    }

    #[test]
    fn large_maps_are_swept_at_most_once_per_interval() {
        let limiter = limiter(Duration::from_secs(60));
        let base = Instant::now();
        for n in 0..RATE_LIMIT_SWEEP_THRESHOLD as u32 {
            limiter.check_n_at(ip(n), 1, base).unwrap();
        }
        assert_eq!(sweeps(&limiter), 0, "below the interval nothing sweeps");
        // Many calls inside one interval, with the map above the threshold.
        for n in 0..500u32 {
            let at = base + Duration::from_millis(u64::from(n));
            limiter.check_n_at(ip(1_000_000 + n), 1, at).unwrap();
        }
        assert_eq!(sweeps(&limiter), 0);
        // Once the interval elapses the next call sweeps exactly once.
        let later = base + RATE_LIMIT_SWEEP_INTERVAL;
        limiter.check_n_at(ip(2_000_000), 1, later).unwrap();
        assert_eq!(sweeps(&limiter), 1);
        limiter.check_n_at(ip(2_000_001), 1, later).unwrap();
        assert_eq!(sweeps(&limiter), 1, "no second sweep inside the interval");
    }

    #[test]
    fn sweeping_drops_only_expired_windows() {
        let limiter = limiter(Duration::from_secs(60));
        let base = Instant::now();
        for n in 0..RATE_LIMIT_SWEEP_THRESHOLD as u32 {
            limiter.check_n_at(ip(n), 1, base).unwrap();
        }
        // A fresh client inside the same window must survive the sweep.
        let mid = base + Duration::from_secs(30);
        limiter.check_n_at(ip(9_000_000), 1, mid).unwrap();
        // The call at `mid` also sweeps (the interval since creation elapsed)
        // but finds nothing expired; the one at `after` removes the base cohort.
        assert_eq!(len(&limiter), RATE_LIMIT_SWEEP_THRESHOLD + 1);
        let after = base + Duration::from_secs(61);
        limiter.check_n_at(ip(9_000_001), 1, after).unwrap();
        assert_eq!(len(&limiter), 2);
    }

    #[test]
    fn limits_do_not_depend_on_sweep_timing() {
        let limiter = limiter(Duration::from_secs(60));
        let base = Instant::now();
        let client = ip(7);
        limiter.check_n_at(client, 1, base).unwrap();
        limiter.check_n_at(client, 1, base).unwrap();
        let retry = limiter.check_n_at(client, 1, base).unwrap_err();
        assert_eq!(retry, 60_000);
        // A new window resets the counter even if no sweep ever removed it.
        limiter
            .check_n_at(client, 1, base + Duration::from_secs(60))
            .unwrap();
    }
}
