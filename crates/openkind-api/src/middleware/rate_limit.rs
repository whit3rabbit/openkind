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

/// Shared fixed-window counter state for [`rate_limit_layer`].
#[derive(Debug, Clone)]
pub struct RateLimiter {
    config: RateLimitConfig,
    buckets: Arc<
        std::sync::Mutex<std::collections::HashMap<std::net::IpAddr, (u32, std::time::Instant)>>,
    >,
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
            buckets: Arc::new(std::sync::Mutex::new(std::collections::HashMap::default())),
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
        if !self.is_enabled() {
            return Ok(());
        }
        let mut buckets = self
            .buckets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let now = std::time::Instant::now();
        if buckets.len() >= RATE_LIMIT_SWEEP_THRESHOLD {
            buckets.retain(|_, (_, start)| now.duration_since(*start) < self.config.window);
        }
        let window = self.config.window;
        let entry = buckets.entry(ip).or_insert((0, now));
        if now.duration_since(entry.1) >= window {
            *entry = (0, now);
        }
        if units > self.config.max_requests.saturating_sub(entry.0) {
            let elapsed = now.duration_since(entry.1);
            let remaining_ms = window
                .saturating_sub(elapsed)
                .as_millis()
                .min(u64::MAX as u128) as u64;
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
