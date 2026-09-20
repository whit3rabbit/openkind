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

impl RateLimiter {
    /// Construct a `RateLimiter` with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
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
    fn check(&self, ip: std::net::IpAddr) -> Result<(), u64> {
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
        if entry.0 >= self.config.max_requests {
            let elapsed = now.duration_since(entry.1);
            let remaining_ms = window
                .saturating_sub(elapsed)
                .as_millis()
                .min(u64::MAX as u128) as u64;
            return Err(remaining_ms.max(1));
        }
        entry.0 += 1;
        Ok(())
    }
}

/// Stackable middleware function: fixed-window rate limit on `/v1/*` per
/// client IP (taken from the `ConnectInfo` extension, which `axum::serve`
/// provides when the router is served via
/// `into_make_service_with_connect_info`). Requests without connect info
/// (unit tests, unix-socket setups) are passed through — limit per-IP is
/// only enforceable when the peer address is known.
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
            Ok(()) => next.run(req).await,
            Err(retry_after_ms) => {
                tracing::debug!(%ip, retry_after_ms, "rate limited");
                crate::error::ApiError::RateLimited { retry_after_ms }.into_response()
            }
        },
        None => next.run(req).await,
    }
}
