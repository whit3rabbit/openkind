//! Cross-cutting middleware for the HTTP layer:
//! - `request_id_layer` — emits an `x-typesafe-request-id` header on
//!   every response (UUIDv4, generated server-side; the SDK reads this).
//! - `auth_layer` — optional `Authorization: Bearer *** gate.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderName, HeaderValue, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Json,
};

/// Name of the request-id header. Mirrors the SDK's `request_id` property.
pub const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-typesafe-request-id");

/// Authorization header.
pub const AUTH_HEADER: HeaderName = HeaderName::from_static("authorization");

/// Maximum allowed length for an inbound client request ID.
pub const MAX_REQUEST_ID_LEN: usize = 128;

/// Validate whether a request ID string contains only safe identifier characters.
pub fn is_safe_request_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_REQUEST_ID_LEN
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

/// Stackable middleware function: stamp every response with a request id.
pub async fn request_id_layer(mut req: Request<Body>, next: Next) -> Response {
    // Honor an inbound id if the client supplied a valid and safe one
    // (lets a proxy thread the id through); otherwise mint a fresh UUIDv4.
    let id = req
        .headers()
        .get(&REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| is_safe_request_id(s))
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    req.extensions_mut().insert(RequestId(id.clone()));

    let mut resp = next.run(req).await;
    if let Ok(v) = HeaderValue::from_str(&id) {
        resp.headers_mut().insert(REQUEST_ID_HEADER.clone(), v);
    }
    resp
}

/// Stored in request extensions by `request_id_layer`.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

/// Optional bearer-auth state. `None` ⇒ no auth required.
#[derive(Clone, Default)]
pub struct AuthConfig {
    /// Expected Bearer API key token wrapped in an `Arc`. If `None`, authentication is disabled.
    pub expected: Arc<Option<String>>,
}

impl AuthConfig {
    /// Construct a new `AuthConfig` with the specified optional expected API key token.
    pub fn new(expected: Option<String>) -> Self {
        Self {
            expected: Arc::new(expected),
        }
    }

    /// Resolve an API key by consulting environment lookup closure, checking `OPENPICK_API_KEY` then `TYPESAFE_API_KEY`.
    pub fn resolve_api_key_with<F>(get_env: F) -> Option<String>
    where
        F: Fn(&str) -> Result<String, std::env::VarError>,
    {
        get_env("OPENPICK_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| get_env("TYPESAFE_API_KEY").ok().filter(|s| !s.is_empty()))
    }

    /// Construct `AuthConfig` by resolving from environment variables `OPENPICK_API_KEY` or `TYPESAFE_API_KEY`.
    pub fn from_env() -> Self {
        Self::new(Self::resolve_api_key_with(|k| std::env::var(k)))
    }

    /// Returns `true` if authentication is required (an expected API key is configured).
    pub fn is_required(&self) -> bool {
        self.expected.is_some()
    }
}

/// Stackable middleware function: gate `/v1/*` requests on a bearer
/// token when one is configured. `/health` and `/metrics` are always
/// open so probes and scrapers don't need credentials.
pub async fn auth_layer(
    State(auth): State<AuthConfig>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    if !auth.is_required()
        || path == "/health"
        || path == "/metrics"
        || req.method() == axum::http::Method::OPTIONS
    {
        return next.run(req).await;
    }

    let supplied = req
        .headers()
        .get(&AUTH_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            s.strip_prefix("Bearer ")
                .or_else(|| s.strip_prefix("bearer "))
        });

    let ok = match (supplied, auth.expected.as_deref()) {
        (Some(given), Some(expected)) => secure_token_eq(given, expected),
        _ => false,
    };

    if !ok {
        let body = Json(serde_json::json!({
            "error": {
                "code": "unauthorized",
                "message": "missing or invalid API key",
            }
        }));
        let mut resp = (StatusCode::UNAUTHORIZED, body).into_response();
        if let Ok(v) = HeaderValue::from_str("Bearer") {
            resp.headers_mut()
                .insert(axum::http::header::WWW_AUTHENTICATE, v);
        }
        // Also stamp the request id on the 401.
        if let Some(req_id) = req.extensions().get::<RequestId>() {
            if let Ok(v) = HeaderValue::from_str(&req_id.0) {
                resp.headers_mut().insert(REQUEST_ID_HEADER.clone(), v);
            }
        }
        return resp;
    }

    next.run(req).await
}

/// Secure constant-time token comparison.
///
/// To completely eliminate timing side-channels (including length-leakage attacks),
/// both inputs are hashed using SHA-256 into fixed 32-byte digests, and the digests
/// are compared in constant time using `subtle::ConstantTimeEq`.
pub fn secure_token_eq(a: &str, b: &str) -> bool {
    use subtle::ConstantTimeEq;
    let digest_a = ring::digest::digest(&ring::digest::SHA256, a.as_bytes());
    let digest_b = ring::digest::digest(&ring::digest::SHA256, b.as_bytes());
    digest_a.as_ref().ct_eq(digest_b.as_ref()).into()
}

/// Build the auth middleware as a Layer for use with `.layer()`.
pub fn auth_layer_for(auth: AuthConfig) -> axum::Router {
    axum::Router::new()
        // dummy route, will be merged into the main router via .merge/.layer
        .route("/", axum::routing::get(|| async { StatusCode::OK }))
        .layer(middleware::from_fn_with_state(auth, auth_layer))
}

// ---------- Rate limiting ----------

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
const RATE_LIMIT_SWEEP_THRESHOLD: usize = 4096;

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
    if !limiter.is_enabled() || !req.uri().path().starts_with("/v1/") {
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    async fn echo() -> &'static str {
        "ok"
    }

    fn app(auth: AuthConfig) -> Router {
        Router::new()
            .route("/v1/ping", get(echo))
            .route("/health", get(echo))
            .layer(middleware::from_fn_with_state(auth.clone(), auth_layer))
            .layer(middleware::from_fn(request_id_layer))
            .with_state(auth)
    }

    #[tokio::test]
    async fn response_carries_request_id_header() {
        let resp = app(AuthConfig::default())
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let id = resp
            .headers()
            .get(&REQUEST_ID_HEADER)
            .expect("response must carry x-typesafe-request-id");
        let s = id.to_str().unwrap();
        assert_eq!(s.len(), 36, "expected uuid length, got {s:?}");
    }

    #[tokio::test]
    async fn inbound_request_id_is_honored() {
        let resp = app(AuthConfig::default())
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header(&REQUEST_ID_HEADER, "test-id-123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(&REQUEST_ID_HEADER).unwrap(),
            "test-id-123"
        );
    }

    #[tokio::test]
    async fn auth_disabled_passes_all_requests() {
        let resp = app(AuthConfig::default())
            .oneshot(
                Request::builder()
                    .uri("/v1/ping")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_enabled_without_token_returns_401() {
        let resp = app(AuthConfig::new(Some("topsecret".into())))
            .oneshot(
                Request::builder()
                    .uri("/v1/ping")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert!(resp
            .headers()
            .get(axum::http::header::WWW_AUTHENTICATE)
            .is_some());
        assert!(resp.headers().get(&REQUEST_ID_HEADER).is_some());
    }

    #[tokio::test]
    async fn auth_enabled_with_wrong_token_returns_401() {
        let resp = app(AuthConfig::new(Some("topsecret".into())))
            .oneshot(
                Request::builder()
                    .uri("/v1/ping")
                    .header(&AUTH_HEADER, "Bearer wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_enabled_with_correct_token_passes() {
        let resp = app(AuthConfig::new(Some("topsecret".into())))
            .oneshot(
                Request::builder()
                    .uri("/v1/ping")
                    .header(&AUTH_HEADER, "Bearer topsecret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_is_always_open_even_when_auth_required() {
        let resp = app(AuthConfig::new(Some("topsecret".into())))
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn secure_token_eq_handles_matching_and_mismatching_tokens() {
        assert!(!secure_token_eq("abc", "abcd"));
        assert!(secure_token_eq("abc", "abc"));
        assert!(!secure_token_eq("abc", "abd"));
        assert!(!secure_token_eq("", "abc"));
        assert!(secure_token_eq(
            "super-secret-key-12345",
            "super-secret-key-12345"
        ));
    }

    #[test]
    fn is_safe_request_id_accepts_safe_shapes() {
        assert!(is_safe_request_id("a"));
        assert!(is_safe_request_id("550e8400-e29b-41d4-a716-446655440000"));
        assert!(is_safe_request_id("allowed-chars_A.b0"));
        assert!(is_safe_request_id(&"a".repeat(MAX_REQUEST_ID_LEN)));
    }

    #[test]
    fn is_safe_request_id_rejects_unsafe_shapes() {
        assert!(!is_safe_request_id(""));
        assert!(!is_safe_request_id(&"a".repeat(MAX_REQUEST_ID_LEN + 1)));
        assert!(!is_safe_request_id("has space"));
        assert!(!is_safe_request_id("semi;colon"));
        assert!(!is_safe_request_id("new\nline"));
        assert!(!is_safe_request_id("tab\tchar"));
        assert!(!is_safe_request_id("sl/ash"));
        assert!(!is_safe_request_id("quer?y"));
        assert!(!is_safe_request_id("ang<l>e"));
        assert!(!is_safe_request_id("émoji"));
    }

    #[tokio::test]
    async fn options_requests_bypass_auth() {
        // `any` routing so OPTIONS reaches the auth layer instead of
        // dying with 405 at the router.
        let app = Router::new()
            .route("/v1/ping", axum::routing::any(echo))
            .layer(middleware::from_fn_with_state(
                AuthConfig::new(Some("topsecret".into())),
                auth_layer,
            ))
            .with_state(());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(axum::http::Method::OPTIONS)
                    .uri("/v1/ping")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "CORS preflight must not require a bearer token"
        );
    }

    #[tokio::test]
    async fn inbound_unsafe_request_id_is_sanitized() {
        // Injection attempt with unsafe chars (spaces, brackets, semicolons)
        let resp = app(AuthConfig::default())
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header(&REQUEST_ID_HEADER, "<script>malicious;id</script>")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = resp
            .headers()
            .get(&REQUEST_ID_HEADER)
            .unwrap()
            .to_str()
            .unwrap();
        assert_ne!(id, "<script>malicious;id</script>");
        assert_eq!(id.len(), 36, "should fall back to fresh UUIDv4");

        // Oversized ID attempt (> 128 chars)
        let long_id = "a".repeat(200);
        let resp2 = app(AuthConfig::default())
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header(&REQUEST_ID_HEADER, long_id)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let id2 = resp2
            .headers()
            .get(&REQUEST_ID_HEADER)
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(id2.len(), 36, "should fall back to fresh UUIDv4");
    }

    #[test]
    fn resolve_api_key_preference() {
        // 1. OPENPICK_API_KEY takes precedence
        let key = AuthConfig::resolve_api_key_with(|k| match k {
            "OPENPICK_API_KEY" => Ok("openpick-key".into()),
            "TYPESAFE_API_KEY" => Ok("typesafe-key".into()),
            _ => Err(std::env::VarError::NotPresent),
        });
        assert_eq!(key, Some("openpick-key".into()));

        // 2. Fallback to TYPESAFE_API_KEY if OPENPICK_API_KEY is not present
        let key2 = AuthConfig::resolve_api_key_with(|k| match k {
            "OPENPICK_API_KEY" => Err(std::env::VarError::NotPresent),
            "TYPESAFE_API_KEY" => Ok("typesafe-key".into()),
            _ => Err(std::env::VarError::NotPresent),
        });
        assert_eq!(key2, Some("typesafe-key".into()));

        // 3. Fallback to TYPESAFE_API_KEY if OPENPICK_API_KEY is empty
        let key3 = AuthConfig::resolve_api_key_with(|k| match k {
            "OPENPICK_API_KEY" => Ok("".into()),
            "TYPESAFE_API_KEY" => Ok("typesafe-key".into()),
            _ => Err(std::env::VarError::NotPresent),
        });
        assert_eq!(key3, Some("typesafe-key".into()));

        // 4. None if both are absent or empty
        let key4 = AuthConfig::resolve_api_key_with(|k| match k {
            "OPENPICK_API_KEY" => Ok("".into()),
            "TYPESAFE_API_KEY" => Ok("".into()),
            _ => Err(std::env::VarError::NotPresent),
        });
        assert_eq!(key4, None);
    }

    // ---------- Rate limit tests ----------

    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn connect_info(ip: u8) -> axum::extract::ConnectInfo<SocketAddr> {
        axum::extract::ConnectInfo(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, ip)),
            40_000,
        ))
    }

    async fn limited_app_requests(
        limiter: RateLimiter,
        ip: u8,
        n: usize,
    ) -> Vec<axum::http::StatusCode> {
        let app = axum::Router::new()
            .route("/v1/ping", get(echo))
            .route("/health", get(echo))
            .layer(middleware::from_fn_with_state(
                limiter.clone(),
                rate_limit_layer,
            ))
            .with_state(());
        let mut statuses = Vec::with_capacity(n);
        for _ in 0..n {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/ping")
                        .extension(connect_info(ip))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            statuses.push(resp.status());
        }
        statuses
    }

    #[tokio::test]
    async fn rate_limiter_blocks_after_limit_with_429_and_retry_headers() {
        let limiter = RateLimiter::new(RateLimitConfig {
            max_requests: 2,
            window: std::time::Duration::from_secs(60),
        });
        let statuses = limited_app_requests(limiter.clone(), 1, 3).await;
        assert_eq!(statuses[0], StatusCode::OK);
        assert_eq!(statuses[1], StatusCode::OK);
        assert_eq!(statuses[2], StatusCode::TOO_MANY_REQUESTS);

        // The 429 body/headers follow the SDK retry contract: error envelope
        // plus `retry-after-ms` / `Retry-After`.
        let app = axum::Router::new()
            .route("/v1/ping", get(echo))
            .layer(middleware::from_fn_with_state(limiter, rate_limit_layer))
            .with_state(());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/ping")
                    .extension(connect_info(1))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let retry_ms = resp
            .headers()
            .get("retry-after-ms")
            .expect("429 must carry retry-after-ms")
            .to_str()
            .unwrap()
            .parse::<u64>()
            .unwrap();
        assert!(retry_ms > 0 && retry_ms <= 60_000);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "60");
        let bytes = http_body_util::BodyExt::collect(resp.into_body())
            .await
            .unwrap()
            .to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["code"], "rate_limited");
    }

    #[tokio::test]
    async fn rate_limiter_is_per_ip() {
        let limiter = RateLimiter::new(RateLimitConfig {
            max_requests: 1,
            window: std::time::Duration::from_secs(60),
        });
        // IP .1 exhausts its budget...
        let first = limited_app_requests(limiter.clone(), 1, 2).await;
        assert_eq!(first[0], StatusCode::OK);
        assert_eq!(first[1], StatusCode::TOO_MANY_REQUESTS);
        // ...IP .2 is unaffected.
        let second = limited_app_requests(limiter, 2, 1).await;
        assert_eq!(second[0], StatusCode::OK);
    }

    #[tokio::test]
    async fn rate_limiter_skips_non_v1_paths() {
        let limiter = RateLimiter::new(RateLimitConfig {
            max_requests: 1,
            window: std::time::Duration::from_secs(60),
        });
        let statuses = limited_app_requests(limiter.clone(), 3, 2).await;
        assert_eq!(statuses[0], StatusCode::OK);
        assert_eq!(statuses[1], StatusCode::TOO_MANY_REQUESTS);

        // /health is exempt: never limited regardless of exhaustion elsewhere.
        let app = axum::Router::new()
            .route("/health", get(echo))
            .layer(middleware::from_fn_with_state(limiter, rate_limit_layer))
            .with_state(());
        for _ in 0..5 {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/health")
                        .extension(connect_info(3))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn rate_limiter_window_resets() {
        let limiter = RateLimiter::new(RateLimitConfig {
            max_requests: 1,
            window: std::time::Duration::from_millis(50),
        });
        let statuses = limited_app_requests(limiter.clone(), 4, 2).await;
        assert_eq!(statuses[0], StatusCode::OK);
        assert_eq!(statuses[1], StatusCode::TOO_MANY_REQUESTS);
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        let after = limited_app_requests(limiter, 4, 1).await;
        assert_eq!(after[0], StatusCode::OK);
    }

    #[tokio::test]
    async fn rate_limiter_disabled_passes_everything() {
        let limiter = RateLimiter::disabled();
        let statuses = limited_app_requests(limiter, 5, 10).await;
        assert!(statuses.iter().all(|s| *s == StatusCode::OK));
    }

    #[tokio::test]
    async fn rate_limiter_passes_through_without_connect_info() {
        // No ConnectInfo extension (oneshot without into_make_service): fail-open.
        let limiter = RateLimiter::new(RateLimitConfig {
            max_requests: 1,
            window: std::time::Duration::from_secs(60),
        });
        let app = axum::Router::new()
            .route("/v1/ping", get(echo))
            .layer(middleware::from_fn_with_state(limiter, rate_limit_layer))
            .with_state(());
        for _ in 0..5 {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/ping")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
    }
}
