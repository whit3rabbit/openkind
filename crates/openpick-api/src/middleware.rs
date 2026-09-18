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
}
