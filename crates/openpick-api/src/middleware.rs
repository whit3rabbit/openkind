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

/// Stackable middleware function: stamp every response with a request id.
pub async fn request_id_layer(mut req: Request<Body>, next: Next) -> Response {
    // Honor an inbound id if the client supplied one (lets a proxy
    // thread the id through); otherwise mint a fresh UUIDv4.
    let id = req
        .headers()
        .get(&REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
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
    pub expected: Arc<Option<String>>,
}

impl AuthConfig {
    pub fn new(expected: Option<String>) -> Self {
        Self {
            expected: Arc::new(expected),
        }
    }

    pub fn resolve_api_key_with<F>(get_env: F) -> Option<String>
    where
        F: Fn(&str) -> Result<String, std::env::VarError>,
    {
        get_env("OPENPICK_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| get_env("TYPESAFE_API_KEY").ok().filter(|s| !s.is_empty()))
    }

    pub fn from_env() -> Self {
        Self::new(Self::resolve_api_key_with(|k| std::env::var(k)))
    }

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
    if !auth.is_required() || path == "/health" || path == "/metrics" {
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

    let ok = match (supplied, auth.expected.as_ref()) {
        (Some(given), Some(expected)) => constant_time_eq(given.as_bytes(), expected.as_bytes()),
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

/// Constant-time byte comparison. Returns false for mismatched lengths.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
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
    fn constant_time_eq_handles_mismatched_lengths() {
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
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
