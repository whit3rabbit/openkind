//! Bearer token authentication middleware and timing-safe verification.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderName, HeaderValue, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};

use super::request_id::{RequestId, REQUEST_ID_HEADER};

/// Authorization header.
pub const AUTH_HEADER: HeaderName = HeaderName::from_static("authorization");

/// Optional bearer-auth state. `None` ⇒ no auth required.
#[derive(Clone, Default)]
pub struct AuthConfig {
    /// Expected Bearer API key token wrapped in an `Arc`. If `None`, authentication is disabled.
    pub expected: Arc<Option<String>>,
    /// SHA-256 digest of the expected token, computed once at construction so
    /// the per-request comparison only hashes the supplied token.
    expected_digest: Option<(Arc<str>, [u8; 32])>,
}

impl AuthConfig {
    /// Construct a new `AuthConfig` with the specified optional expected API key token.
    pub fn new(expected: Option<String>) -> Self {
        let expected_digest = expected
            .as_deref()
            .map(|token| (Arc::from(token), digest_of(token)));
        Self {
            expected: Arc::new(expected),
            expected_digest,
        }
    }

    /// Resolve an API key by consulting environment lookup closure.
    ///
    /// `OPENDECISION_API_KEY` and `OPENPICK_API_KEY` remain deprecated fallbacks
    /// so that upgrading a deployment cannot silently disable authentication.
    pub fn resolve_api_key_with<F>(get_env: F) -> Option<String>
    where
        F: Fn(&str) -> Result<String, std::env::VarError>,
    {
        get_env("OPENKIND_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| {
                get_env("OPENDECISION_API_KEY")
                    .ok()
                    .filter(|s| !s.is_empty())
            })
            .or_else(|| get_env("TYPESAFE_API_KEY").ok().filter(|s| !s.is_empty()))
            .or_else(|| get_env("OPENPICK_API_KEY").ok().filter(|s| !s.is_empty()))
    }

    /// Construct `AuthConfig` by resolving from the supported API-key environment variables.
    pub fn from_env() -> Self {
        Self::new(Self::resolve_api_key_with(|k| std::env::var(k)))
    }

    /// Returns `true` if authentication is required (an expected API key is configured).
    pub fn is_required(&self) -> bool {
        self.expected.is_some()
    }

    pub(crate) fn token_matches(&self, supplied: &str) -> bool {
        use subtle::ConstantTimeEq;
        let Some(expected) = self.expected.as_deref() else {
            return false;
        };
        // `expected` is public and can be replaced or edited through its Arc.
        // Reuse the digest only while its configuration snapshot still matches.
        let expected_digest = self
            .expected_digest
            .as_ref()
            .filter(|(cached, _)| cached.as_ref() == expected)
            .map(|(_, digest)| *digest)
            .unwrap_or_else(|| digest_of(expected));
        digest_of(supplied).ct_eq(&expected_digest).into()
    }
}

/// Stackable middleware function: gate `/v1/*` requests on a bearer
/// token when one is configured. `/health` and `/metrics` are always
/// open so probes and scrapers don't need credentials. `/playground` is
/// likewise open when the route is enabled: it serves an inert HTML shell,
/// and evaluation plus `/playground/api/*` model controls remain gated
/// (the UI collects an optional API key for those calls).
pub async fn auth_layer(
    State(auth): State<AuthConfig>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let path = req.uri().path();
    if !auth.is_required()
        || path == "/health"
        || path == "/metrics"
        || path == "/playground"
        || req.method() == axum::http::Method::OPTIONS
    {
        return next.run(req).await;
    }

    let supplied = req
        .headers()
        .get(&AUTH_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            let (scheme, token) = s.split_once(' ')?;
            scheme.eq_ignore_ascii_case("Bearer").then_some(token)
        });

    let ok = supplied.is_some_and(|token| auth.token_matches(token));

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

/// SHA-256 digest of one token as a fixed 32-byte array.
fn digest_of(token: &str) -> [u8; 32] {
    let digest = ring::digest::digest(&ring::digest::SHA256, token.as_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_ref());
    out
}

/// Secure constant-time token comparison.
///
/// To completely eliminate timing side-channels (including length-leakage attacks),
/// both inputs are hashed using SHA-256 into fixed 32-byte digests, and the digests
/// are compared in constant time using `subtle::ConstantTimeEq`.
pub fn secure_token_eq(a: &str, b: &str) -> bool {
    use subtle::ConstantTimeEq;
    digest_of(a).ct_eq(&digest_of(b)).into()
}

/// Dummy route handler used when attaching authentication middleware as an independent router layer.
async fn auth_layer_dummy_handler() -> StatusCode {
    StatusCode::OK
}

/// Build the auth middleware as a Layer for use with `.layer()`.
pub fn auth_layer_for(auth: AuthConfig) -> axum::Router {
    axum::Router::new()
        // Dummy root route to attach auth middleware layer.
        .route("/", axum::routing::get(auth_layer_dummy_handler))
        .layer(axum::middleware::from_fn_with_state(auth, auth_layer))
}
