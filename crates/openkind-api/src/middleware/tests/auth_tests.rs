//! Unit tests for authentication middleware and constant-time token comparison.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware,
    routing::get,
    Router,
};
use tower::ServiceExt;

use crate::middleware::{
    auth_layer, request_id_layer, secure_token_eq, AuthConfig, AUTH_HEADER, REQUEST_ID_HEADER,
};

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
async fn bearer_scheme_is_case_insensitive_while_tokens_remain_case_sensitive() {
    let router = app(AuthConfig::new(Some("secret".into())));
    for (authorization, expected) in [
        ("BEARER secret", StatusCode::OK),
        ("bEaReR secret", StatusCode::OK),
        ("bEaReR SECRET", StatusCode::UNAUTHORIZED),
        ("Basic secret", StatusCode::UNAUTHORIZED),
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/ping")
                    .header(&AUTH_HEADER, authorization)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            expected,
            "authorization: {authorization}"
        );
    }
}

#[tokio::test]
async fn auth_configuration_mutation_uses_the_current_key() {
    let mut replaced = AuthConfig::new(Some("old-key".into()));
    replaced.expected = std::sync::Arc::new(Some("new-key".into()));
    let mut edited = AuthConfig::new(Some("old-key".into()));
    *std::sync::Arc::make_mut(&mut edited.expected) = Some("new-key".into());
    let mut enabled = AuthConfig::default();
    enabled.expected = std::sync::Arc::new(Some("new-key".into()));

    for auth in [replaced, edited, enabled] {
        let router = app(auth);
        for (token, expected) in [
            ("old-key", StatusCode::UNAUTHORIZED),
            ("new-key", StatusCode::OK),
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/ping")
                        .header(&AUTH_HEADER, format!("Bearer {token}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected, "token: {token}");
        }
    }
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

#[test]
fn resolve_api_key_preference() {
    // 1. OPENKIND_API_KEY takes precedence
    let key = AuthConfig::resolve_api_key_with(|k| match k {
        "OPENKIND_API_KEY" => Ok("openkind-key".into()),
        "TYPESAFE_API_KEY" => Ok("typesafe-key".into()),
        _ => Err(std::env::VarError::NotPresent),
    });
    assert_eq!(key, Some("openkind-key".into()));

    // 2. Fallback to TYPESAFE_API_KEY if OPENKIND_API_KEY is not present
    let key2 = AuthConfig::resolve_api_key_with(|k| match k {
        "OPENKIND_API_KEY" => Err(std::env::VarError::NotPresent),
        "TYPESAFE_API_KEY" => Ok("typesafe-key".into()),
        _ => Err(std::env::VarError::NotPresent),
    });
    assert_eq!(key2, Some("typesafe-key".into()));

    // 3. Fallback to TYPESAFE_API_KEY if OPENKIND_API_KEY is empty
    let key3 = AuthConfig::resolve_api_key_with(|k| match k {
        "OPENKIND_API_KEY" => Ok("".into()),
        "TYPESAFE_API_KEY" => Ok("typesafe-key".into()),
        _ => Err(std::env::VarError::NotPresent),
    });
    assert_eq!(key3, Some("typesafe-key".into()));

    // 4. Fall back to the deprecated OPENDECISION_API_KEY during upgrades
    let key4 = AuthConfig::resolve_api_key_with(|k| match k {
        "OPENKIND_API_KEY" => Ok("".into()),
        "OPENDECISION_API_KEY" => Ok("opendecision-key".into()),
        "TYPESAFE_API_KEY" => Ok("typesafe-key".into()),
        _ => Err(std::env::VarError::NotPresent),
    });
    assert_eq!(key4, Some("opendecision-key".into()));

    // 5. OPENKIND_API_KEY takes precedence over OPENDECISION_API_KEY
    let key5 = AuthConfig::resolve_api_key_with(|k| match k {
        "OPENKIND_API_KEY" => Ok("openkind-key".into()),
        "OPENDECISION_API_KEY" => Ok("opendecision-key".into()),
        _ => Err(std::env::VarError::NotPresent),
    });
    assert_eq!(key5, Some("openkind-key".into()));

    // 6. Fall back to the older deprecated OPENPICK_API_KEY if others absent
    let key6 = AuthConfig::resolve_api_key_with(|k| match k {
        "OPENKIND_API_KEY" => Ok("".into()),
        "OPENDECISION_API_KEY" => Ok("".into()),
        "TYPESAFE_API_KEY" => Ok("".into()),
        "OPENPICK_API_KEY" => Ok("legacy-key".into()),
        _ => Err(std::env::VarError::NotPresent),
    });
    assert_eq!(key6, Some("legacy-key".into()));

    // 7. None if all variables are absent or empty
    let key7 = AuthConfig::resolve_api_key_with(|_| Err(std::env::VarError::NotPresent));
    assert_eq!(key7, None);
}
