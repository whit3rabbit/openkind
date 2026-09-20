//! Unit tests for rate limiting middleware and burst controls.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware,
    routing::get,
};
use tower::ServiceExt;

use crate::middleware::{rate_limit_layer, RateLimitConfig, RateLimiter};

async fn echo() -> &'static str {
    "ok"
}

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

#[tokio::test]
async fn rate_limiter_skips_options_requests() {
    let limiter = RateLimiter::new(RateLimitConfig {
        max_requests: 1,
        window: std::time::Duration::from_secs(60),
    });
    let app = axum::Router::new()
        .route("/v1/ping", axum::routing::any(echo))
        .layer(middleware::from_fn_with_state(limiter, rate_limit_layer))
        .with_state(());

    for _ in 0..2 {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(axum::http::Method::OPTIONS)
                    .uri("/v1/ping")
                    .extension(connect_info(6))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/ping")
                .extension(connect_info(6))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
