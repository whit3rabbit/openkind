use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use super::helpers::{app, get};

#[tokio::test]
async fn every_response_has_request_id_header() {
    struct Case {
        method: &'static str,
        headers: Vec<(&'static str, &'static str)>,
        body: Value,
    }
    let cases = vec![
        Case {
            method: "GET",
            headers: vec![],
            body: json!({}),
        },
        Case {
            method: "POST",
            headers: vec![],
            body: json!({"state":"x","model":"mock","questions":{"q":{"type":"noul","instructions":"?"}}}),
        },
    ];
    let paths: Vec<&str> = vec!["/health", "/v1/models", "/v1/systemone", "/metrics"];

    for path in paths {
        for case in &cases {
            let mut b = Request::builder().method(case.method).uri(path);
            for (k, v) in &case.headers {
                b = b.header(*k, *v);
            }
            let req = if case.method == "POST" {
                b.header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&case.body).unwrap()))
                    .unwrap()
            } else {
                b.body(Body::empty()).unwrap()
            };
            let resp = app().oneshot(req).await.unwrap();
            let method = case.method;
            assert!(
                resp.headers().get("x-typesafe-request-id").is_some(),
                "{method} {path} must carry x-typesafe-request-id, got headers: {:?}",
                resp.headers()
            );
        }
    }
}

#[tokio::test]
async fn request_id_header_value_is_uuid() {
    let (_, resp) = get("/health", &[]).await;
    let id = resp.headers().get("x-typesafe-request-id").unwrap();
    let s = id.to_str().unwrap();
    assert_eq!(s.len(), 36, "expected uuid, got {s:?}");
}

#[tokio::test]
async fn request_id_is_unique_per_request() {
    let (_, r1) = get("/health", &[]).await;
    let (_, r2) = get("/health", &[]).await;
    let a = r1
        .headers()
        .get("x-typesafe-request-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let b = r2
        .headers()
        .get("x-typesafe-request-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert_ne!(a, b, "request ids must differ across requests");
}

#[tokio::test]
async fn inbound_request_id_is_honored() {
    // Lets a proxy thread the id through.
    let (status, resp) = get(
        "/health",
        &[("x-typesafe-request-id", "client-supplied-id-123")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        resp.headers().get("x-typesafe-request-id").unwrap(),
        "client-supplied-id-123"
    );
}

#[tokio::test]
async fn sdk_response_request_id_is_a_valid_uuid() {
    // Spec: SDK exposes `response.request_id` from `x-typesafe-request-id`.
    let (_, resp) = get("/health", &[]).await;
    let id = resp
        .headers()
        .get("x-typesafe-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    // UUIDv4: 8-4-4-4-12 hex, dash-separated.
    let parts: Vec<&str> = id.split('-').collect();
    assert_eq!(parts.len(), 5);
    assert_eq!(parts[0].len(), 8);
    assert_eq!(parts[1].len(), 4);
    assert_eq!(parts[2].len(), 4);
    assert_eq!(parts[3].len(), 4);
    assert_eq!(parts[4].len(), 12);
    for p in &parts[..4] {
        assert!(p.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
