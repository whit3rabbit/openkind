use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use axum::Router;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use openkind_api::RateLimiter;
use openkind_core::{NoulQuestion, Question, State, SystemRequest};
use openkind_server::{http, AppState, AuthConfig, EngineRegistry, MockEngine};
use tower::ServiceExt;

const API_KEY: &str = "criterion-bench-key";
const QUESTION_COUNTS: [usize; 3] = [1, 8, 32];

fn app() -> Router {
    let mut registry = EngineRegistry::new();
    registry.register("mock", Arc::new(MockEngine::new()));
    http::router_with_state_auth_rate_limit(
        AppState::new(registry),
        AuthConfig::new(Some(API_KEY.into())),
        http::MAX_PAYLOAD_SIZE_BYTES,
        RateLimiter::disabled(),
    )
}

fn request_body(question_count: usize) -> Vec<u8> {
    let questions = (0..question_count)
        .map(|index| {
            (
                format!("q{index}"),
                Question::Noul(NoulQuestion {
                    instructions: serde_json::json!(format!("Question {index}?")),
                    criteria: None,
                }),
            )
        })
        .collect::<HashMap<_, _>>();
    serde_json::to_vec(&SystemRequest {
        state: State::Text("A deterministic server benchmark state.".into()),
        model: "mock".into(),
        questions,
    })
    .expect("serialize benchmark request")
}

fn health_request() -> Request<Body> {
    Request::builder()
        .uri("/health")
        .body(Body::empty())
        .expect("build health request")
}

fn systemone_request(body: &[u8]) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header(header::AUTHORIZATION, format!("Bearer {API_KEY}"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_vec()))
        .expect("build SystemOne request")
}

async fn execute(app: Router, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app.oneshot(request).await.expect("route request");
    let status = response.status();
    let body = to_bytes(response.into_body(), http::MAX_PAYLOAD_SIZE_BYTES)
        .await
        .expect("read response body")
        .to_vec();
    (status, body)
}

fn bench_server(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build Tokio runtime");
    let app = app();

    let (status, body) = runtime.block_on(execute(app.clone(), health_request()));
    assert_eq!(status, StatusCode::OK);
    assert!(!body.is_empty());

    c.bench_function("server_http/health", |b| {
        b.to_async(&runtime).iter_batched(
            || (app.clone(), health_request()),
            |(app, request)| async move { black_box(execute(app, request).await) },
            BatchSize::SmallInput,
        );
    });

    let mut group = c.benchmark_group("server_http_systemone");
    for question_count in QUESTION_COUNTS {
        let body = request_body(question_count);
        let (status, response_body) =
            runtime.block_on(execute(app.clone(), systemone_request(&body)));
        assert_eq!(status, StatusCode::OK);
        assert!(!response_body.is_empty());

        group.throughput(Throughput::Elements(question_count as u64));
        group.bench_with_input(
            BenchmarkId::new("questions", question_count),
            &body,
            |b, body| {
                b.to_async(&runtime).iter_batched(
                    || (app.clone(), systemone_request(body)),
                    |(app, request)| async move { black_box(execute(app, request).await) },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_server);
criterion_main!(benches);
