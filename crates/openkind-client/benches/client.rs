use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use openkind_api::{http, AppState, AuthConfig, RateLimiter};
use openkind_client::{
    Client, NoulQuestion, Question, RetryPolicy, State, SystemRequest, SystemResponse,
};
use openkind_engine::{EngineRegistry, MockEngine};
use tokio::sync::oneshot;

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

async fn spawn_server() -> (String, oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let address = listener.local_addr().expect("read listener address");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        axum::serve(listener, app())
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("serve benchmark router");
    });
    (format!("http://{address}"), shutdown_tx, task)
}

fn request(question_count: usize) -> SystemRequest {
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
        .collect::<HashMap<_, _, _>>();
    SystemRequest {
        state: State::Text("A deterministic client benchmark state.".into()),
        model: "mock".into(),
        questions,
    }
}

fn assert_response(response: &SystemResponse, question_count: usize) {
    assert_eq!(response.model, "mock");
    assert_eq!(response.answers.len(), question_count);
}

fn bench_client(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build Tokio runtime");
    let (base_url, shutdown_tx, server_task) = runtime.block_on(spawn_server());
    let client = Client::builder()
        .api_key(API_KEY)
        .base_url(base_url)
        .default_model("mock")
        .timeout(Duration::from_secs(5))
        .retry(RetryPolicy::new().max_retries(0))
        .build()
        .expect("build benchmark client");

    let health = runtime.block_on(client.health()).expect("warm health call");
    assert_eq!(health.status, "ok");
    c.bench_function("client_http/health", |b| {
        b.to_async(&runtime)
            .iter(|| async { black_box(client.health().await.expect("request health")) });
    });
    c.bench_function("client_http/health_parallel_4", |b| {
        b.to_async(&runtime).iter(|| async {
            let responses = tokio::join!(
                client.health(),
                client.health(),
                client.health(),
                client.health(),
            );
            black_box(responses.0.expect("first health request"));
            black_box(responses.1.expect("second health request"));
            black_box(responses.2.expect("third health request"));
            black_box(responses.3.expect("fourth health request"));
        });
    });

    let mut group = c.benchmark_group("client_http_systemone");
    for question_count in QUESTION_COUNTS {
        let request = request(question_count);
        let response = runtime
            .block_on(client.evaluate(request.clone()))
            .expect("warm SystemOne call");
        assert_response(&response, question_count);

        group.throughput(Throughput::Elements(question_count as u64));
        group.bench_with_input(
            BenchmarkId::new("questions", question_count),
            &request,
            |b, request| {
                b.to_async(&runtime).iter_batched(
                    || request.clone(),
                    |request| async {
                        black_box(
                            client
                                .evaluate(request)
                                .await
                                .expect("evaluate benchmark request"),
                        )
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();

    shutdown_tx.send(()).ok();
    runtime
        .block_on(server_task)
        .expect("join benchmark server task");
}

criterion_group!(benches, bench_client);
criterion_main!(benches);
