//! Integration test: spin up a real gRPC server bound to an ephemeral
//! port, connect with the generated client, and round-trip a request.
//!
//! Catches the "builds but doesn't speak to itself" class of bug — both
//! halves of the wire have to agree on every field.

use std::collections::HashMap;

use openpick_api::{grpc, AppState};
use openpick_engine::MockEngine;
use openpick_proto::openpick::system_one_client::SystemOneClient;
use openpick_proto::openpick::state::Value as PbStateValue;
use openpick_proto::openpick::question::Kind as PbQKind;
use openpick_proto::openpick::State as PbState;
use openpick_proto::openpick::Question as PbQuestion;
use openpick_proto::openpick::SystemOneRequest as PbRequest;
use openpick_proto::openpick::NoulQuestion as PbNoul;
use tokio::sync::oneshot;
use tonic::transport::Server;

async fn run_server(
    state: AppState,
) -> (
    std::net::SocketAddr,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let svc = grpc::service((*state.registry).clone());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let (tx, rx) = oneshot::channel::<()>();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(svc)
            .serve_with_incoming_shutdown(incoming, async move {
                let _ = rx.await;
            })
            .await;
    });
    (addr, tx, handle)
}

fn noul_q(id: &str) -> PbQuestion {
    PbQuestion {
        kind: Some(PbQKind::Noul(PbNoul {
            instructions_json: serde_json::to_vec(&serde_json::json!("?")).unwrap().into(),
            criteria: None,
        })),
    }
}

fn req(model: &str, questions: HashMap<String, PbQuestion>) -> PbRequest {
    PbRequest {
        state: Some(PbState {
            value: Some(PbStateValue::Text("x".to_string())),
        }),
        model: model.to_string(),
        questions,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_roundtrip_returns_one_answer_per_question() {
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::new();
    questions.insert("is_urgent".to_string(), noul_q("is_urgent"));
    let pb_req = req("mock", questions);

    let resp = client.evaluate(pb_req).await.unwrap().into_inner();
    assert_eq!(resp.model, "mock");
    assert!(resp.answers.contains_key("is_urgent"));
    let usage = resp.usage.expect("usage is required");
    assert!(usage.input_tokens > 0);

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_empty_questions_returns_error() {
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let pb_req = req("mock", HashMap::new());
    let resp = client.evaluate(pb_req).await;
    assert!(resp.is_err(), "empty questions must produce a gRPC error");

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_unknown_model_returns_not_found() {
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::new();
    questions.insert("q".to_string(), noul_q("q"));
    let pb_req = req("no-such-model", questions);
    let resp = client.evaluate(pb_req).await;
    assert!(resp.is_err(), "unknown model must produce a gRPC error");
    let status = resp.unwrap_err();
    assert_eq!(status.code(), tonic::Code::NotFound, "got: {status:?}");

    let _ = shutdown.send(());
    let _ = server.await;
}