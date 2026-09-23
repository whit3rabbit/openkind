use std::collections::HashMap;

use openkind_api::AppState;
use openkind_engine::MockEngine;
use openkind_proto::openkind::question::Kind as PbQKind;
use openkind_proto::openkind::system_one_client::SystemOneClient;
use openkind_proto::openkind::NoulQuestion as PbNoul;
use openkind_proto::openkind::Question as PbQuestion;
use openkind_proto::openkind::SystemOneRequest as PbRequest;

use super::helpers::{noul_q, req, run_server};

#[tokio::test(flavor = "multi_thread")]
async fn grpc_empty_questions_returns_error() {
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let pb_req = req("mock", HashMap::default());
    let resp = client.evaluate(pb_req).await;
    assert!(resp.is_err(), "empty questions must produce a gRPC error");

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_unknown_model_returns_not_found() {
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::default();
    questions.insert("q".to_string(), noul_q());
    let pb_req = req("no-such-model", questions);
    let resp = client.evaluate(pb_req).await;
    assert!(resp.is_err(), "unknown model must produce a gRPC error");
    let status = resp.unwrap_err();
    assert_eq!(status.code(), tonic::Code::NotFound, "got: {status:?}");

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_missing_state_returns_invalid_argument() {
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::default();
    questions.insert("q".to_string(), noul_q());
    let pb_req = PbRequest {
        state: None,
        model: "mock".into(),
        questions,
    };
    let resp = client.evaluate(pb_req).await;
    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_question_without_kind_returns_invalid_argument() {
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::default();
    questions.insert("q".to_string(), PbQuestion { kind: None });
    let pb_req = req("mock", questions);

    let resp = client.evaluate(pb_req).await;
    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_malformed_instructions_json_returns_invalid_argument() {
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::default();
    questions.insert(
        "q".to_string(),
        PbQuestion {
            kind: Some(PbQKind::Noul(PbNoul {
                instructions_json: b"{not valid json".to_vec().into(),
                criteria: None,
            })),
        },
    );
    let pb_req = req("mock", questions);

    let resp = client.evaluate(pb_req).await;
    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_error_carries_request_id_metadata() {
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let pb_req = req("unknown-model", HashMap::default());
    let resp = client.evaluate(pb_req).await;
    let status = resp.unwrap_err();
    let req_id_header = status.metadata().get("x-typesafe-request-id");
    assert!(
        req_id_header.is_some(),
        "gRPC error response must contain x-typesafe-request-id metadata"
    );

    let _ = shutdown.send(());
    let _ = server.await;
}
