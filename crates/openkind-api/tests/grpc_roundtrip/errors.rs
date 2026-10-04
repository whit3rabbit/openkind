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

#[tokio::test(flavor = "multi_thread")]
async fn grpc_conversion_errors_preserve_or_generate_request_ids() {
    use openkind_proto::openkind::{state::Value, State, Structured};

    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let (addr, shutdown, server) = run_server(AppState::new(registry)).await;
    let mut client = SystemOneClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let valid = req("mock", HashMap::from([("q".into(), noul_q())]));
    let mut cases = Vec::new();
    let mut missing_state = valid.clone();
    missing_state.state = None;
    cases.push(missing_state);
    let mut missing_value = valid.clone();
    missing_value.state = Some(State { value: None });
    cases.push(missing_value);
    for json in [b"{".as_slice(), b"true".as_slice()] {
        let mut malformed_state = valid.clone();
        malformed_state.state = Some(State {
            value: Some(Value::Structured(Structured {
                json: json.to_vec().into(),
            })),
        });
        cases.push(malformed_state);
    }
    cases.push(req(
        "mock",
        HashMap::from([("q".into(), PbQuestion { kind: None })]),
    ));
    cases.push(req(
        "mock",
        HashMap::from([(
            "q".into(),
            PbQuestion {
                kind: Some(PbQKind::Noul(PbNoul {
                    instructions_json: b"{".to_vec().into(),
                    criteria: None,
                })),
            },
        )]),
    ));

    for malformed in cases {
        for supplied_id in [None, Some("conversion-error-id")] {
            let mut request = tonic::Request::new(malformed.clone());
            if let Some(id) = supplied_id {
                request
                    .metadata_mut()
                    .insert("x-typesafe-request-id", id.parse().unwrap());
            }
            let status = client.evaluate(request).await.unwrap_err();
            assert_eq!(status.code(), tonic::Code::InvalidArgument);
            let returned = status
                .metadata()
                .get("x-typesafe-request-id")
                .expect("conversion errors must carry a request ID")
                .to_str()
                .unwrap();
            if let Some(id) = supplied_id {
                assert_eq!(returned, id);
            } else {
                uuid::Uuid::parse_str(returned).expect("generated request ID must be a UUID");
            }
        }
    }

    let _ = shutdown.send(());
    let _ = server.await;
}

// ------------------------------------------------------------------
// gRPC error parity: every EngineError variant maps to its documented
// tonic status code, matching the HTTP layer's mapping.
// ------------------------------------------------------------------

struct FaultEngine(openkind_engine::EngineError);

#[async_trait::async_trait]
impl openkind_engine::DecisionEngine for FaultEngine {
    fn backend_id(&self) -> &str {
        "fault-engine"
    }

    fn model_metadata(&self) -> openkind_core::ModelInfo {
        openkind_core::ModelInfo {
            name: "fault".into(),
            description: "fault injector".into(),
            release_date: "1970-01-01".into(),
        }
    }

    async fn evaluate(
        &self,
        _request: openkind_core::SystemRequest,
    ) -> Result<openkind_core::SystemResponse, openkind_engine::EngineError> {
        Err(match &self.0 {
            openkind_engine::EngineError::Overloaded {
                backend,
                retry_after_ms,
            } => openkind_engine::EngineError::Overloaded {
                backend: backend.clone(),
                retry_after_ms: *retry_after_ms,
            },
            openkind_engine::EngineError::Backend { backend, message } => {
                openkind_engine::EngineError::Backend {
                    backend: backend.clone(),
                    message: message.clone(),
                }
            }
            openkind_engine::EngineError::DeadlineExceeded {
                backend,
                timeout_ms,
            } => openkind_engine::EngineError::DeadlineExceeded {
                backend: backend.clone(),
                timeout_ms: *timeout_ms,
            },
            openkind_engine::EngineError::BackendValidation { backend, .. } => {
                openkind_engine::EngineError::BackendValidation {
                    backend: backend.clone(),
                    source: openkind_core::ValidationError::MissingAnswer("q".into()),
                }
            }
            other => unreachable!("unexpected injected error {other:?}"),
        })
    }
}

async fn fault_status(error: openkind_engine::EngineError) -> tonic::Status {
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("fault", std::sync::Arc::new(FaultEngine(error)));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let mut client = SystemOneClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let mut questions = HashMap::default();
    questions.insert("q".to_string(), noul_q());
    let result = client.evaluate(req("fault", questions)).await;
    let _ = shutdown.send(());
    let _ = server.await;
    result.expect_err("a fault engine always fails")
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_overloaded_maps_to_unavailable() {
    let status = fault_status(openkind_engine::EngineError::Overloaded {
        backend: "fault-engine".into(),
        retry_after_ms: 25,
    })
    .await;
    assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
    assert!(status.message().contains("overloaded"), "{status:?}");
    assert!(status.metadata().get("retry-after-ms").is_some());
    assert!(status.metadata().get("retry-after").is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_backend_fault_maps_to_internal() {
    let status = fault_status(openkind_engine::EngineError::Backend {
        backend: "fault-engine".into(),
        message: "exploded".into(),
    })
    .await;
    assert_eq!(status.code(), tonic::Code::Internal, "{status:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_deadline_maps_to_deadline_exceeded() {
    let status = fault_status(openkind_engine::EngineError::DeadlineExceeded {
        backend: "fault-engine".into(),
        timeout_ms: 1_500,
    })
    .await;
    assert_eq!(status.code(), tonic::Code::DeadlineExceeded, "{status:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_backend_validation_maps_to_internal() {
    let status = fault_status(openkind_engine::EngineError::BackendValidation {
        backend: "fault-engine".into(),
        source: openkind_core::ValidationError::MissingAnswer("q".into()),
    })
    .await;
    assert_eq!(status.code(), tonic::Code::Internal);
    assert!(status.message().contains("missing an answer"));
}
