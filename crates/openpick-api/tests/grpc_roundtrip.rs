//! Integration test: spin up a real gRPC server bound to an ephemeral
//! port, connect with the generated client, and round-trip a request.
//!
//! Catches the "builds but doesn't speak to itself" class of bug — both
//! halves of the wire have to agree on every field.

use std::collections::HashMap;

use openpick_api::{grpc, AppState};
use openpick_engine::MockEngine;
use openpick_proto::openpick::question::Kind as PbQKind;
use openpick_proto::openpick::state::Value as PbStateValue;
use openpick_proto::openpick::system_one_client::SystemOneClient;
use openpick_proto::openpick::NoulQuestion as PbNoul;
use openpick_proto::openpick::Question as PbQuestion;
use openpick_proto::openpick::State as PbState;
use openpick_proto::openpick::SystemOneRequest as PbRequest;
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

fn noul_q() -> PbQuestion {
    PbQuestion {
        kind: Some(PbQKind::Noul(PbNoul {
            instructions_json: serde_json::to_vec(&serde_json::json!("?")).unwrap().into(),
            criteria: None,
        })),
    }
}

fn choice_q(criteria: HashMap<String, String>) -> PbQuestion {
    PbQuestion {
        kind: Some(PbQKind::Choice(openpick_proto::openpick::ChoiceQuestion {
            instructions_json: serde_json::to_vec(&serde_json::json!("Pick one team"))
                .unwrap()
                .into(),
            criteria,
        })),
    }
}

fn score_q(criteria: Vec<String>) -> PbQuestion {
    PbQuestion {
        kind: Some(PbQKind::Score(openpick_proto::openpick::ScoreQuestion {
            instructions_json: serde_json::to_vec(&serde_json::json!("Rate severity"))
                .unwrap()
                .into(),
            criteria,
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
    questions.insert("is_urgent".to_string(), noul_q());
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
async fn grpc_roundtrip_choice_and_score_questions() {
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut choice_criteria = HashMap::new();
    choice_criteria.insert("billing".into(), "Billing issues".into());
    choice_criteria.insert("tech".into(), "".into()); // empty string mapped to None

    let mut questions = HashMap::new();
    questions.insert("dept".to_string(), choice_q(choice_criteria));
    questions.insert(
        "severity".to_string(),
        score_q(vec!["Low".into(), "Medium".into(), "High".into()]),
    );

    let pb_req = req("mock", questions);
    let resp = client.evaluate(pb_req).await.unwrap().into_inner();

    assert_eq!(resp.model, "mock");
    assert_eq!(resp.answers.len(), 2);

    let dept_ans = resp.answers.get("dept").unwrap();
    match &dept_ans.kind {
        Some(openpick_proto::openpick::answer::Kind::Choice(c)) => {
            assert!(c.probabilities.contains_key("billing"));
            assert!(c.probabilities.contains_key("tech"));
            assert!(c.confidence >= 0.0 && c.confidence <= 1.0);
        }
        _ => panic!("expected Choice answer"),
    }

    let sev_ans = resp.answers.get("severity").unwrap();
    match &sev_ans.kind {
        Some(openpick_proto::openpick::answer::Kind::Score(s)) => {
            assert_eq!(s.legend.len(), 3);
            assert_eq!(s.probabilities.len(), 3);
            assert!(s.score >= 0.0 && s.score <= 2.0);
        }
        _ => panic!("expected Score answer"),
    }

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_roundtrip_structured_state_and_noul_criteria() {
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let struct_json = serde_json::to_vec(&serde_json::json!({
        "customer": "Alice",
        "orders": [123, 456]
    }))
    .unwrap();

    let mut questions = HashMap::new();
    questions.insert(
        "q".to_string(),
        PbQuestion {
            kind: Some(PbQKind::Noul(PbNoul {
                instructions_json: serde_json::to_vec(&serde_json::json!("Is Alice happy?"))
                    .unwrap()
                    .into(),
                criteria: Some(openpick_proto::openpick::NoulCriteria {
                    is_true: "Customer is pleased".into(),
                    is_false: "Customer is unhappy".into(),
                }),
            })),
        },
    );

    let pb_req = PbRequest {
        state: Some(PbState {
            value: Some(PbStateValue::Structured(
                openpick_proto::openpick::Structured {
                    json: struct_json.into(),
                },
            )),
        }),
        model: "mock".into(),
        questions,
    };

    let resp = client.evaluate(pb_req).await.unwrap().into_inner();
    assert_eq!(resp.model, "mock");
    assert!(resp.answers.contains_key("q"));

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
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::new();
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
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::new();
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
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::new();
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
async fn grpc_structured_array_state_roundtrip() {
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::new();
    questions.insert("q".to_string(), noul_q());
    let array_json = serde_json::to_vec(&serde_json::json!([
        {"speaker": "user", "text": "Hello"},
        {"speaker": "assistant", "text": "Hi there"}
    ]))
    .unwrap();

    let pb_req = PbRequest {
        state: Some(PbState {
            value: Some(PbStateValue::Structured(
                openpick_proto::openpick::Structured {
                    json: array_json.into(),
                },
            )),
        }),
        model: "mock".into(),
        questions,
    };

    let resp = client.evaluate(pb_req).await.unwrap().into_inner();
    assert_eq!(resp.model, "mock");
    assert!(resp.answers.contains_key("q"));

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_error_carries_request_id_metadata() {
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let pb_req = req("unknown-model", HashMap::new());
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
async fn grpc_auth_enforces_bearer_token() {
    use openpick_api::AuthConfig;

    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let auth = AuthConfig::new(Some("secret-grpc-token".into()));

    let svc = grpc::service_with_auth(registry, auth);
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

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::new();
    questions.insert("q".to_string(), noul_q());
    let pb_req = PbRequest {
        state: Some(PbState {
            value: Some(PbStateValue::Text("test".into())),
        }),
        model: "mock".into(),
        questions,
    };

    // 1. Request without auth fails with Unauthenticated
    let unauth_err = client.evaluate(pb_req.clone()).await.unwrap_err();
    assert_eq!(unauth_err.code(), tonic::Code::Unauthenticated);
    assert!(unauth_err.metadata().get("x-typesafe-request-id").is_some());

    // 2. Request with invalid auth fails with Unauthenticated
    let mut bad_req = tonic::Request::new(pb_req.clone());
    bad_req
        .metadata_mut()
        .insert("authorization", "Bearer wrong-token".parse().unwrap());
    let bad_err = client.evaluate(bad_req).await.unwrap_err();
    assert_eq!(bad_err.code(), tonic::Code::Unauthenticated);

    // 3. Request with valid auth succeeds
    let mut good_req = tonic::Request::new(pb_req);
    good_req
        .metadata_mut()
        .insert("authorization", "Bearer secret-grpc-token".parse().unwrap());
    let good_resp = client.evaluate(good_req).await.unwrap().into_inner();
    assert_eq!(good_resp.model, "mock");
    assert!(good_resp.answers.contains_key("q"));

    let _ = tx.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_auth_accepts_x_api_key_and_lowercase_bearer() {
    use openpick_api::AuthConfig;

    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let auth = AuthConfig::new(Some("secret-grpc-token".into()));

    let svc = grpc::service_with_auth(registry, auth);
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
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let mut client = SystemOneClient::connect(format!("http://{addr}"))
        .await
        .unwrap();

    let mut questions = HashMap::new();
    questions.insert("q".to_string(), noul_q());
    let pb_req = PbRequest {
        state: Some(PbState {
            value: Some(PbStateValue::Text("test".into())),
        }),
        model: "mock".into(),
        questions,
    };

    // 1. The x-api-key metadata is accepted as an alternative to the
    //    Authorization header.
    let mut key_req = tonic::Request::new(pb_req.clone());
    key_req
        .metadata_mut()
        .insert("x-api-key", "secret-grpc-token".parse().unwrap());
    let key_resp = client.evaluate(key_req).await.unwrap().into_inner();
    assert_eq!(key_resp.model, "mock");

    // 2. A lowercase `bearer` prefix is accepted too (HTTP auth headers
    //    are case-insensitive per RFC 9110 §11.6.1).
    let mut lower_req = tonic::Request::new(pb_req);
    lower_req
        .metadata_mut()
        .insert("authorization", "bearer secret-grpc-token".parse().unwrap());
    let lower_resp = client.evaluate(lower_req).await.unwrap().into_inner();
    assert_eq!(lower_resp.model, "mock");

    let _ = tx.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_unsafe_request_id_is_replaced_with_uuid() {
    let mut registry = openpick_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let (addr, tx, handle) = run_server(AppState::new(registry)).await;

    let mut client = SystemOneClient::connect(format!("http://{addr}"))
        .await
        .unwrap();

    let mut questions = HashMap::new();
    questions.insert("q".to_string(), noul_q());
    let mut request = tonic::Request::new(req("mock", questions));
    // Header-injection attempt: spaces are not safe id characters.
    let unsafe_id = "injected id; drop table";
    request
        .metadata_mut()
        .insert("x-typesafe-request-id", unsafe_id.parse().unwrap());

    let resp = client.evaluate(request).await.unwrap();
    let stamped = resp
        .metadata()
        .get("x-typesafe-request-id")
        .expect("success responses must carry x-typesafe-request-id")
        .to_str()
        .unwrap()
        .to_string();
    assert_ne!(stamped, unsafe_id, "unsafe id must not be echoed back");
    assert_eq!(stamped.len(), 36, "should be a fresh UUIDv4");

    let _ = tx.send(());
    let _ = handle.await;
}
