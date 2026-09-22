use std::collections::HashMap;

use openkind_api::{grpc, AppState, AuthConfig};
use openkind_engine::MockEngine;
use openkind_proto::openkind::state::Value as PbStateValue;
use openkind_proto::openkind::system_one_client::SystemOneClient;
use openkind_proto::openkind::State as PbState;
use openkind_proto::openkind::SystemOneRequest as PbRequest;
use tokio::sync::oneshot;
use tonic::transport::Server;

use super::helpers::{noul_q, req, run_server};

#[tokio::test(flavor = "multi_thread")]
async fn grpc_auth_enforces_bearer_token() {
    let mut registry = openkind_engine::EngineRegistry::new();
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
    let mut registry = openkind_engine::EngineRegistry::new();
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
    let mut registry = openkind_engine::EngineRegistry::new();
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
