//! Real HTTP and gRPC listeners share budgets without mixing authentication failures with work.

use openkind_api::{grpc, http, AppState, AuthConfig, RateLimitConfig, RequestLimits};
use openkind_engine::{EngineRegistry, MockEngine};
use openkind_proto::openkind as pb;
use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::oneshot;
use tonic::transport::Server;

struct Servers {
    http: String,
    grpc: pb::system_one_client::SystemOneClient<tonic::transport::Channel>,
    stops: Vec<oneshot::Sender<()>>,
}
impl Drop for Servers {
    fn drop(&mut self) {
        for stop in self.stops.drain(..) {
            let _ = stop.send(());
        }
    }
}
async fn servers(max_requests: u32) -> Servers {
    let mut registry = EngineRegistry::new();
    registry.register("mock", Arc::new(MockEngine::new()));
    let auth = AuthConfig::new(Some("local-key".into()));
    let limits = RequestLimits::new(RateLimitConfig {
        max_requests,
        window: Duration::from_secs(60),
    });
    let app = http::router_daemon_with_arrow_and_limits(
        AppState::new(registry.clone()),
        auth.clone(),
        http::MAX_PAYLOAD_SIZE_BYTES,
        limits.clone(),
        false,
        false,
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = listener.local_addr().unwrap();
    let (http_tx, http_rx) = oneshot::channel();
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async {
            let _ = http_rx.await;
        })
        .await
        .unwrap();
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let grpc_addr = listener.local_addr().unwrap();
    let (grpc_tx, grpc_rx) = oneshot::channel();
    tokio::spawn(async move {
        Server::builder()
            .add_service(grpc::service_with_auth_and_limits(registry, auth, limits))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = grpc_rx.await;
                },
            )
            .await
            .unwrap();
    });
    Servers {
        http: format!("http://{http_addr}"),
        grpc: pb::system_one_client::SystemOneClient::connect(format!("http://{grpc_addr}"))
            .await
            .unwrap(),
        stops: vec![http_tx, grpc_tx],
    }
}
fn grpc_request(key: &str) -> tonic::Request<pb::SystemOneRequest> {
    let mut request = tonic::Request::new(pb::SystemOneRequest {
        model: "mock".into(),
        state: Some(pb::State {
            value: Some(pb::state::Value::Text("state".into())),
        }),
        questions: HashMap::from([(
            "q".into(),
            pb::Question {
                kind: Some(pb::question::Kind::Noul(pb::NoulQuestion {
                    instructions_json: br#""?""#.to_vec().into(),
                    criteria: None,
                })),
            },
        )]),
    });
    request
        .metadata_mut()
        .insert("authorization", format!("Bearer {key}").parse().unwrap());
    request
}
async fn http_request(base: &str, key: &str) -> reqwest::Response {
    reqwest::Client::new().post(format!("{base}/v1/systemone")).bearer_auth(key)
        .json(&serde_json::json!({"model":"mock","state":"state","questions":{"q":{"type":"noul","instructions":"?"}}}))
        .send().await.unwrap()
}
fn retry_metadata(status: &tonic::Status) {
    assert!(status.metadata().get("retry-after-ms").is_some());
    assert!(status.metadata().get("retry-after").is_some());
    assert!(status.metadata().get("x-typesafe-request-id").is_some());
}
#[tokio::test]
async fn evaluation_budget_cannot_be_bypassed_by_switching_transports() {
    let mut servers = servers(2).await;
    assert_eq!(http_request(&servers.http, "local-key").await.status(), 200);
    servers
        .grpc
        .evaluate(grpc_request("local-key"))
        .await
        .unwrap();
    let error = servers
        .grpc
        .evaluate(grpc_request("local-key"))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    retry_metadata(&error);
    let response = http_request(&servers.http, "local-key").await;
    assert_eq!(response.status(), 429);
    assert!(response.headers().contains_key("retry-after-ms"));
}
#[tokio::test]
async fn failed_authentication_has_its_own_shared_budget() {
    let mut servers = servers(2).await;
    assert_eq!(http_request(&servers.http, "wrong").await.status(), 401);
    let error = servers
        .grpc
        .evaluate(grpc_request("wrong"))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
    let error = servers
        .grpc
        .evaluate(grpc_request("wrong"))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    retry_metadata(&error);
    assert_eq!(http_request(&servers.http, "wrong").await.status(), 429);
    servers
        .grpc
        .evaluate(grpc_request("local-key"))
        .await
        .unwrap();
    assert_eq!(http_request(&servers.http, "local-key").await.status(), 200);
}
#[tokio::test]
async fn zero_disables_both_budgets() {
    let mut servers = servers(0).await;
    for _ in 0..4 {
        assert_eq!(http_request(&servers.http, "wrong").await.status(), 401);
        assert_eq!(
            servers
                .grpc
                .evaluate(grpc_request("wrong"))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::Unauthenticated
        );
        assert_eq!(http_request(&servers.http, "local-key").await.status(), 200);
        servers
            .grpc
            .evaluate(grpc_request("local-key"))
            .await
            .unwrap();
    }
}
