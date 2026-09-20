use std::collections::HashMap;

use opendecision_api::{grpc, AppState};
use opendecision_proto::opendecision::question::Kind as PbQKind;
use opendecision_proto::opendecision::state::Value as PbStateValue;
use opendecision_proto::opendecision::NoulQuestion as PbNoul;
use opendecision_proto::opendecision::Question as PbQuestion;
use opendecision_proto::opendecision::State as PbState;
use opendecision_proto::opendecision::SystemOneRequest as PbRequest;
use tokio::sync::oneshot;
use tonic::transport::Server;

pub async fn run_server(
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

pub fn noul_q() -> PbQuestion {
    PbQuestion {
        kind: Some(PbQKind::Noul(PbNoul {
            instructions_json: serde_json::to_vec(&serde_json::json!("?")).unwrap().into(),
            criteria: None,
        })),
    }
}

pub fn choice_q(criteria: HashMap<String, String>) -> PbQuestion {
    PbQuestion {
        kind: Some(PbQKind::Choice(
            opendecision_proto::opendecision::ChoiceQuestion {
                instructions_json: serde_json::to_vec(&serde_json::json!("Pick one team"))
                    .unwrap()
                    .into(),
                criteria,
            },
        )),
    }
}

pub fn score_q(criteria: Vec<String>) -> PbQuestion {
    PbQuestion {
        kind: Some(PbQKind::Score(
            opendecision_proto::opendecision::ScoreQuestion {
                instructions_json: serde_json::to_vec(&serde_json::json!("Rate severity"))
                    .unwrap()
                    .into(),
                criteria,
            },
        )),
    }
}

pub fn req(model: &str, questions: HashMap<String, PbQuestion>) -> PbRequest {
    PbRequest {
        state: Some(PbState {
            value: Some(PbStateValue::Text("x".to_string())),
        }),
        model: model.to_string(),
        questions,
    }
}
