use std::collections::HashMap;

use openkind_api::AppState;
use openkind_engine::MockEngine;
use openkind_proto::openkind::question::Kind as PbQKind;
use openkind_proto::openkind::state::Value as PbStateValue;
use openkind_proto::openkind::system_one_client::SystemOneClient;
use openkind_proto::openkind::NoulQuestion as PbNoul;
use openkind_proto::openkind::Question as PbQuestion;
use openkind_proto::openkind::State as PbState;
use openkind_proto::openkind::SystemOneRequest as PbRequest;

use super::helpers::{choice_q, noul_q, req, run_server, score_q};

#[tokio::test(flavor = "multi_thread")]
async fn grpc_roundtrip_returns_one_answer_per_question() {
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::default();
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
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut choice_criteria = HashMap::default();
    choice_criteria.insert("billing".into(), "Billing issues".into());
    choice_criteria.insert("tech".into(), "".into()); // empty string mapped to None

    let mut questions = HashMap::default();
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
        Some(openkind_proto::openkind::answer::Kind::Choice(c)) => {
            assert!(c.probabilities.contains_key("billing"));
            assert!(c.probabilities.contains_key("tech"));
            assert!(c.confidence >= 0.0 && c.confidence <= 1.0);
        }
        _ => panic!("expected Choice answer"),
    }

    let sev_ans = resp.answers.get("severity").unwrap();
    match &sev_ans.kind {
        Some(openkind_proto::openkind::answer::Kind::Score(s)) => {
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
    let mut registry = openkind_engine::EngineRegistry::new();
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

    let mut questions = HashMap::default();
    questions.insert(
        "q".to_string(),
        PbQuestion {
            kind: Some(PbQKind::Noul(PbNoul {
                instructions_json: serde_json::to_vec(&serde_json::json!("Is Alice happy?"))
                    .unwrap()
                    .into(),
                criteria: Some(openkind_proto::openkind::NoulCriteria {
                    is_true: "Customer is pleased".into(),
                    is_false: "Customer is unhappy".into(),
                }),
            })),
        },
    );

    let pb_req = PbRequest {
        state: Some(PbState {
            value: Some(PbStateValue::Structured(
                openkind_proto::openkind::Structured {
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
async fn grpc_structured_array_state_roundtrip() {
    let mut registry = openkind_engine::EngineRegistry::new();
    registry.register("mock", std::sync::Arc::new(MockEngine::new()));
    let state = AppState::new(registry);

    let (addr, shutdown, server) = run_server(state).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("http://{addr}");
    let mut client = SystemOneClient::connect(url).await.unwrap();

    let mut questions = HashMap::default();
    questions.insert("q".to_string(), noul_q());
    let array_json = serde_json::to_vec(&serde_json::json!([
        {"speaker": "user", "text": "Hello"},
        {"speaker": "assistant", "text": "Hi there"}
    ]))
    .unwrap();

    let pb_req = PbRequest {
        state: Some(PbState {
            value: Some(PbStateValue::Structured(
                openkind_proto::openkind::Structured {
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
