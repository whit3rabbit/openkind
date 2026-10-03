use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use openkind_engine::{EngineRegistry, MockEngine};
use serde_json::json;
use tower::ServiceExt;

use super::*;
use crate::middleware::AuthConfig;
use crate::AppState;

fn app() -> Router {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    router(reg)
}

fn playground_app(auth: AuthConfig) -> Router {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    router_daemon(
        AppState::new(reg),
        auth,
        MAX_PAYLOAD_SIZE_BYTES,
        crate::middleware::RateLimiter::disabled(),
        true,
    )
}

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // The body is a pre-encoded constant; pin the bytes and content type so
    // probes and SDK health checks keep seeing the JSON shape.
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/json")
    );
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body.as_ref(), b"{\"status\":\"ok\"}");
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_text() {
    // Must be 200 with a text body even when no recorder is installed,
    // so scrapers never see errors.
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/plain; version=0.0.4")
    );
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert!(!body.is_empty());
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.starts_with('#') || text.contains("openkind"));
}

#[tokio::test]
async fn playground_route_is_absent_by_default() {
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/playground")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn playground_route_serves_the_embedded_page_when_enabled() {
    let resp = playground_app(AuthConfig::default())
        .oneshot(
            Request::builder()
                .uri("/playground")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/html; charset=utf-8")
    );
    // no-store so an upgraded daemon never leaves a stale UI in the cache.
    assert_eq!(
        resp.headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok()),
        Some("no-store")
    );
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("openkind playground"), "marker missing");
}

#[tokio::test]
async fn playground_page_bypasses_auth_while_v1_stays_gated() {
    let gated = playground_app(AuthConfig::new(Some("topsecret".into())));
    let resp = gated
        .clone()
        .oneshot(
            Request::builder()
                .uri("/playground")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = gated
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn models_endpoint_returns_jev_shape() {
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    // Jev shape: top-level `models` array with name/description/release_date.
    assert!(v["models"].is_array(), "expected `models` array, got {v}");
    assert!(v.get("data").is_none());
    assert!(v.get("object").is_none());
    let arr = v["models"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["name"], "mock");
    assert!(arr[0]["description"].is_string());
    assert!(arr[0]["release_date"].is_string());
}

#[tokio::test]
async fn systemone_evaluates_request_and_returns_one_answer_per_question() {
    let body = json!({
        "state": "Help!",
        "model": "mock",
        "questions": {
            "is_urgent": { "type": "noul", "instructions": "?" },
            "dept": {
                "type": "choice",
                "instructions": "?",
                "criteria": { "billing": "pay", "tech": "bugs" }
            },
            "frust": {
                "type": "score",
                "instructions": "?",
                "criteria": ["Calm", "Angry"]
            }
        }
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["model"].as_str(), Some("mock"));
    let answers = v["answers"].as_object().unwrap();
    assert!(answers.contains_key("is_urgent"));
    assert!(answers.contains_key("dept"));
    assert!(answers.contains_key("frust"));
    let usage = &v["usage"];
    assert!(usage["input_tokens"].as_u64().unwrap() > 0);
    assert!(usage["output_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn unknown_model_returns_404() {
    let body = json!({
        "state": "x",
        "model": "no-such-model",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn invalid_body_returns_422() {
    let body = json!({
        "state": "x",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn malformed_json_returns_400() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from("{not valid json"))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn request_body_exceeding_custom_limit_is_rejected() {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    let custom_app = router_with_state_and_limit(AppState::new(reg), AuthConfig::default(), 1024);
    let big_body = serde_json::to_vec(&json!({
        "state": "x",
        "model": "mock",
        "padding": "x".repeat(2048)
    }))
    .unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(big_body))
        .unwrap();
    let resp = custom_app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn rejected_auth_does_not_consume_rate_limit_budget() {
    let peer: axum::extract::ConnectInfo<std::net::SocketAddr> =
        axum::extract::ConnectInfo("127.0.0.1:40000".parse().unwrap());
    let limited_app = router_with_state_auth_rate_limit(
        AppState::new(EngineRegistry::new()),
        AuthConfig::new(Some("topsecret".into())),
        MAX_PAYLOAD_SIZE_BYTES,
        crate::middleware::RateLimiter::new(crate::middleware::RateLimitConfig {
            max_requests: 1,
            window: std::time::Duration::from_secs(60),
        }),
    );

    for _ in 0..2 {
        let resp = limited_app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .extension(peer)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    let resp = limited_app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "Bearer topsecret")
                .extension(peer)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn playground_model_controls_require_opt_in_auth_and_explicit_header() {
    use crate::playground::{PlaygroundModel, PlaygroundModels};
    struct Controls(Arc<EngineRegistry>);
    #[async_trait::async_trait]
    impl PlaygroundModels for Controls {
        async fn list(&self) -> Result<Vec<PlaygroundModel>, ApiError> {
            Ok(vec![PlaygroundModel {
                name: "mock".into(),
                description: "Demo".into(),
                source: "mock".into(),
                loaded: self.0.get("mock").is_some(),
                manageable: true,
            }])
        }
        async fn set_loaded(&self, name: String, loaded: bool) -> Result<(), ApiError> {
            if loaded {
                self.0.register_if_absent(name, Arc::new(MockEngine::new()));
            } else {
                self.0.unregister(&name);
            }
            Ok(())
        }
    }
    let mut registry = EngineRegistry::new();
    registry.register("mock", Arc::new(MockEngine::new()));
    let mut state = AppState::new(registry);
    state.playground_models = Some(Arc::new(Controls(state.registry.clone())));
    let disabled = router_daemon(
        state.clone(),
        AuthConfig::default(),
        MAX_PAYLOAD_SIZE_BYTES,
        crate::RateLimiter::disabled(),
        false,
    );
    let resp = disabled
        .oneshot(
            Request::builder()
                .uri("/playground/api/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let router = router_daemon(
        state.clone(),
        AuthConfig::new(Some("secret".into())),
        MAX_PAYLOAD_SIZE_BYTES,
        crate::RateLimiter::disabled(),
        true,
    );
    let resp = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/playground/api/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert!(resp.headers().contains_key("x-typesafe-request-id"));
    for (header, site, expected) in [
        ("", "same-origin", StatusCode::UNAUTHORIZED),
        ("1", "cross-site", StatusCode::UNAUTHORIZED),
        ("1", "same-origin", StatusCode::OK),
    ] {
        let resp = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/playground/api/models")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .header("x-openkind-playground", header)
                    .header("sec-fetch-site", site)
                    .body(Body::from(r#"{"name":"mock","loaded":false}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), expected);
        assert!(resp.headers().contains_key("x-typesafe-request-id"));
        if expected != StatusCode::OK {
            assert!(state.registry.get("mock").is_some());
        }
    }
    assert!(state.registry.get("mock").is_none());
    let resp = router
        .oneshot(
            Request::builder()
                .uri("/playground/api/models")
                .header("authorization", "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.headers()["cache-control"], "no-store");
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()["models"][0]["loaded"],
        false
    );
}

// ---------- Unofficial Arrow bulk endpoint (`POST /v1/arrow`) ----------

fn arrow_questions() -> serde_json::Value {
    json!({
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["Can wait", "Within a few days", "Today"]
        },
        "refund": {
            "type": "noul",
            "instructions": "Is a refund being requested?"
        },
        "department": {
            "type": "choice",
            "instructions": "Which department?",
            "criteria": {
                "billing": "Payments, refunds",
                "shipping": null,
                "other": null
            }
        }
    })
}

fn arrow_request_body(states: serde_json::Value) -> serde_json::Value {
    json!({
        "model": "mock",
        "states": states,
        "questions": arrow_questions()
    })
}

fn arrow_app(auth: AuthConfig) -> Router {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    router_daemon_with_arrow(
        AppState::new(reg),
        auth,
        MAX_PAYLOAD_SIZE_BYTES,
        crate::middleware::RateLimiter::disabled(),
        false,
        true,
    )
}

async fn post_arrow(
    router: Router,
    body: serde_json::Value,
) -> axum::http::Response<axum::body::Body> {
    router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/arrow")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
}

/// Decode the response body as a schema-first Arrow IPC stream and assert
/// the stream is well-formed (exactly one batch, then end-of-stream).
async fn decode_arrow_response(
    resp: axum::http::Response<axum::body::Body>,
) -> (
    std::sync::Arc<arrow_schema::Schema>,
    arrow_array::RecordBatch,
) {
    use arrow_ipc::reader::StreamReader;
    use std::io::Cursor;

    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let mut reader = StreamReader::try_new(Cursor::new(bytes.to_vec()), None).unwrap();
    let schema = reader.schema().clone();
    let batch = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none(), "stream must end after one batch");
    (schema, batch)
}

#[tokio::test]
async fn arrow_route_is_absent_by_default() {
    let resp = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/arrow")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn arrow_route_serves_ipc_stream_with_row_per_state() {
    let (schema, batch) = decode_arrow_response(
        post_arrow(
            arrow_app(AuthConfig::default()),
            arrow_request_body(json!(["Where is my parcel?", "Please refund the shoes."])),
        )
        .await,
    )
    .await;

    assert_eq!(batch.num_rows(), 2);
    // Columns follow the sorted question ids.
    let names: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    assert_eq!(names, ["department", "refund", "urgency"]);

    let metadata = schema.metadata();
    assert_eq!(
        metadata.get(crate::arrow::META_ARROW_VERSION).unwrap(),
        crate::arrow::ARROW_MAPPING_VERSION
    );
    assert_eq!(metadata.get(crate::arrow::META_MODEL).unwrap(), "mock");
    // The mock fills usage from the dispatch estimator: deterministic per
    // (state, questions), so the aggregate is exactly twice one evaluation.
    let input: u64 = metadata
        .get(crate::arrow::META_USAGE_INPUT_TOKENS)
        .unwrap()
        .parse()
        .unwrap();
    let output: u64 = metadata
        .get(crate::arrow::META_USAGE_OUTPUT_TOKENS)
        .unwrap()
        .parse()
        .unwrap();
    assert!(input > 0, "aggregate input tokens must be present");
    assert!(output > 0, "aggregate output tokens must be present");
}

#[tokio::test]
async fn arrow_rows_match_systemone_answers() {
    use crate::arrow::answers_from_batch;

    let questions = arrow_questions();
    let single_state = json!({
        "model": "mock",
        "state": "Where is my parcel?",
        "questions": questions
    });
    let resp = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&single_state).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let single: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();

    let (_, batch) = decode_arrow_response(
        post_arrow(
            arrow_app(AuthConfig::default()),
            arrow_request_body(json!(["Where is my parcel?", "Please refund the shoes."])),
        )
        .await,
    )
    .await;
    let rows = answers_from_batch(&batch).unwrap();

    // The mock engine seeds answers per (question id, instructions), so the
    // bulk row and the single-state answer must be the same object. Floats
    // are compared with ulp-scale tolerance: serde_json's decimal parser can
    // drift one ulp on a JSON round-trip, while the Arrow path itself is
    // bit-exact end to end.
    for id in ["refund", "department", "urgency"] {
        let bulk = serde_json::to_value(&rows[0][id]).unwrap();
        assert_json_ulp_close(
            &bulk,
            &single["answers"][id],
            &format!("row 0 answer `{id}`"),
        );
    }
    // Noul answers carry no confidence, on the bulk path either.
    let noul = serde_json::to_value(&rows[0]["refund"]).unwrap();
    assert!(noul.get("confidence").is_none());
}

/// Structural JSON equality with ulp-scale tolerance on floats, so Arrow
/// answers can be diffed against a JSON response that serde_json's parser
/// may have nudged by one ulp.
fn assert_json_ulp_close(left: &serde_json::Value, right: &serde_json::Value, context: &str) {
    match (left, right) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= a.abs().max(b.abs()) * 1e-15,
                "{context}: {a} vs {b}"
            );
        }
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{context}");
            for (index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_json_ulp_close(x, y, &format!("{context}[{index}]"));
            }
        }
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            assert_eq!(a.len(), b.len(), "{context}");
            for (key, x) in a {
                assert_json_ulp_close(x, &b[key], &format!("{context}.{key}"));
            }
        }
        _ => assert_eq!(left, right, "{context}"),
    }
}

#[tokio::test]
async fn arrow_usage_metadata_equals_sum_of_systemone_calls() {
    let state_text = "Where is my parcel?";
    let questions = arrow_questions();
    let single = json!({
        "model": "mock",
        "state": state_text,
        "questions": questions
    });
    let resp = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&single).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let single: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();

    let (schema, _) = decode_arrow_response(
        post_arrow(
            arrow_app(AuthConfig::default()),
            arrow_request_body(json!([state_text, state_text])),
        )
        .await,
    )
    .await;
    let metadata = schema.metadata();
    let expected_input: u64 = 2 * single["usage"]["input_tokens"].as_u64().unwrap();
    let expected_output: u64 = 2 * single["usage"]["output_tokens"].as_u64().unwrap();
    assert_eq!(
        metadata
            .get(crate::arrow::META_USAGE_INPUT_TOKENS)
            .map(String::as_str),
        Some(expected_input.to_string()).as_deref()
    );
    assert_eq!(
        metadata
            .get(crate::arrow::META_USAGE_OUTPUT_TOKENS)
            .map(String::as_str),
        Some(expected_output.to_string()).as_deref()
    );
}

#[tokio::test]
async fn arrow_route_is_gated_by_auth() {
    let router = arrow_app(AuthConfig::new(Some("secret".into())));
    let resp = post_arrow(router.clone(), arrow_request_body(json!(["s"]))).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert!(resp.headers().contains_key("x-typesafe-request-id"));

    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/arrow")
                .header("authorization", "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&arrow_request_body(json!(["s"]))).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn arrow_rejects_empty_questions_and_bad_bodies_with_error_envelopes() {
    for (body, expected_status, expected_code) in [
        (
            json!({"model": "mock", "states": ["s"], "questions": {}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_body",
        ),
        (
            // Score rubrics need at least two levels (core validation).
            json!({"model": "mock", "states": ["s"], "questions": {"one": {"type": "score", "instructions": "x", "criteria": ["only one level"]}}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_body",
        ),
        (
            json!({"model": "mock", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_body",
        ),
    ] {
        let resp = post_arrow(arrow_app(AuthConfig::default()), body).await;
        assert_eq!(resp.status(), expected_status);
        let value: serde_json::Value =
            serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert_eq!(value["error"]["code"], expected_code, "{value}");
    }
}

#[tokio::test]
async fn arrow_choice_over_256_options_is_unprocessable() {
    let mut criteria = serde_json::Map::new();
    for i in 0..257 {
        criteria.insert(format!("option-{i:03}"), serde_json::Value::Null);
    }
    let body = json!({
        "model": "mock",
        "states": ["s"],
        "questions": {
            "wide": {"type": "choice", "instructions": "pick", "criteria": criteria}
        }
    });
    let resp = post_arrow(arrow_app(AuthConfig::default()), body).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let value: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(value["error"]["code"], "invalid_body");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("uint8"),
        "{value}"
    );
}

#[tokio::test]
async fn arrow_unknown_model_fails_as_json_error_not_stream() {
    let mut body = arrow_request_body(json!(["s"]));
    body["model"] = json!("does-not-exist");
    let resp = post_arrow(arrow_app(AuthConfig::default()), body).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/json")
    );
    let value: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(value["error"]["code"], "unknown_model");
}

#[tokio::test]
async fn arrow_empty_states_yield_zero_row_stream() {
    let (schema, batch) = decode_arrow_response(
        post_arrow(
            arrow_app(AuthConfig::default()),
            arrow_request_body(json!([])),
        )
        .await,
    )
    .await;
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(schema.fields().len(), 3);
    assert_eq!(
        schema.metadata().get(crate::arrow::META_MODEL).unwrap(),
        "mock"
    );
    assert_eq!(
        schema
            .metadata()
            .get(crate::arrow::META_USAGE_INPUT_TOKENS)
            .unwrap(),
        "0"
    );
}

#[test]
fn bearer_extraction_handles_prefixes_whitespace_and_garbage() {
    let header = |value: &str| {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            axum::http::HeaderValue::from_str(value).expect("header value"),
        );
        headers
    };
    let no_header = axum::http::HeaderMap::new();
    assert_eq!(bearer_of(&no_header), None);

    assert_eq!(bearer_of(&header("Bearer tok")), Some("tok".to_owned()));
    assert_eq!(bearer_of(&header("bearer tok")), Some("tok".to_owned()));
    assert_eq!(
        bearer_of(&header("Bearer   spaced  ")),
        Some("spaced".to_owned()),
        "the token is trimmed"
    );
    assert_eq!(
        bearer_of(&header("Bearer ")),
        None,
        "an empty token is None"
    );
    assert_eq!(bearer_of(&header("Basic dXNlcjpwYXNz")), None);
    assert_eq!(bearer_of(&header("Bearer")), None);
    // A raw (non-UTF8) header value cannot carry a bearer token.
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::AUTHORIZATION,
        axum::http::HeaderValue::from_bytes(&[0xFF, 0xFE]).expect("opaque bytes"),
    );
    assert_eq!(bearer_of(&headers), None);
}

#[tokio::test]
async fn proxy_credential_isolation_respects_forwards_caller_credentials() {
    struct TestProxy {
        forwards_caller: bool,
        captured_key: std::sync::Mutex<Option<Option<String>>>,
    }

    #[async_trait::async_trait]
    impl crate::proxy::SystemProxy for TestProxy {
        fn wants(&self, req: &openkind_core::SystemRequest) -> bool {
            req.model == "proxied-model"
        }
        fn forwards_caller_credentials(&self) -> bool {
            self.forwards_caller
        }
        async fn evaluate(
            &self,
            _req: openkind_core::SystemRequest,
            caller_key: Option<String>,
        ) -> Result<crate::proxy::ProxyOutcome, ApiError> {
            *self.captured_key.lock().unwrap() = Some(caller_key);
            Ok(crate::proxy::ProxyOutcome {
                response: openkind_core::SystemResponse {
                    model: "upstream-model".into(),
                    answers: [(
                        "q1".into(),
                        openkind_core::Answer::Choice(openkind_core::ChoiceAnswer {
                            choice: "alpha".into(),
                            probabilities: [("alpha".into(), 1.0)].into_iter().collect(),
                            confidence: 1.0,
                        }),
                    )]
                    .into_iter()
                    .collect(),
                    usage: openkind_core::Usage {
                        input_tokens: 0,
                        output_tokens: 0,
                    },
                },
                source: crate::proxy::ProxySource::Upstream,
                detail: None,
            })
        }
        async fn models(&self) -> Option<crate::models::ModelsResponse> {
            None
        }
    }

    let request_body = json!({
        "model": "proxied-model",
        "state": "sample state",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "classify",
                "criteria": {"alpha": "is alpha"}
            }
        }
    });

    // 1. When forwards_caller_credentials is false, inbound bearer token is not extracted or passed.
    {
        let proxy = Arc::new(TestProxy {
            forwards_caller: false,
            captured_key: std::sync::Mutex::new(None),
        });
        let mut state = AppState::new(EngineRegistry::new());
        state.proxy = Some(proxy.clone());
        let auth = AuthConfig::new(Some("local-daemon-key".to_string()));
        let app = router_with_state(state, auth);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/systemone")
                    .header("Authorization", "Bearer local-daemon-key")
                    .header("Content-Type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            *proxy.captured_key.lock().unwrap(),
            Some(None),
            "inbound bearer token must NOT be passed to proxy when forwards_caller_credentials is false"
        );
    }

    // 2. When forwards_caller_credentials is true, inbound bearer token IS passed to proxy.
    {
        let proxy = Arc::new(TestProxy {
            forwards_caller: true,
            captured_key: std::sync::Mutex::new(None),
        });
        let mut state = AppState::new(EngineRegistry::new());
        state.proxy = Some(proxy.clone());
        let app = router_with_state(state, AuthConfig::default());

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/systemone")
                    .header("Authorization", "Bearer caller-upstream-key")
                    .header("Content-Type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            *proxy.captured_key.lock().unwrap(),
            Some(Some("caller-upstream-key".to_string())),
            "inbound bearer token must be passed to proxy when forwards_caller_credentials is true"
        );
    }
}
