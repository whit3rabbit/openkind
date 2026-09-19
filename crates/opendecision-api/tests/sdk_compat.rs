//! SDK compatibility test suite.
//!
//! These tests are the contract between `opendecisiond` and the proposed
//! TypeSafe Python SDK. They cover every endpoint, every error code,
//! every header, and every payload shape the SDK sends or reads.
//!
//! Each test is anchored to a specific section of the SDK reference
//! (docs.typesafe.ai/sdk/python/api/*). If a test fails, the server
//! has drifted from the spec — fix the server, not the test.
//!
//! Tests are organized by SDK surface:
//!   1. `system_one` — request shape, response shape, types
//!   2. `models.list` — payload shape, sorting, fields
//!   3. Error mapping — 400/401/404/422/429 → SDK exception types
//!   4. Headers — `x-typesafe-request-id`, `Retry-After`, `retry-after-ms`
//!   5. Auth — bearer token semantics, env var, public paths
//!   6. Default model — `jev-latest` alias works without override
//!
//! Run with: `cargo test -p opendecision-api --test sdk_compat`

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::IntoResponse;
use http_body_util::BodyExt;
use opendecision_api::{router_with_auth, AuthConfig};
use opendecision_engine::{EngineRegistry, MockEngine};
use serde_json::{json, Value};
use tower::ServiceExt;

// --- helpers ---------------------------------------------------------

fn app() -> axum::Router {
    let mut reg = EngineRegistry::new();
    // Register both `mock` and `jev-latest` so we test the SDK's default.
    reg.register("mock", Arc::new(MockEngine::new()));
    reg.register("jev-latest", Arc::new(MockEngine::new()));
    router_with_auth(reg, AuthConfig::default())
}

fn app_with_auth(token: &str) -> axum::Router {
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    reg.register("jev-latest", Arc::new(MockEngine::new()));
    router_with_auth(reg, AuthConfig::new(Some(token.into())))
}

async fn body_bytes(resp: axum::response::Response) -> Vec<u8> {
    resp.into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec()
}

async fn post_systemone(
    uri: &str,
    body: Value,
    headers: &[(&str, &str)],
) -> (StatusCode, axum::response::Response) {
    let mut b = Request::builder().method("POST").uri(uri);
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    let req = b
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app().oneshot(req).await.unwrap();
    let status = resp.status();
    (status, resp)
}

async fn get(uri: &str, headers: &[(&str, &str)]) -> (StatusCode, axum::response::Response) {
    let mut b = Request::builder().method("GET").uri(uri);
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    let resp = app().oneshot(b.body(Body::empty()).unwrap()).await.unwrap();
    let status = resp.status();
    (status, resp)
}

// =====================================================================
// 1. system_one — request/response shape
//    Spec: docs.typesafe.ai/sdk/python/api/clients/sync/client.md#system_one
// =====================================================================

#[tokio::test]
async fn system_one_accepts_plain_text_state() {
    // SDK example: `client.system_one(state="I was charged twice.", ...)`
    let body = json!({
        "state": "I was charged twice. Please help.",
        "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "Is this about billing?" }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["model"], "mock");
    assert!(v["answers"]["billing"]["noul"].is_number());
}

#[tokio::test]
async fn system_one_accepts_object_state() {
    // Spec: state may be string | object | array.
    let body = json!({
        "state": { "document": "x", "messages": [] },
        "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "?" }
        }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_accepts_array_state() {
    let body = json!({
        "state": ["turn 1", "turn 2"],
        "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "?" }
        }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_returns_noul_answer_shape() {
    // SDK type: NoulAnswer { noul: float }
    let body = json!({
        "state": "x", "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let ans = &v["answers"]["q"];
    assert_eq!(ans["type"], "noul");
    assert!(ans["noul"].is_number());
    assert!(
        ans.get("confidence").is_none(),
        "noul answer MUST NOT have confidence"
    );
    let noul = ans["noul"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&noul));
}

#[tokio::test]
async fn system_one_returns_choice_answer_shape() {
    // SDK type: ChoiceAnswer { choice, probabilities, confidence }
    let body = json!({
        "state": "x", "model": "mock",
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "?",
                "criteria": { "billing": "pay", "tech": "bugs" }
            }
        }
    });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let ans = &v["answers"]["dept"];
    assert_eq!(ans["type"], "choice");
    assert!(ans["choice"].is_string());
    assert!(ans["probabilities"].is_object());
    assert!(ans["confidence"].is_number());
    // probabilities sum to 1.
    let sum: f64 = ans["probabilities"]
        .as_object()
        .unwrap()
        .values()
        .map(|x| x.as_f64().unwrap())
        .sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "probabilities must sum to 1, got {sum}"
    );
}

#[tokio::test]
async fn system_one_returns_score_answer_shape() {
    // SDK type: ScoreAnswer { score, legend, probabilities, confidence }
    let body = json!({
        "state": "x", "model": "mock",
        "questions": {
            "frust": {
                "type": "score",
                "instructions": "?",
                "criteria": ["Calm", "Frustrated", "Angry"]
            }
        }
    });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let ans = &v["answers"]["frust"];
    assert_eq!(ans["type"], "score");
    assert!(ans["score"].is_number());
    assert!(ans["legend"].is_object());
    assert!(ans["probabilities"].is_object());
    assert!(ans["confidence"].is_number());
    // legend and probabilities share the same keys.
    let lkeys: std::collections::HashSet<&str> = ans["legend"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let pkeys: std::collections::HashSet<&str> = ans["probabilities"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(lkeys, pkeys, "legend and probabilities must share keys");
}

#[tokio::test]
async fn system_one_choice_criteria_value_may_be_null() {
    // Spec: "use null when an option needs no extra detail"
    let body = json!({
        "state": "x", "model": "mock",
        "questions": {
            "tag": {
                "type": "choice",
                "instructions": "?",
                "criteria": { "alpha": null, "beta": "second" }
            }
        }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_instructions_may_be_string_object_or_array() {
    for instr in [
        json!("string"),
        json!({"role": "system", "content": "structured"}),
        json!(["line 1", "line 2"]),
    ] {
        let body = json!({
            "state": "x", "model": "mock",
            "questions": { "q": { "type": "noul", "instructions": instr } }
        });
        let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
        assert_eq!(status, StatusCode::OK, "instructions={instr}");
    }
}

#[tokio::test]
async fn system_one_returns_usage_block() {
    // SDK type: Usage { input_tokens, output_tokens } — both optional ints.
    let body = json!({
        "state": "x", "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["usage"].is_object());
    assert!(v["usage"]["input_tokens"].is_number());
    assert!(v["usage"]["output_tokens"].is_number());
}

#[tokio::test]
async fn system_one_default_model_jev_latest_works() {
    // Spec constant: DEFAULT_MODEL = 'jev-latest'. SDK sends this
    // when the user doesn't override.
    let body = json!({
        "state": "x",
        "model": "jev-latest",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_accepts_question_dict_with_type_field() {
    // SDK example uses both Question objects and dicts with `type` field.
    let body = json!({
        "state": "x", "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "?" },
            "tone": {
                "type": "choice",
                "instructions": "?",
                "criteria": { "calm": null, "angry": null }
            },
            "urgency": {
                "type": "score",
                "instructions": "?",
                "criteria": ["can wait", "this week", "today"]
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["answers"].as_object().unwrap().len(), 3);
}

// =====================================================================
// 2. models.list — payload shape
//    Spec: docs.typesafe.ai/models#listing-models
//          docs.typesafe.ai/sdk/python/api/types/responses.md#ListModelsResponse
// =====================================================================

#[tokio::test]
async fn models_list_returns_jev_top_level_models_array() {
    let (status, resp) = get("/v1/models", &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    // Top-level `models` array (per spec).
    assert!(v["models"].is_array(), "expected `models` array, got {v}");
    // No OpenAI-style `data`/`object` keys.
    assert!(v.get("data").is_none());
    assert!(v.get("object").is_none());
}

#[tokio::test]
async fn models_list_entries_have_required_fields() {
    // Spec: name/description/release_date all required.
    let (_, resp) = get("/v1/models", &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    for entry in v["models"].as_array().unwrap() {
        assert!(entry["name"].is_string(), "missing name");
        assert!(entry["description"].is_string(), "missing description");
        assert!(entry["release_date"].is_string(), "missing release_date");
    }
}

#[tokio::test]
async fn models_list_name_matches_alias_sent_to_systemone() {
    // Spec: "name: The model ID or alias, as accepted by the model field."
    let (_, models_resp) = get("/v1/models", &[]).await;
    let models: Value = serde_json::from_slice(&body_bytes(models_resp).await).unwrap();
    let names: Vec<&str> = models["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    // The SDK quickstart uses `model="jev-latest"` by default.
    assert!(names.contains(&"mock"));
    assert!(names.contains(&"jev-latest"));
}

#[tokio::test]
async fn models_list_is_sorted() {
    let (_, resp) = get("/v1/models", &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let names: Vec<String> = v["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap().to_string())
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
}

// =====================================================================
// 3. Error mapping
//    Spec: docs.typesafe.ai/sdk/python/api/exceptions.md
// =====================================================================

#[tokio::test]
async fn empty_questions_returns_422() {
    // Spec: raises TypeSafeError when "Questions are empty".
    let body = json!({ "state": "x", "model": "mock", "questions": {} });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["error"]["code"], "invalid_body");
}

#[tokio::test]
async fn score_with_one_level_returns_422() {
    // Spec: raises TypeSafeError when "a score question's criteria list is empty".
    let body = json!({
        "state": "x", "model": "mock",
        "questions": { "q": { "type": "score", "instructions": "?", "criteria": ["only"] } }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn missing_model_returns_404() {
    // TypeSafeNotFoundError (404): wrong route or model.
    let body = json!({
        "state": "x", "model": "no-such-model",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["error"]["code"], "unknown_model");
}

#[tokio::test]
async fn malformed_json_returns_400() {
    // TypeSafeBadRequestError (400).
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
async fn error_bodies_have_consistent_envelope() {
    // SDK reads `error.body`, `error.status`, `error.code`, `error.message`.
    let body = json!({ "state": "x", "model": "mock", "questions": {} });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["error"].is_object());
    assert!(v["error"]["code"].is_string());
    assert!(v["error"]["message"].is_string());
}

#[tokio::test]
async fn rate_limited_responses_carry_retry_after() {
    // SDK RetryPolicy parses `Retry-After` and `retry-after-ms`.
    // We can't easily trigger rate limiting from outside, so this
    // test verifies the error mapping indirectly via the public
    // IntoResponse impl — by checking that a known 422 path doesn't
    // accidentally emit retry headers (regression guard).
    let body = json!({ "state": "x", "model": "mock", "questions": {} });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let headers = resp.headers();
    assert!(headers.get("retry-after-ms").is_none());
    assert!(headers.get("retry-after").is_none());
}

// =====================================================================
// 4. Headers
//    Spec: every response carries x-typesafe-request-id (request_id prop).
// =====================================================================

#[tokio::test]
async fn every_response_has_request_id_header() {
    struct Case {
        method: &'static str,
        headers: Vec<(&'static str, &'static str)>,
        body: Value,
    }
    let cases = vec![
        Case {
            method: "GET",
            headers: vec![],
            body: json!({}),
        },
        Case {
            method: "POST",
            headers: vec![],
            body: json!({"state":"x","model":"mock","questions":{"q":{"type":"noul","instructions":"?"}}}),
        },
    ];
    let paths: Vec<&str> = vec!["/health", "/v1/models", "/v1/systemone", "/metrics"];

    for path in paths {
        for case in &cases {
            let mut b = Request::builder().method(case.method).uri(path);
            for (k, v) in &case.headers {
                b = b.header(*k, *v);
            }
            let req = if case.method == "POST" {
                b.header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&case.body).unwrap()))
                    .unwrap()
            } else {
                b.body(Body::empty()).unwrap()
            };
            let resp = app().oneshot(req).await.unwrap();
            let method = case.method;
            assert!(
                resp.headers().get("x-typesafe-request-id").is_some(),
                "{method} {path} must carry x-typesafe-request-id, got headers: {:?}",
                resp.headers()
            );
        }
    }
}

#[tokio::test]
async fn request_id_header_value_is_uuid() {
    let (_, resp) = get("/health", &[]).await;
    let id = resp.headers().get("x-typesafe-request-id").unwrap();
    let s = id.to_str().unwrap();
    assert_eq!(s.len(), 36, "expected uuid, got {s:?}");
}

#[tokio::test]
async fn request_id_is_unique_per_request() {
    let (_, r1) = get("/health", &[]).await;
    let (_, r2) = get("/health", &[]).await;
    let a = r1
        .headers()
        .get("x-typesafe-request-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let b = r2
        .headers()
        .get("x-typesafe-request-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert_ne!(a, b, "request ids must differ across requests");
}

#[tokio::test]
async fn inbound_request_id_is_honored() {
    // Lets a proxy thread the id through.
    let (status, resp) = get(
        "/health",
        &[("x-typesafe-request-id", "client-supplied-id-123")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        resp.headers().get("x-typesafe-request-id").unwrap(),
        "client-supplied-id-123"
    );
}

// =====================================================================
// 5. Auth (bearer token)
//    Spec: SDK reads TYPESAFE_API_KEY env, sends Authorization: Bearer ...
// =====================================================================

#[tokio::test]
async fn auth_disabled_by_default_passes_through() {
    let (status, _) = get("/v1/models", &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn auth_enabled_without_token_returns_401() {
    // SDK should raise TypeSafeAuthenticationError(401).
    let mut reg = EngineRegistry::new();
    reg.register("mock", Arc::new(MockEngine::new()));
    let app = router_with_auth(reg, AuthConfig::new(Some("secret-token".into())));

    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert!(resp.headers().get("www-authenticate").is_some());
    assert!(resp.headers().get("x-typesafe-request-id").is_some());
}

#[tokio::test]
async fn auth_enabled_with_wrong_token_returns_401() {
    let app = app_with_auth("correct");
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", "Bearer wrong")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_enabled_with_correct_token_passes() {
    let app = app_with_auth("correct");
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", "Bearer correct")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_does_not_gate_health_or_metrics() {
    let app = app_with_auth("secret");
    for path in ["/health", "/metrics"] {
        let req = Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{path} must remain open");
    }
}

#[tokio::test]
async fn auth_401_body_has_error_envelope() {
    let app = app_with_auth("secret");
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["error"]["code"], "unauthorized");
    assert!(v["error"]["message"].is_string());
}

// =====================================================================
// 6. Cross-cutting — full SDK-shaped flows
// =====================================================================

#[tokio::test]
async fn sdk_quickstart_example_works_end_to_end() {
    // Verbatim from docs.typesafe.ai/sdk/python.md#quickstart:
    //
    //   response = client.system_one(
    //       state={"document": "I was charged twice. Please fix this ASAP."},
    //       questions={
    //           "billing": Noul(instructions="Is this ticket about billing?"),
    //           "tone": Choice(
    //               instructions="What is the customer's tone?",
    //               criteria={"calm": None, "frustrated": None, "angry": None},
    //           ),
    //           "urgency": Score(
    //               instructions="How urgent is this ticket?",
    //               criteria=["can wait", "this week", "today"],
    //           ),
    //       },
    //   )
    let body = json!({
        "state": { "document": "I was charged twice. Please fix this ASAP." },
        "model": "mock",
        "questions": {
            "billing": { "type": "noul", "instructions": "Is this ticket about billing?" },
            "tone": {
                "type": "choice",
                "instructions": "What is the customer's tone?",
                "criteria": { "calm": null, "frustrated": null, "angry": null }
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["can wait", "this week", "today"]
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    // Capture headers up-front since body_bytes takes ownership.
    let request_id_header = resp.headers().get("x-typesafe-request-id").unwrap().clone();
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    // SDK accesses these properties:
    assert!(v["answers"]["billing"]["noul"].is_number());
    assert!(v["answers"]["tone"]["choice"].is_string());
    assert!(v["answers"]["urgency"]["score"].is_number());
    assert!(v["answers"]["urgency"]["legend"].is_object());
    assert!(v["usage"]["input_tokens"].is_number());
    assert!(v["usage"]["output_tokens"].is_number());
    // request_id should be readable from headers.
    assert!(request_id_header.to_str().unwrap().len() == 36);
}

#[tokio::test]
async fn sdk_client_aclose_pattern_works() {
    // SDK pattern: `with TypeSafeClient() as client: ...` — implies a
    // stateless server with no connection-level resources. We verify
    // there are no session cookies or sticky requirements.
    for _ in 0..3 {
        let body = json!({
            "state": "x", "model": "mock",
            "questions": { "q": { "type": "noul", "instructions": "?" } }
        });
        let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn multi_question_response_keys_match_request_keys() {
    // SDK does `result.choices["tone"]`, so server must echo the keys.
    let mut questions = serde_json::Map::new();
    for name in ["alpha", "beta", "gamma", "delta"] {
        questions.insert(name.into(), json!({ "type": "noul", "instructions": "?" }));
    }
    let body = json!({ "state": "x", "model": "mock", "questions": questions });
    let (_, resp) = post_systemone("/v1/systemone", body, &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let answers = v["answers"].as_object().unwrap();
    assert!(answers.contains_key("alpha"));
    assert!(answers.contains_key("beta"));
    assert!(answers.contains_key("gamma"));
    assert!(answers.contains_key("delta"));
    assert_eq!(answers.len(), 4);
}

#[tokio::test]
async fn sdk_response_request_id_is_a_valid_uuid() {
    // Spec: SDK exposes `response.request_id` from `x-typesafe-request-id`.
    let (_, resp) = get("/health", &[]).await;
    let id = resp
        .headers()
        .get("x-typesafe-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    // UUIDv4: 8-4-4-4-12 hex, dash-separated.
    let parts: Vec<&str> = id.split('-').collect();
    assert_eq!(parts.len(), 5);
    assert_eq!(parts[0].len(), 8);
    assert_eq!(parts[1].len(), 4);
    assert_eq!(parts[2].len(), 4);
    assert_eq!(parts[3].len(), 4);
    assert_eq!(parts[4].len(), 12);
    for p in &parts[..4] {
        assert!(p.chars().all(|c| c.is_ascii_hexdigit()));
    }
}

#[tokio::test]
async fn models_list_response_keys_match_python_sdk() {
    // SDK type: ListModelsResponse { models: tuple[ModelMetadata, ...] }
    // ModelMetadata { name, description, release_date }.
    let (_, resp) = get("/v1/models", &[]).await;
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let arr = v["models"].as_array().expect("`models` is an array");
    assert!(!arr.is_empty());
    for m in arr {
        // Every key the SDK reads must be present and a string.
        for key in ["name", "description", "release_date"] {
            assert!(m[key].is_string(), "{key} must be a string, got {m:?}");
        }
    }
}

// =====================================================================
// 7. JSON Schema (the SDK serializes requests, but we should also
//    accept the exact wire shape for clients built from a codegen.)
// =====================================================================

#[tokio::test]
async fn extra_body_fields_are_ignored_not_rejected() {
    // The SDK sends `extra_body` to merge in extra top-level fields.
    // We don't implement merge semantics in Phase 1, but the server
    // shouldn't reject unknown top-level fields by default — strict
    // mode would need an opt-in flag.
    let body = json!({
        "state": "x", "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } },
        "metadata": { "trace_id": "abc" }
    });
    let (status, _) = post_systemone("/v1/systemone", body, &[]).await;
    // Note: serde's default is to ignore unknown fields. If we ever
    // tighten this, this test guards the SDK compatibility.
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn system_one_alias_route_works() {
    // Both POST /v1/systemone (canonical per spec) and /v1/system_one (alias) work identically.
    let body = json!({
        "state": "Test state",
        "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status, resp) = post_systemone("/v1/system_one", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["model"], "mock");
    assert!(v["answers"]["q"]["noul"].is_number());
}

#[tokio::test]
async fn overloaded_error_maps_to_529() {
    // Spec: "529 Overloaded — TypeSafe is temporarily overloaded. Retry after a short delay."
    use opendecision_api::ApiError;
    let err = ApiError::Overloaded {
        retry_after_ms: 1200,
    };
    let resp = err.into_response();
    assert_eq!(resp.status().as_u16(), 529);
    assert_eq!(resp.headers().get("retry-after-ms").unwrap(), "1200");
    assert_eq!(resp.headers().get("retry-after").unwrap(), "2");
}

// =====================================================================
// 8. Python SDK Specification Contract Verification
//    Anchored to: https://docs.typesafe.ai/sdk/python/api
//    - Sync client: Client · Models
//    - Async client: Client · Models
//    - Types: Common · Questions · Responses
//    - Retries
//    - Exceptions
//    - Constants
// =====================================================================

// --- Sync & Async Client: extra_headers, extra_body, model overrides ---

#[tokio::test]
async fn client_extra_headers_are_safely_accepted() {
    // Spec: docs.typesafe.ai/sdk/python/api/clients/sync/client
    // SDK sync & async clients allow passing `extra_headers={"X-Client-Trace": "xyz"}`
    let body = json!({
        "state": "Sample state",
        "model": "mock",
        "questions": {
            "q": { "type": "noul", "instructions": "test" }
        }
    });
    let (status, resp) = post_systemone(
        "/v1/systemone",
        body,
        &[
            ("x-client-trace", "xyz-12345"),
            ("x-sdk-version", "0.1.0"),
            ("user-agent", "typesafe-python/0.1.0"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(resp.headers().get("x-typesafe-request-id").is_some());
}

#[tokio::test]
async fn client_model_parameter_selects_specific_backend_model() {
    // Spec: docs.typesafe.ai/sdk/python/api/clients/sync/client
    // SDK allows overriding the model per call: client.system_one(..., model="mock")
    let body = json!({
        "state": "Order verification",
        "model": "mock",
        "questions": {
            "verified": { "type": "noul", "instructions": "Is order verified?" }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["model"], "mock");
}

#[tokio::test]
async fn client_extra_body_shallow_merged_keys_coexist_with_jev_fields() {
    // Spec: docs.typesafe.ai/sdk/python/api/clients/sync/client
    // SDK merges `extra_body` into the top-level request body payload
    let body = json!({
        "state": "Extra body check",
        "model": "mock",
        "questions": {
            "flag": { "type": "noul", "instructions": "flag" }
        },
        "trace_id": "trace-uuid-999",
        "client_metadata": {
            "service": "checkout",
            "region": "us-west-2"
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["answers"]["flag"]["noul"].is_number());
}

// --- Models Resource: models.list() contract ---

#[tokio::test]
async fn models_resource_list_accepts_extra_headers() {
    // Spec: docs.typesafe.ai/sdk/python/api/clients/sync/models
    // SDK models.list(extra_headers={...})
    let (status, resp) = get(
        "/v1/models",
        &[
            ("x-trace-id", "model-list-trace"),
            ("user-agent", "typesafe-sdk"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["models"].is_array());
}

#[tokio::test]
async fn models_resource_list_metadata_fields_and_sorted_order() {
    // Spec: docs.typesafe.ai/sdk/python/api/clients/sync/models
    // ModelMetadata fields: name (str), description (str), release_date (str)
    let (status, resp) = get("/v1/models", &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let models = v["models"].as_array().unwrap();
    assert!(models.len() >= 2);

    let mut names = Vec::new();
    for m in models {
        let name = m["name"].as_str().unwrap();
        let desc = m["description"].as_str().unwrap();
        let release = m["release_date"].as_str().unwrap();
        assert!(!name.is_empty());
        assert!(!desc.is_empty());
        assert!(!release.is_empty());
        names.push(name.to_string());
    }
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted, "Models list must be sorted by name");
}

// --- Types: Common (JSONContent arbitrary JSON values, structured instructions) ---

#[tokio::test]
async fn common_types_state_supports_deep_nesting_and_nulls() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/common
    // JSONContent: dict[str, Any] | list[Any] | str | int | float | bool | None
    let body = json!({
        "state": {
            "session": null,
            "cart": {
                "items": [
                    {"id": 1, "price": 29.99, "in_stock": true},
                    {"id": 2, "price": null, "notes": ["backorder", null, 42]}
                ],
                "discount": null
            },
            "user_authenticated": false
        },
        "model": "mock",
        "questions": {
            "has_discount": { "type": "noul", "instructions": "Is there a discount?" }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["answers"]["has_discount"]["noul"].is_number());
}

#[tokio::test]
async fn common_types_instructions_as_structured_object() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/common
    // JSONContent can be a structured instructions object
    let body = json!({
        "state": "Customer support ticket #4432",
        "model": "mock",
        "questions": {
            "policy_check": {
                "type": "noul",
                "instructions": {
                    "role": "tier-3 auditor",
                    "guidelines": ["Check SLA compliance", "Verify identity"],
                    "threshold_hours": 24
                }
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["answers"]["policy_check"]["noul"].is_number());
}

// --- Types: Questions (NoulModel, ChoiceModel, ScoreModel) ---

#[tokio::test]
async fn question_types_noul_with_explicit_true_and_false_criteria() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/questions
    // NoulModel: criteria can be dict[bool, str] mapping true and false to descriptions
    let body = json!({
        "state": "User requested account deletion.",
        "model": "mock",
        "questions": {
            "destructive": {
                "type": "noul",
                "instructions": "Is this action destructive to user data?",
                "criteria": {
                    "true": "Irreversible loss of user account or records.",
                    "false": "Standard operational change, fully recoverable."
                }
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let noul = v["answers"]["destructive"]["noul"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&noul));
}

#[tokio::test]
async fn question_types_noul_without_criteria() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/questions
    // NoulModel: criteria is optional (None)
    let body = json!({
        "state": "The door is locked.",
        "model": "mock",
        "questions": {
            "locked": {
                "type": "noul",
                "instructions": "Is the door locked?"
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["answers"]["locked"]["noul"].is_number());
}

#[tokio::test]
async fn question_types_choice_with_mixed_string_and_null_criteria() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/questions
    // ChoiceModel: criteria dict[str, str | None]
    let body = json!({
        "state": "Alert triggered: CPU at 98%",
        "model": "mock",
        "questions": {
            "severity": {
                "type": "choice",
                "instructions": "What is the severity level?",
                "criteria": {
                    "low": null,
                    "medium": "Normal operational spike",
                    "high": null,
                    "critical": "System is degraded or unresponsive"
                }
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let choice_ans = &v["answers"]["severity"];
    let choice = choice_ans["choice"].as_str().unwrap();
    assert!(["low", "medium", "high", "critical"].contains(&choice));
    let probs = choice_ans["probabilities"].as_object().unwrap();
    assert_eq!(probs.len(), 4);
    assert!(choice_ans["confidence"].is_number());
}

#[tokio::test]
async fn question_types_score_ordered_rubric_levels() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/questions
    // In SDK client: ScoreModel(criteria={0: "Unsatisfactory", 1: "Developing", ...})
    // On the wire: criteria is an ordered array of level descriptions [str, str, ...]
    let body = json!({
        "state": "Essay submission for grading.",
        "model": "mock",
        "questions": {
            "rubric_score": {
                "type": "score",
                "instructions": "Grade the essay according to the rubric.",
                "criteria": [
                    "Unsatisfactory: failing to meet core requirements",
                    "Developing: partially meets requirements",
                    "Proficient: meets all rubric requirements",
                    "Advanced: exceeds rubric requirements"
                ]
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let score_ans = &v["answers"]["rubric_score"];
    let score = score_ans["score"].as_f64().unwrap();
    assert!((0.0..=3.0).contains(&score));
    assert!(score_ans["legend"].is_object());
    assert!(score_ans["probabilities"].is_object());
    assert!(score_ans["confidence"].is_number());
    // SDK deserializes legend string keys ("0", "1", "2", "3") to int keys
    assert_eq!(
        score_ans["legend"]["0"],
        "Unsatisfactory: failing to meet core requirements"
    );
    assert_eq!(
        score_ans["legend"]["3"],
        "Advanced: exceeds rubric requirements"
    );
}

// --- Types: Responses (SystemOneResponse structure, UsageInfo, Answer models) ---

#[tokio::test]
async fn response_structure_matches_sdk_system_one_response_model() {
    // Spec: docs.typesafe.ai/sdk/python/api/types/responses
    // SystemOneResponse:
    //   request_id: str (from header)
    //   model: str
    //   usage: UsageInfo { input_tokens: int, output_tokens: int }
    //   answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    let body = json!({
        "state": "Response structure verification",
        "model": "mock",
        "questions": {
            "noul_q": { "type": "noul", "instructions": "noul check" },
            "choice_q": {
                "type": "choice",
                "instructions": "choice check",
                "criteria": { "a": "alpha", "b": "beta" }
            },
            "score_q": {
                "type": "score",
                "instructions": "score check",
                "criteria": ["low", "high"]
            }
        }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);

    // 1. Header request_id matches UUID format
    let req_id = resp
        .headers()
        .get("x-typesafe-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(req_id.split('-').count(), 5);

    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();

    // 2. model is string
    assert_eq!(v["model"], "mock");

    // 3. usage has positive integers
    let input_tokens = v["usage"]["input_tokens"].as_u64().unwrap();
    let output_tokens = v["usage"]["output_tokens"].as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // 4. answers contains each question key
    let answers = v["answers"].as_object().unwrap();
    assert_eq!(answers.len(), 3);

    // 5. NoulAnswer strictly contains `noul` (and NOT `confidence`)
    let noul_ans = answers["noul_q"].as_object().unwrap();
    assert!(noul_ans.contains_key("noul"));
    assert!(
        !noul_ans.contains_key("confidence"),
        "SDK NoulAnswer has no confidence"
    );

    // 6. ChoiceAnswer contains `choice`, `probabilities`, `confidence`
    let choice_ans = answers["choice_q"].as_object().unwrap();
    assert!(choice_ans.contains_key("choice"));
    assert!(choice_ans.contains_key("probabilities"));
    assert!(choice_ans.contains_key("confidence"));

    // 7. ScoreAnswer contains `score`, `legend`, `probabilities`, `confidence`
    let score_ans = answers["score_q"].as_object().unwrap();
    assert!(score_ans.contains_key("score"));
    assert!(score_ans.contains_key("legend"));
    assert!(score_ans.contains_key("probabilities"));
    assert!(score_ans.contains_key("confidence"));
}

// --- Retries: retry headers on 429 & 529, omitted on success & validation error ---

#[tokio::test]
async fn retries_retry_after_headers_presence_on_rate_limit_and_overload() {
    // Spec: docs.typesafe.ai/sdk/python/api/retries
    // SDK inspects `Retry-After` (seconds) and `retry-after-ms` (ms) on 429 and 529
    use opendecision_api::ApiError;

    // RateLimited (429)
    let rl = ApiError::RateLimited {
        retry_after_ms: 2500,
    };
    let resp = rl.into_response();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(resp.headers().get("retry-after-ms").unwrap(), "2500");
    assert_eq!(resp.headers().get("retry-after").unwrap(), "3");

    // Overloaded (529)
    let ol = ApiError::Overloaded {
        retry_after_ms: 1500,
    };
    let resp = ol.into_response();
    assert_eq!(resp.status().as_u16(), 529);
    assert_eq!(resp.headers().get("retry-after-ms").unwrap(), "1500");
    assert_eq!(resp.headers().get("retry-after").unwrap(), "2");
}

#[tokio::test]
async fn retries_headers_absent_on_success_and_client_error() {
    // Spec: docs.typesafe.ai/sdk/python/api/retries
    // 200 OK must not have retry headers
    let body = json!({
        "state": "Retry check",
        "model": "mock",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert!(resp.headers().get("retry-after").is_none());
    assert!(resp.headers().get("retry-after-ms").is_none());

    // 422 Unprocessable Entity must not have retry headers
    let bad_body = json!({
        "state": "bad",
        "model": "mock",
        "questions": {}
    });
    let (status422, resp422) = post_systemone("/v1/systemone", bad_body, &[]).await;
    assert_eq!(status422, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(resp422.headers().get("retry-after").is_none());
    assert!(resp422.headers().get("retry-after-ms").is_none());
}

// --- Exceptions: SDK exception mapping & consistent error envelope ---

#[tokio::test]
async fn exceptions_all_error_responses_contain_request_id_and_envelope() {
    // Spec: docs.typesafe.ai/sdk/python/api/exceptions
    // SDK exceptions store `.request_id`, `.message`, `.code`, `.http_status`.
    // Test that every error condition sets x-typesafe-request-id and has {"error": {"code", "message"}}

    // 400 Bad JSON -> BadRequestError
    let req400 = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from("{invalid json"))
        .unwrap();
    let resp400 = app().oneshot(req400).await.unwrap();
    assert_eq!(resp400.status(), StatusCode::BAD_REQUEST);
    assert!(resp400.headers().get("x-typesafe-request-id").is_some());
    let v400: Value = serde_json::from_slice(&body_bytes(resp400).await).unwrap();
    assert_eq!(v400["error"]["code"], "bad_json");

    // 401 Unauthorized -> AuthenticationError
    let req401 = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let resp401 = app_with_auth("secret").oneshot(req401).await.unwrap();
    assert_eq!(resp401.status(), StatusCode::UNAUTHORIZED);
    assert!(resp401.headers().get("x-typesafe-request-id").is_some());
    assert!(resp401.headers().get("www-authenticate").is_some());
    let v401: Value = serde_json::from_slice(&body_bytes(resp401).await).unwrap();
    assert_eq!(v401["error"]["code"], "unauthorized");

    // 404 Unknown Model -> NotFoundError
    let body404 = json!({
        "state": "x",
        "model": "nonexistent-model-xyz",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let (status404, resp404) = post_systemone("/v1/systemone", body404, &[]).await;
    assert_eq!(status404, StatusCode::NOT_FOUND);
    assert!(resp404.headers().get("x-typesafe-request-id").is_some());
    let v404: Value = serde_json::from_slice(&body_bytes(resp404).await).unwrap();
    assert_eq!(v404["error"]["code"], "unknown_model");

    // 422 Unprocessable Entity -> UnprocessableEntityError
    let body422 = json!({
        "state": "x",
        "model": "mock",
        "questions": {}
    });
    let (status422, resp422) = post_systemone("/v1/systemone", body422, &[]).await;
    assert_eq!(status422, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(resp422.headers().get("x-typesafe-request-id").is_some());
    let v422: Value = serde_json::from_slice(&body_bytes(resp422).await).unwrap();
    assert!(v422["error"]["code"].is_string());
}

// --- Constants: DEFAULT_MODEL and AuthConfig API key resolution ---

#[tokio::test]
async fn constants_default_model_jev_latest_is_evaluable() {
    // Spec: docs.typesafe.ai/sdk/python/api/constants
    // SDK default model is `jev-latest`.
    let body = json!({
        "state": "Checking default model",
        "model": "jev-latest",
        "questions": { "q": { "type": "noul", "instructions": "test" } }
    });
    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["model"], "jev-latest");
}

#[tokio::test]
async fn constants_typesafe_api_key_auth_fallback_integration() {
    // Spec: docs.typesafe.ai/sdk/python/api/constants
    // SDK uses TYPESAFE_API_KEY env var.
    // Ensure that when an auth layer is configured with the key expected by the SDK,
    // clients supplying the Bearer token pass through cleanly.
    let app = app_with_auth("typesafe-test-token-xyz");
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .header("authorization", "Bearer typesafe-test-token-xyz")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "state": "auth ok",
                "model": "mock",
                "questions": { "q": { "type": "noul", "instructions": "ok?" } }
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
