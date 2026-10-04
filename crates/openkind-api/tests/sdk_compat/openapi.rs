use axum::http::StatusCode;
use serde_json::Value;

use super::helpers::{body_bytes, get, post_systemone};

/// Verifies that all endpoints declared in `openapi.yaml` are implemented and live in the router.
#[tokio::test]
async fn openapi_all_paths_are_registered_in_router() {
    let spec: Value = serde_yaml_ng::from_str(include_str!("../../openapi.yaml")).unwrap();
    let expected = [
        ("/v1/systemone", "post"),
        ("/v1/system_one", "post"),
        ("/v1/models", "get"),
        ("/health", "get"),
        ("/metrics", "get"),
    ];
    assert_eq!(spec["paths"].as_object().unwrap().len(), expected.len());
    for (path, method) in expected {
        let methods = spec["paths"][path].as_object().unwrap();
        assert_eq!(
            methods.keys().map(String::as_str).collect::<Vec<_>>(),
            vec![method]
        );
        assert!(methods[method]["responses"]["200"].is_object());
    }
    for path in ["/v1/systemone", "/v1/system_one"] {
        for (status, component) in [
            ("400", "BadRequest"),
            ("401", "Unauthorized"),
            ("404", "NotFound"),
            ("413", "PayloadTooLarge"),
            ("422", "UnprocessableEntity"),
            ("429", "RateLimited"),
            ("500", "InternalServerError"),
            ("502", "BadGateway"),
            ("504", "GatewayTimeout"),
            ("529", "Overloaded"),
        ] {
            let reference = format!("#/components/responses/{component}");
            assert_eq!(
                spec["paths"][path]["post"]["responses"][status]["$ref"],
                reference
            );
            assert_eq!(
                spec["components"]["responses"][component]["content"]["application/json"]["schema"]
                    ["$ref"],
                "#/components/schemas/ErrorEnvelope"
            );
        }
    }
    assert_eq!(
        spec["paths"]["/v1/systemone"]["post"]["responses"],
        spec["paths"]["/v1/system_one"]["post"]["responses"]
    );

    // Verify GET /health
    let (health_status, health_resp) = get("/health", &[]).await;
    assert_eq!(health_status, StatusCode::OK);
    let health_v: Value = serde_json::from_slice(&body_bytes(health_resp).await).unwrap();
    assert_eq!(health_v["status"], "ok");

    // Verify GET /metrics
    let (metrics_status, _) = get("/metrics", &[]).await;
    assert_eq!(metrics_status, StatusCode::OK);

    // Verify GET /v1/models
    let (models_status, models_resp) = get("/v1/models", &[]).await;
    assert_eq!(models_status, StatusCode::OK);
    let models_v: Value = serde_json::from_slice(&body_bytes(models_resp).await).unwrap();
    assert!(models_v["models"].is_array());
}

/// Verifies that the quickstart Noul example in `openapi.yaml` evaluates successfully.
#[tokio::test]
async fn openapi_example_quickstart_noul_evaluates() {
    let body = serde_json::json!({
        "state": "I was charged twice for order #1042.",
        "model": "mock",
        "questions": {
            "billing": {
                "type": "noul",
                "instructions": "Is this inquiry related to a billing issue?"
            }
        }
    });

    // Validate using openkind_core
    let req: openkind_core::SystemRequest = serde_json::from_value(body.clone()).unwrap();
    openkind_core::validate_request(&req).unwrap();

    // Verify canonical path /v1/systemone
    let (status1, resp1) = post_systemone("/v1/systemone", body.clone(), &[]).await;
    assert_eq!(status1, StatusCode::OK);
    let v1: Value = serde_json::from_slice(&body_bytes(resp1).await).unwrap();
    assert!(v1["answers"]["billing"]["noul"].is_number());
    assert_eq!(v1["answers"]["billing"]["type"], "noul");
    assert!(
        v1["answers"]["billing"].get("confidence").is_none(),
        "Noul has strictly no confidence"
    );

    // Verify alias path /v1/system_one
    let (status2, resp2) = post_systemone("/v1/system_one", body, &[]).await;
    assert_eq!(status2, StatusCode::OK);
    let v2: Value = serde_json::from_slice(&body_bytes(resp2).await).unwrap();
    assert!(v2["answers"]["billing"]["noul"].is_number());
}

/// Verifies that the multi-question example in `openapi.yaml` (Noul, Choice, Score) evaluates successfully.
#[tokio::test]
async fn openapi_example_multi_question_evaluation_evaluates() {
    let body = serde_json::json!({
        "state": "Customer support transcript: Agent resolved issue in 4 minutes.",
        "model": "mock",
        "questions": {
            "is_resolved": {
                "type": "noul",
                "instructions": "Was the issue resolved?",
                "criteria": {
                    "true": "Customer issue was successfully resolved.",
                    "false": "Issue remains unresolved or escalated."
                }
            },
            "department": {
                "type": "choice",
                "instructions": "Which team handled this request?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": null
                }
            },
            "satisfaction": {
                "type": "score",
                "instructions": "Rate customer satisfaction",
                "criteria": [
                    "Dissatisfied",
                    "Neutral",
                    "Delighted"
                ]
            }
        }
    });

    let req: openkind_core::SystemRequest = serde_json::from_value(body.clone()).unwrap();
    openkind_core::validate_request(&req).unwrap();

    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();

    // Verify Noul
    assert!(v["answers"]["is_resolved"]["noul"].is_number());
    assert!(v["answers"]["is_resolved"].get("confidence").is_none());

    // Verify Choice
    assert!(v["answers"]["department"]["choice"].is_string());
    assert!(v["answers"]["department"]["probabilities"].is_object());
    assert!(v["answers"]["department"]["confidence"].is_number());

    // Verify Score
    assert!(v["answers"]["satisfaction"]["score"].is_number());
    assert_eq!(v["answers"]["satisfaction"]["legend"]["0"], "Dissatisfied");
    assert_eq!(v["answers"]["satisfaction"]["legend"]["1"], "Neutral");
    assert_eq!(v["answers"]["satisfaction"]["legend"]["2"], "Delighted");
    assert!(v["answers"]["satisfaction"]["probabilities"].is_object());
    assert!(v["answers"]["satisfaction"]["confidence"].is_number());

    // Verify Usage
    assert!(v["usage"]["input_tokens"].is_number());
    assert!(v["usage"]["output_tokens"].is_number());
}

/// Verifies that structured object state and structured instructions from `openapi.yaml` evaluate successfully.
#[tokio::test]
async fn openapi_example_structured_state_and_instructions_evaluates() {
    let body = serde_json::json!({
        "state": {
            "customer_id": "cust_9812",
            "tier": "enterprise",
            "message": "Can you guarantee 99.999% uptime for our dedicated instance?",
            "metadata": {
                "region": "us-east-1"
            }
        },
        "model": "mock",
        "questions": {
            "sla_guarantee": {
                "type": "noul",
                "instructions": {
                    "role": "compliance auditor",
                    "policy_reference": "SLA Section 4.1",
                    "question": "Does the customer message request five-nines uptime guarantee?"
                }
            },
            "risk_level": {
                "type": "score",
                "instructions": "Evaluate contractual risk tier",
                "criteria": [
                    "Standard SLA terms",
                    "Custom terms requiring legal review",
                    "Unfulfillable guarantee"
                ]
            }
        }
    });

    let req: openkind_core::SystemRequest = serde_json::from_value(body.clone()).unwrap();
    openkind_core::validate_request(&req).unwrap();

    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["answers"]["sla_guarantee"]["noul"].is_number());
    assert!(v["answers"]["risk_level"]["score"].is_number());
}

/// Verifies that the extra_body metadata example from `openapi.yaml` evaluates successfully.
#[tokio::test]
async fn openapi_example_extra_body_metadata_evaluates() {
    let body = serde_json::json!({
        "state": "Deployment pipeline failed during container build step.",
        "model": "mock",
        "questions": {
            "infra_issue": {
                "type": "noul",
                "instructions": "Is this an infrastructure failure?"
            }
        },
        "trace_id": "trace-99210-abcdef",
        "client_metadata": {
            "service": "ci-watcher",
            "environment": "production"
        }
    });

    let (status, resp) = post_systemone("/v1/systemone", body, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["answers"]["infra_issue"]["noul"].is_number());
}

#[tokio::test]
async fn documented_error_responses_are_executable_on_both_aliases() {
    use async_trait::async_trait;
    use axum::{body::Body, http::Request};
    use openkind_api::{
        http, ApiError, AppState, AuthConfig, ModelsResponse, ProxyOutcome, SystemProxy,
    };
    use openkind_core::{SystemRequest, SystemResponse};
    use openkind_engine::{DecisionEngine, EngineError, EngineRegistry, EngineResult};
    use std::sync::Arc;
    use tower::ServiceExt;

    struct DeadlineEngine;
    #[async_trait]
    impl DecisionEngine for DeadlineEngine {
        fn backend_id(&self) -> &str {
            "deadline"
        }
        async fn evaluate(&self, _: SystemRequest) -> EngineResult<SystemResponse> {
            Err(EngineError::DeadlineExceeded {
                backend: "deadline".into(),
                timeout_ms: 1,
            })
        }
    }
    struct FailedProxy;
    #[async_trait]
    impl SystemProxy for FailedProxy {
        fn wants(&self, _: &SystemRequest) -> bool {
            true
        }
        async fn evaluate(
            &self,
            _: SystemRequest,
            _: Option<String>,
        ) -> Result<ProxyOutcome, ApiError> {
            Err(ApiError::BadGateway("upstream unavailable".into()))
        }
        async fn models(&self) -> Option<ModelsResponse> {
            None
        }
    }
    let mut registry = EngineRegistry::new();
    registry.register("mock", Arc::new(DeadlineEngine));
    let deadline = AppState::new(registry.clone());
    let mut proxy = AppState::new(registry);
    proxy.proxy = Some(Arc::new(FailedProxy));
    let spec: Value = serde_yaml_ng::from_str(include_str!("../../openapi.yaml")).unwrap();
    for path in ["/v1/systemone", "/v1/system_one"] {
        for (status, code, state, limit) in [
            (413, "payload_too_large", deadline.clone(), 1),
            (
                502,
                "bad_gateway",
                proxy.clone(),
                http::MAX_PAYLOAD_SIZE_BYTES,
            ),
            (
                504,
                "deadline_exceeded",
                deadline.clone(),
                http::MAX_PAYLOAD_SIZE_BYTES,
            ),
        ] {
            let app = http::router_with_state_and_limit(state, AuthConfig::default(), limit);
            let request = Request::builder().method("POST").uri(path).header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&serde_json::json!({"model":"mock","state":"state","questions":{"q":{"type":"noul","instructions":"?"}}})).unwrap())).unwrap();
            let response = app.oneshot(request).await.unwrap();
            assert_eq!(response.status().as_u16(), status);
            assert!(response.headers().contains_key("x-typesafe-request-id"));
            let body: Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
            assert_eq!(body["error"]["code"], code);
            let reference = spec["paths"][path]["post"]["responses"][status.to_string()]["$ref"]
                .as_str()
                .unwrap();
            let component = reference.strip_prefix("#/components/responses/").unwrap();
            assert_eq!(
                spec["components"]["responses"][component]["content"]["application/json"]
                    ["example"]["error"]["code"],
                code
            );
        }
    }
}
