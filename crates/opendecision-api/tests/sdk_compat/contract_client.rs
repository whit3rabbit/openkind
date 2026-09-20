use axum::http::StatusCode;
use serde_json::{json, Value};

use super::helpers::{body_bytes, get, post_systemone};

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
