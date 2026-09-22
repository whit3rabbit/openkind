use axum::http::StatusCode;
use serde_json::Value;

use super::helpers::{body_bytes, get};

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
