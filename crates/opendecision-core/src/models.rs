//! `ModelInfo` and `ModelsResponse` — match the wire shape on
//! docs.typesafe.ai/models#listing-models.
//!
//! > `GET /v1/models` returns the names your account can send in the
//! > `model` field, with a description and release date for each. It
//! > currently lists the aliases.
//!
//! Response shape: `{ "models": [ { "name", "description", "release_date" } ] }`

use serde::{Deserialize, Serialize};

/// One entry in the `models` array. Fields are required and order-stable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// The model ID or alias, as accepted by the `model` field.
    pub name: String,
    /// What the model is for.
    pub description: String,
    /// ISO-8601 release date (e.g. `"2024-09-17"`).
    pub release_date: String,
}

/// Top-level `GET /v1/models` response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelsResponse {
    /// List of available model descriptors and aliases.
    pub models: Vec<ModelInfo>,
}

impl ModelsResponse {
    /// Construct a new `ModelsResponse` from a list of model infos.
    pub fn new(models: Vec<ModelInfo>) -> Self {
        Self { models }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_to_jev_shape() {
        let resp = ModelsResponse::new(vec![ModelInfo {
            name: "mock".into(),
            description: "A mock engine for testing".into(),
            release_date: "2026-01-01".into(),
        }]);
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["models"][0]["name"], "mock");
        assert_eq!(v["models"][0]["description"], "A mock engine for testing");
        assert_eq!(v["models"][0]["release_date"], "2026-01-01");
        assert!(v.get("data").is_none());
        assert!(v.get("object").is_none());
    }
}
