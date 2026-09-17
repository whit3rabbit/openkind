//! `state` field — the content to evaluate.
//!
//! From the spec:
//! > `state`: string | object | array (required)
//! > The content to evaluate. A plain string for text, or structured data
//! > (object/array) for things like chat logs, records, or the current
//! > state of your application.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// State to evaluate. Permissive JSON shape: any string, object, or array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum State {
    /// Plain text.
    Text(String),
    /// Structured object (e.g. `{ "user": {...}, "logs": [...], "diff": "..." }`).
    Object(serde_json::Map<String, serde_json::Value>),
    /// Ordered list (e.g. chat log entries).
    Array(Vec<serde_json::Value>),
}