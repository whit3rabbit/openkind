//! Response body. Spec: <https://docs.typesafe.ai/api#response-body>
//!
//! > One answer per question, returned under the same ids you provided.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::answer::Answer;

/// Token usage. **Required** (per spec — ChatGPT plan said `Option<Usage>`,
/// that's wrong).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SystemResponse {
    /// The model that performed the evaluation. **Required.**
    pub model: String,

    /// One answer per question id. **Required.**
    pub answers: HashMap<String, Answer>,

    /// Required token usage.
    pub usage: Usage,
}
