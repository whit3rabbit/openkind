//! Response body. Spec: <https://docs.typesafe.ai/api#response-body>
//!
//! > One answer per question, returned under the same ids you provided.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::answer::Answer;
use crate::request::WireHashState;

/// Token usage accounting for the evaluation request.
///
/// **Required** per the Jev wire specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Usage {
    /// Number of tokens in the input state and question instructions.
    pub input_tokens: u32,
    /// Number of tokens accounted to generating the evaluated judgments.
    pub output_tokens: u32,
}

/// Evaluation response payload containing answers to all submitted questions.
///
/// See <https://docs.typesafe.ai/api#response-body> for the canonical wire specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SystemResponse {
    /// The model backend that performed the evaluation. **Required.**
    pub model: String,

    /// One answer per question id. **Required.**
    pub answers: HashMap<String, Answer, WireHashState>,

    /// Required token usage.
    pub usage: Usage,
}
