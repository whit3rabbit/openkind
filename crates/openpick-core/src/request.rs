//! Request body. Spec: <https://docs.typesafe.ai/api#request-body>

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::question::Question;
use crate::state::State;

/// Wire API version. Bumped on any breaking schema change.
pub const API_VERSION: &str = "jev-compatible-0.1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SystemRequest {
    /// Required. The content to evaluate.
    pub state: State,

    /// Required. `"jev-latest"` or a registered model alias.
    /// We keep this required even though Phase 0 doesn't dispatch — the
    /// engine layer in Phase 2 will use it to pick a backend.
    pub model: String,

    /// Required. Map of user-chosen id → typed question. Keys are NOT sent
    /// to the model and are NOT used in inference (per spec).
    pub questions: HashMap<String, Question>,
}