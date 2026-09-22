//! `openkind-core`: Zero-dependency, Jev-compatible wire protocol types and validation.
//!
//! # Overview
//! `openkind-core` is the foundational crate in the `openkind` workspace. It acts as the
//! canonical, single source of truth for the wire schema spoken by both HTTP/REST
//! (`/v1/systemone` and `/v1/system_one`) and gRPC (`openkind.SystemOne/Evaluate`) transports.
//!
//! Pinned to the TypeSafe Jev API contract documented at <https://docs.typesafe.ai/api> and
//! the Python SDK specification at <https://docs.typesafe.ai/sdk/python/api>.
//!
//! # Architectural Invariants
//! - **Strict Dependency Layering**: `openkind-core` depends on no other workspace crates (`openkind-engine`,
//!   `openkind-api`, etc.). All higher layers depend on `openkind-core`.
//! - **64-bit IEEE 754 Precision**: All floating-point fields (`probabilities`, `score`, `noul`, `confidence`)
//!   use `f64` to prevent wire representation regressions (such as `0.92f32` round-tripping to `0.9200000166893005`).
//! - **Zero Confidence on Noul**: Per the Jev specification, boolean probability (`Noul`) answers contain only
//!   `noul` and never include a `confidence` field.
//! - **Polymorphic State & Instructions**: `state` and question `instructions` allow plain text strings, JSON
//!   objects, or JSON arrays.
//!
//! See `docs/ARCHITECTURE.md` and `crates/openkind-core/schemas/` for generated JSON Schema definitions.

#![warn(missing_docs)]

/// Wire answer types for Noul, Choice, and Score evaluations.
pub mod answer;
/// Request and response schema validation errors and checks.
pub mod error;
/// Model metadata structures for `/v1/models` discovery.
pub mod models;
/// Question descriptors and rubric criteria representations.
pub mod question;
/// System evaluation request envelope and versioning constants.
pub mod request;
/// System evaluation response payload and usage metrics.
pub mod response;
/// Flexible evaluation state input (text, structured object, or array).
pub mod state;

pub use answer::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
pub use error::{
    validate_request, validate_response, ValidationError, ValidationResult, MAX_CRITERIA_OPTIONS,
    MAX_QUESTIONS_PER_REQUEST,
};
pub use models::{ModelInfo, ModelsResponse};
pub use question::{ChoiceQuestion, NoulCriteria, NoulQuestion, Question, ScoreQuestion};
pub use request::{SystemRequest, API_VERSION};
pub use response::{SystemResponse, Usage};
pub use state::State;

/// Current API version constant. Bumped when the wire schema breaks compat.
pub const fn api_version() -> &'static str {
    API_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_version_constant_is_stable() {
        // Pin the wire version. Changing this is a breaking change.
        assert_eq!(api_version(), "jev-compatible-0.1");
    }
}
