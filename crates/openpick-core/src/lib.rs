//! openpick-core: Jev-compatible protocol types.
//!
//! Pinned to the TypeSafe Jev API contract documented at
//! <https://docs.typesafe.ai/api>. Every type, field, and validation rule
//! here is derived from a specific sentence or example in that spec — see
//! the `validate_*` functions and the conformance tests in `tests/`.

pub mod answer;
pub mod error;
pub mod models;
pub mod question;
pub mod request;
pub mod response;
pub mod state;

pub use answer::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
pub use error::{validate_request, validate_response, ValidationError, ValidationResult};
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