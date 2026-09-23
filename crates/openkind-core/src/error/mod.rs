//! Validation errors and pure verification routines for the Jev protocol specification.
//!
//! Spec lists four error statuses (401/422/429/529); `422 Unprocessable Entity` is the
//! one this module emits — invalid request or response bodies.

mod types;
mod validate;

#[cfg(test)]
mod tests;

pub use types::{ValidationError, ValidationResult};
pub use validate::{
    validate_request, validate_response, validate_response_for_request, ResponseContract,
    MAX_CRITERIA_OPTIONS, MAX_QUESTIONS_PER_REQUEST,
};
