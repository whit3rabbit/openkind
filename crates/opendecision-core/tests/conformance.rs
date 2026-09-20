//! Conformance tests against the Jev API contract.
//!
//! Each test is anchored to a specific example or rule from
//! <https://docs.typesafe.ai/api>. If you change a schema field or validator,
//! you should be able to point at a comment in this file that names the spec
//! section it came from.

#[path = "conformance/edge_cases.rs"]
mod edge_cases;
#[path = "conformance/examples.rs"]
mod examples;
#[path = "conformance/schema.rs"]
mod schema;
#[path = "conformance/validation.rs"]
mod validation;
