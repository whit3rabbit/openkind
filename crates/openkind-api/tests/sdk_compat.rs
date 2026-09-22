//! SDK compatibility test suite.
//!
//! These tests are the contract between `openkindd` and the proposed
//! TypeSafe Python SDK. They cover every endpoint, every error code,
//! every header, and every payload shape the SDK sends or reads.
//!
//! Each test is anchored to a specific section of the SDK reference
//! (docs.typesafe.ai/sdk/python/api/*). If a test fails, the server
//! has drifted from the spec — fix the server, not the test.
//!
//! Tests are organized by SDK surface:
//!   1. `system_one` — request shape, response shape, types
//!   2. `models.list` — payload shape, sorting, fields
//!   3. Error mapping — 400/401/404/422/429 → SDK exception types
//!   4. Headers — `x-typesafe-request-id`, `Retry-After`, `retry-after-ms`
//!   5. Auth — bearer token semantics, env var, public paths
//!   6. Default model — `jev-latest` alias works without override
//!   7. SDK contract — client options, types, models, responses, retries, errors
//!
//! Run with: `cargo test -p openkind-api --test sdk_compat`

#[path = "sdk_compat/helpers.rs"]
mod helpers;

#[path = "sdk_compat/system_one.rs"]
mod system_one;

#[path = "sdk_compat/models.rs"]
mod models;

#[path = "sdk_compat/errors.rs"]
mod errors;

#[path = "sdk_compat/headers.rs"]
mod headers;

#[path = "sdk_compat/auth.rs"]
mod auth;

#[path = "sdk_compat/flows.rs"]
mod flows;

#[path = "sdk_compat/contract_client.rs"]
mod contract_client;

#[path = "sdk_compat/contract_types.rs"]
mod contract_types;

#[path = "sdk_compat/contract_errors.rs"]
mod contract_errors;

#[path = "sdk_compat/openapi.rs"]
mod openapi;
