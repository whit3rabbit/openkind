//! Ports of the behavioral tests from the TypeSafe Python SDK's
//! `tests/test_retry.py` (`github.com/typesafe-ai/typesafe-sdk-python`).
//!
//! Where the Python suite asserts values handed to a mocked sleep, these
//! ports assert wall-clock bounds with millisecond-scale delays; semantics
//! (attempt counts, retry-count header sequences, precedence rules) are
//! asserted exactly.

mod common;

#[path = "sdk_parity_retry/status.rs"]
mod status;

#[path = "sdk_parity_retry/overrides.rs"]
mod overrides;

#[path = "sdk_parity_retry/delays.rs"]
mod delays;

#[path = "sdk_parity_retry/exhaustion.rs"]
mod exhaustion;

#[path = "sdk_parity_retry/concurrency.rs"]
mod concurrency;
