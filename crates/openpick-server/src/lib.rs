//! `openpick-server`: Daemon library and entrypoint for `openpickd`.
//!
//! # Architecture & Responsibilities
//! `openpick-server` packages the `openpickd` daemon binary.
//! Per `docs/ARCHITECTURE.md`, it composes:
//! - Configuration parsing via `clap`
//! - Structured tracing via `tracing-subscriber` (JSON or pretty formats)
//! - Shared [`openpick_engine::EngineRegistry`] instance
//! - Concurrent HTTP server ([`openpick_api::http`]) on `--http-addr`
//! - Concurrent gRPC server ([`openpick_api::grpc`]) on `--grpc-addr`
//! - Coordinated graceful shutdown across both listeners on SIGINT / SIGTERM

#![warn(missing_docs)]

pub use openpick_api::{grpc, http, AppState, AuthConfig};
pub use openpick_engine::{DecisionEngine, EngineRegistry, MockEngine};
