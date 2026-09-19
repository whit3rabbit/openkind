//! `opendecision-server`: Daemon library and entrypoint for `opendecisiond`.
//!
//! # Architecture & Responsibilities
//! `opendecision-server` packages the `opendecisiond` daemon binary.
//! Per `docs/ARCHITECTURE.md`, it composes:
//! - Configuration parsing via `clap`
//! - Structured tracing via `tracing-subscriber` (env-filtered)
//! - Shared [`opendecision_engine::EngineRegistry`] instance
//! - Concurrent HTTP server ([`opendecision_api::http`]) on `--http-addr`
//! - Concurrent gRPC server ([`opendecision_api::grpc`]) on `--grpc-addr`
//! - Coordinated graceful shutdown across both listeners on SIGINT / SIGTERM

#![warn(missing_docs)]

pub use opendecision_api::{grpc, http, AppState, AuthConfig};
pub use opendecision_engine::{DecisionEngine, EngineRegistry, MockEngine};
