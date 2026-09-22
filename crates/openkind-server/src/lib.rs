//! `openkind-server`: Daemon library and entrypoint for `openkindd`.
//!
//! # Architecture & Responsibilities
//! `openkind-server` packages the `openkindd` daemon binary.
//! Per `docs/ARCHITECTURE.md`, it composes:
//! - Configuration parsing via `clap`
//! - Structured tracing via `tracing-subscriber` (env-filtered)
//! - Shared [`openkind_engine::EngineRegistry`] instance
//! - Concurrent HTTP server ([`openkind_api::http`]) on `--http-addr`
//! - Concurrent gRPC server ([`openkind_api::grpc`]) on `--grpc-addr`
//! - Coordinated graceful shutdown across both listeners on SIGINT / SIGTERM

#![warn(missing_docs)]

pub use openkind_api::{grpc, http, AppState, AuthConfig};
pub use openkind_engine::{DecisionEngine, EngineRegistry, MockEngine};
