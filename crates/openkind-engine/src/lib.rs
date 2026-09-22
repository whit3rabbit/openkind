//! `openkind-engine`: Runtime-agnostic decision engine abstractions and dispatch.
//!
//! # Architecture & Responsibilities
//! `openkind-engine` defines the central execution abstractions for Jev-compatible decision inference.
//! Per `docs/ARCHITECTURE.md`, the engine layer sits between `openkind-core` and the transport
//! layers (`openkind-api`):
//!
//! `core` ⇐ `engine` ⇐ `api` ⇐ `server/cli`.
//!
//! The engine abstraction is completely transport-agnostic: it accepts a [`SystemRequest`] and returns
//! a [`SystemResponse`], remaining oblivious to whether evaluation was triggered over HTTP/REST or gRPC.
//!
//! # Core Components
//! - [`DecisionEngine`]: Trait implemented by inference backends (e.g. [`MockEngine`], Candle, GGUF/llama.cpp, ONNX).
//! - [`EngineRegistry`]: Thread-safe mapping of public model aliases (e.g. `"jev-latest"`, `"mock"`) to engine instances.
//! - [`dispatch`]: Unified entrypoint that orchestrates validation, metrics recording, token estimation, and backend evaluation.

#![warn(missing_docs)]

/// Decision request dispatching, validation, and token accounting.
pub mod dispatch;
/// DecisionEngine trait and token estimation helpers.
pub mod engine;
/// Engine-specific error types and result aliases.
pub mod error;
/// In-memory mock decision engine implementation for testing.
pub mod mock;
/// Model execution profile contracts and parity specifications.
pub mod profile;
/// Thread-safe registry mapping model aliases to decision engines.
pub mod registry;

#[cfg(test)]
mod tests;

pub use dispatch::dispatch;
pub use engine::DecisionEngine;
pub use error::{EngineError, EngineResult};
pub use mock::MockEngine;
pub use profile::{
    ArtifactIdentity, ExecutionSemantics, ModelExecutionProfile, ParityContract, ProbabilitySpace,
    ProfileSource, ProfileValidationError,
};
pub use registry::EngineRegistry;
