//! `opendecision-api`: HTTP and gRPC transport protocols, middleware, and SDK compatibility surface.
//!
//! # Architecture & Responsibilities
//! `opendecision-api` provides dual transport interfaces for `opendecision`:
//! - **HTTP/REST Transport** ([`http`]): Axum 0.8 router serving `POST /v1/systemone` (canonical),
//!   `POST /v1/system_one` (SDK alias), `GET /v1/models`, `GET /health`, and `GET /metrics`.
//! - **gRPC Transport** ([`grpc`]): Tonic 0.14 service implementing `opendecision.SystemOne/Evaluate`.
//!
//! Both transports route evaluation requests through `opendecision_engine::dispatch`, decoupling transport
//! encoding from inference backend execution.
//!
//! # Middleware & Wire Compatibility
//! As specified in `docs/ARCHITECTURE.md`:
//! - Outermost Request ID layer stamps `x-typesafe-request-id` on every response, including errors and 401s.
//! - Constant-time Bearer token gate on `/v1/*` routes when `OPENDECISION_API_KEY` (or `TYPESAFE_API_KEY`) is configured.
//! - Error mapping with `Retry-After` and `retry-after-ms` headers on rate limits (429) and overload (529).

#![warn(missing_docs)]

/// HTTP error mapping and Axum response conversion.
pub mod error;
/// Tonic gRPC service implementation for `opendecision.SystemOne`.
pub mod grpc;
/// Axum HTTP router and endpoint handlers.
pub mod http;
/// Request ID, authentication, and rate limiting middleware.
pub mod middleware;
/// Model metadata structures and descriptors.
pub mod models;

pub use error::ApiError;
pub use http::{router, router_with_auth, router_with_state};
pub use middleware::{AuthConfig, RateLimitConfig, RateLimiter, REQUEST_ID_HEADER};
pub use models::{ModelInfo, ModelsResponse};

use std::sync::Arc;

use opendecision_engine::EngineRegistry;

/// Shared application state passed to every axum handler and tonic method.
#[derive(Clone)]
pub struct AppState {
    /// Thread-safe registry mapping model aliases to their decision engine instances.
    pub registry: Arc<EngineRegistry>,
}

impl AppState {
    /// Construct a new `AppState` wrapping the given engine registry.
    pub fn new(registry: EngineRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
        }
    }
}
