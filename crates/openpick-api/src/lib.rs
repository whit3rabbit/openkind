//! openpick-api: HTTP and gRPC protocol layer.
//!
//! Two transports, one engine surface. Both call into
//! `openpick_engine::dispatch` — neither knows what model is running.

pub mod error;
pub mod grpc;
pub mod http;
pub mod middleware;
pub mod models;

pub use error::ApiError;
pub use http::{router, router_with_auth, router_with_state};
pub use middleware::{AuthConfig, REQUEST_ID_HEADER};
pub use models::{ModelInfo, ModelsResponse};

use std::sync::Arc;

use openpick_engine::EngineRegistry;

/// Shared application state passed to every axum handler and tonic method.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<EngineRegistry>,
}

impl AppState {
    pub fn new(registry: EngineRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
        }
    }
}