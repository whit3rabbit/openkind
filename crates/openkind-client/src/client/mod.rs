//! The async [`Client`] for the openkind / TypeSafe SystemOne HTTP API.
//!
//! - **Transports**: `POST /v1/systemone` (canonical) and `GET /v1/models`;
//!   `GET /health` for liveness. Wire-compatible with `https://api.typesafe.ai`,
//!   so the same client targets either the hosted service or a local
//!   `openkindd` daemon by swapping `base_url`.
//! - **Retries**: automatic, per [`crate::retry::RetryPolicy`] — 429/529 carry
//!   `Retry-After` and are honored; transient 5xx, connection, and timeout
//!   errors back off exponentially.
//! - **Errors**: typed per [`crate::error`].

mod builder;
mod core;
mod http1;
mod options;
mod transport;

#[cfg(test)]
mod tests;

pub use builder::ClientBuilder;
pub use core::Client;
pub use options::{
    Health, IntoState, RequestOptions, DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT,
};
pub use transport::MAX_RESPONSE_BODY_SIZE;
