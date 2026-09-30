//! Cross-cutting middleware for the HTTP layer:
//! - `request_id_layer` — emits an `x-typesafe-request-id` header on
//!   every response (UUIDv4, generated server-side; the SDK reads this).
//! - `auth_layer` — optional `Authorization: Bearer *** gate.
//! - `rate_limit_layer` — fixed-window client-IP rate limiter.

mod auth;
mod rate_limit;
mod request_id;

#[cfg(test)]
mod tests;

pub use auth::{auth_layer, auth_layer_for, secure_token_eq, AuthConfig, AUTH_HEADER};
#[doc(hidden)]
pub use rate_limit::RateLimitContext;
pub use rate_limit::{rate_limit_layer, RateLimitConfig, RateLimiter};
pub use request_id::{
    is_safe_request_id, request_id_layer, RequestId, MAX_REQUEST_ID_LEN, REQUEST_ID_HEADER,
};
