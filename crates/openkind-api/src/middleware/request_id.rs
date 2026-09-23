//! Request ID header stamping and sanitization middleware.

use axum::{
    body::Body,
    http::{HeaderName, HeaderValue, Request},
    middleware::Next,
    response::Response,
};

/// Name of the request-id header. Mirrors the SDK's `request_id` property.
pub const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-typesafe-request-id");

/// Maximum allowed length for an inbound client request ID.
pub const MAX_REQUEST_ID_LEN: usize = 128;

/// Validate whether a request ID string contains only safe identifier characters.
pub fn is_safe_request_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_REQUEST_ID_LEN
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

/// Stored in request extensions by `request_id_layer`.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

/// Stack-format a UUIDv4 without the `Display` machinery, which dominates
/// this layer's per-request cost.
fn new_request_id() -> String {
    let uuid = uuid::Uuid::new_v4();
    let mut buffer = [0u8; uuid::fmt::Hyphenated::LENGTH];
    uuid.hyphenated().encode_lower(&mut buffer);
    // UUIDs are ASCII by construction; the check is free compared to fmt.
    std::str::from_utf8(&buffer)
        .expect("hyphenated UUID is ASCII")
        .to_owned()
}

/// Stackable middleware function: stamp every response with a request id.
pub async fn request_id_layer(mut req: Request<Body>, next: Next) -> Response {
    // Honor an inbound id if the client supplied a valid and safe one
    // (lets a proxy thread the id through); otherwise mint a fresh UUIDv4.
    let id = req
        .headers()
        .get(&REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| is_safe_request_id(s))
        .map(|s| s.to_string())
        .unwrap_or_else(new_request_id);

    req.extensions_mut().insert(RequestId(id.clone()));

    let mut resp = next.run(req).await;
    if let Ok(v) = HeaderValue::from_str(&id) {
        resp.headers_mut().insert(REQUEST_ID_HEADER.clone(), v);
    }
    resp
}
