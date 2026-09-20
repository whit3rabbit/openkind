//! Integration test: spin up a real gRPC server bound to an ephemeral
//! port, connect with the generated client, and round-trip a request.
//!
//! Catches the "builds but doesn't speak to itself" class of bug — both
//! halves of the wire have to agree on every field.

#[path = "grpc_roundtrip/helpers.rs"]
mod helpers;

#[path = "grpc_roundtrip/auth.rs"]
mod auth;

#[path = "grpc_roundtrip/errors.rs"]
mod errors;

#[path = "grpc_roundtrip/evaluate.rs"]
mod evaluate;
