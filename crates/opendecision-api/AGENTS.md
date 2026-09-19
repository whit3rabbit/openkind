# AGENTS.md — opendecision-api

> LLM developer guide for `opendecision-api`. Read this before modifying HTTP routes, gRPC services, middleware, or error handling.

## Crate Purpose & Boundaries

`opendecision-api` is the protocol presentation layer for `opendecision`. It adapts inbound network traffic into `SystemRequest` calls to `opendecision_engine::dispatch`, and formats the returned `SystemResponse` into HTTP and gRPC envelopes.

It defines:
- Axum 0.8 HTTP router and handlers (`systemone`, `list_models`, `health`, `prometheus_metrics`).
- Tonic 0.14 gRPC `SystemOne` service implementation.
- Outermost middleware (`request_id_layer`, `auth_layer`, `TraceLayer`).
- Error mapping into TypeSafe JSON envelopes and HTTP status codes.

### Critical Invariants

1. **Middleware Ordering Matters**:
   In Axum, middleware added *later* wraps *earlier* layers (outermost execution).
   - `request_id_layer` MUST be outermost so that `x-typesafe-request-id` is stamped on **every single response**, including 401s from `auth_layer`, 422s from validation, and 5xx crashes.
   - `auth_layer` gates `/v1/*`, but MUST NEVER gate `/health` or `/metrics`.
2. **SDK Contract (`tests/sdk_compat.rs`)**:
   - `crates/opendecision-api/tests/sdk_compat.rs` is the pinned executable specification matching TypeSafe's Python SDK.
   - If a code change breaks any test in `tests/sdk_compat.rs`, you are introducing a client-facing regression.
3. **HTTP / gRPC Parity**:
   - Both transports call `opendecision_engine::dispatch`.
   - Error categories in gRPC (`Status::invalid_argument`, `Status::not_found`, `Status::internal`) must map symmetrically to HTTP statuses (`422`, `404`, `500`).
   - gRPC responses also stamp metadata key `x-typesafe-request-id`.

## Key Files & Types

- [`openapi.yaml`](./openapi.yaml):
  - Canonical OpenAPI 3.1.0 specification for all HTTP endpoints, request/response headers, status codes, and Jev data schemas.
- [`src/http.rs`](./src/http.rs):
  - `router(registry)` / `router_with_auth(registry, auth)` / `router_with_state(state, auth)`.
  - Routes:
    - `POST /v1/systemone` (aliased to `/v1/system_one`)
    - `GET  /v1/models`
    - `GET  /health`
    - `GET  /metrics`
  - Metrics initialization via `install_metrics_recorder()`.
- [`src/grpc.rs`](./src/grpc.rs):
  - `SystemOneService`: Implements `opendecision::system_one_server::SystemOne`.
  - Converts Protobuf types $\leftrightarrow$ `opendecision_core` types (`pb_state_to_core`, `pb_questions_to_core`, `core_to_pb_response`).
- [`src/middleware.rs`](./src/middleware.rs):
  - `REQUEST_ID_HEADER = "x-typesafe-request-id"`.
  - `request_id_layer`: Checks for inbound client header, falls back to `Uuid::new_v4()`.
  - `auth_layer`: Constant-time bearer token check using `constant_time_eq` (supports `OPENDECISION_API_KEY` and `TYPESAFE_API_KEY`).
- [`src/error.rs`](./src/error.rs):
  - `ApiError` enum and `IntoResponse` implementation:
    - Formats body as `{"error":{"code": ..., "message": ...}}`.
    - Handles `Retry-After` (seconds, via `ms.div_ceil(1000)`) and `retry-after-ms` (milliseconds) headers for rate-limiting (429) and overload (529).

## Verification Commands

```bash
# Run all API tests (unit tests, grpc roundtrip, and sdk compat)
cargo test -p opendecision-api

# Run just the SDK compatibility contract
cargo test -p opendecision-api --test sdk_compat

# Run gRPC integration tests
cargo test -p opendecision-api --test grpc_roundtrip
```
