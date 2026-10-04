# AGENTS.md — openkind-api

> LLM developer guide for `openkind-api`. Read this before modifying HTTP routes, gRPC services, middleware, or error handling.

## Crate Purpose & Boundaries

`openkind-api` is the protocol presentation layer for `openkind`. It adapts inbound network traffic into `SystemRequest` calls to `openkind_engine::dispatch`, and formats the returned `SystemResponse` into HTTP and gRPC envelopes.

It defines:
- Axum HTTP router and handlers (`systemone`, `list_models`, `health`, `prometheus_metrics`).
- Tonic gRPC `SystemOne` service implementation.
- Outermost middleware stack (`request_id_layer`, `auth_layer`, `rate_limit_layer`, `TraceLayer`).
- Error mapping into TypeSafe JSON envelopes and standard HTTP status codes.

### Critical Invariants

1. **Middleware Ordering Matters**:
   In Axum, middleware added *later* wraps *earlier* layers (outermost execution).
   - `request_id_layer` MUST be outermost so that `x-typesafe-request-id` is stamped on **every single response**, including 401s from `auth_layer`, 422s from validation, 429s from rate limiting, and 5xx crashes.
   - `auth_layer` gates `/v1/*`, but MUST NEVER gate `/health` or `/metrics`.
2. **SDK Contract (`tests/sdk_compat.rs`)**:
   - `crates/openkind-api/tests/sdk_compat.rs` is the pinned executable specification matching TypeSafe's Python SDK.
   - If a code change breaks any test in `tests/sdk_compat.rs`, you are introducing a client-facing regression.
3. **HTTP / gRPC Parity**:
   - Both transports call `openkind_engine::dispatch`.
   - Error categories in gRPC (`Status::invalid_argument`, `Status::not_found`, `Status::unavailable`, `Status::internal`) must map symmetrically to HTTP statuses (`422`, `404`, `529`, `500`).
   - gRPC responses also stamp metadata key `x-typesafe-request-id`.

## Key Files & Types

- [`openapi.yaml`](./openapi.yaml):
  - Canonical OpenAPI 3.1.0 specification for all HTTP endpoints, request/response headers, status codes, and Jev data schemas.
  - Linked at `docs/openapi.yaml` and documented for TypeSafe Python SDK compatibility (`https://docs.typesafe.ai/sdk/python/api`).
  - Validated via `npx --yes @redocly/cli@1.34.5 lint docs/openapi.yaml`.
- [`src/http.rs`](./src/http.rs):
  - `router(registry)` / `router_with_auth(registry, auth)` / `router_with_state(state, auth)` / `router_daemon(state, auth, limit, rate_limiter, playground)` / `router_daemon_with_arrow(state, auth, limit, rate_limiter, playground, arrow)`.
  - Routes:
    - `POST /v1/systemone` (aliased to `/v1/system_one`)
    - `GET  /v1/models`
    - `GET  /health`
    - `GET  /metrics`
    - `GET  /playground` — only when the daemon passes `playground = true` (`openkindd --playground on`)
    - `POST /v1/arrow` — only when the daemon passes `arrow = true` (`openkindd --arrow on`)
  - Metrics initialization via `install_metrics_recorder()`.
- [`src/arrow.rs`](./src/arrow.rs) & [`src/arrow_tests.rs`](./src/arrow_tests.rs):
  - Implements the opt-in Arrow IPC endpoint outside the TypeSafe contract. It dispatches once per state and returns a schema-first stream. [`docs/ARROW.md`](../../docs/ARROW.md) owns mapping and limits. Round-trip tests define decoding.
- [`src/playground.rs`](./src/playground.rs) & [`assets/playground.html`](./assets/playground.html):
  - Serves a self-contained HTML asset with `no-store`. Evaluation uses `/v1/systemone` and `/v1/models`. Optional local model controls use gated `/playground/api/models`; mutations require `x-openkind-playground: 1`.
- [`src/http_tests.rs`](./src/http_tests.rs):
  - Unit tests for HTTP routes, handler dispatch, model listing, and error response formatting.
- [`src/models.rs`](./src/models.rs):
  - Model response construction and metadata projection helpers.
- [`src/proxy.rs`](./src/proxy.rs):
  - Intercepts requests before proxy-cache dispatch (see [`docs/PROXY_CACHE.md`](../../docs/PROXY_CACHE.md)). `SystemProxy` can serve cached answers, forward with the caller's bearer key, or replace model listings. `/v1/systemone` alone carries cache headers. The wire schema stays unchanged.
  - `ApiError::BadGateway` (HTTP 502, `bad_gateway`) carries upstream forwarding failures.
- [`src/grpc.rs`](./src/grpc.rs):
  - `SystemOneService`: Implements `openkind::system_one_server::SystemOne`.
  - Converts Protobuf types $\leftrightarrow$ `openkind_core` types (`pb_state_to_core`, `pb_questions_to_core`, `core_to_pb_response`).
- [`src/middleware/`](./src/middleware/):
  - Modular middleware stack:
    - [`src/middleware/request_id.rs`](./src/middleware/request_id.rs): `REQUEST_ID_HEADER = "x-typesafe-request-id"`, `request_id_layer` (checks for inbound client header, falls back to `Uuid::new_v4()`).
    - [`src/middleware/auth.rs`](./src/middleware/auth.rs): `auth_layer` and `auth_layer_with_rate_limit` hash bearer tokens with SHA-256 via `ring` and compare with `subtle::ConstantTimeEq`. Rejected credentials increment `openkind_auth_failures_total` (with a fixed `transport` label) and consume an independent `failed_auth` rate limit budget without spending the evaluation quota.
    - [`src/middleware/rate_limit.rs`](./src/middleware/rate_limit.rs): `rate_limit_layer`, `RateLimiter`, and `RequestLimits` (independent evaluation and failed-authentication budgets emitting 429 status and retry headers).
    - [`src/middleware/tests/`](./src/middleware/tests/): Dedicated test suites (`request_id_tests.rs`, `auth_tests.rs`, `rate_limit_tests.rs`).
- [`src/error.rs`](./src/error.rs):
  - `ApiError` enum and `IntoResponse` implementation:
    - Formats body as `{"error":{"code": ..., "message": ...}}`.
    - Handles `Retry-After` (seconds, via `ms.div_ceil(1000)`) and `retry-after-ms` (milliseconds) headers for rate-limiting (429) and overload (529).
    - Maps `EngineError::BackendValidation` to HTTP 500 (`internal_error`) / gRPC `Code::Internal`.
- [`tests/sdk_compat.rs`](./tests/sdk_compat.rs) & [`tests/sdk_compat/`](./tests/sdk_compat/):
  - Executable compatibility contract with the TypeSafe Python SDK modularized into:
    - [`helpers.rs`](./tests/sdk_compat/helpers.rs): Test server routing and HTTP helper functions.
    - [`system_one.rs`](./tests/sdk_compat/system_one.rs): `/v1/systemone` and `/v1/system_one` contracts.
    - [`models.rs`](./tests/sdk_compat/models.rs): `/v1/models` listing and sort invariants.
    - [`errors.rs`](./tests/sdk_compat/errors.rs): HTTP 400/401/404/422/529 error envelopes and codes.
    - [`headers.rs`](./tests/sdk_compat/headers.rs): `x-typesafe-request-id` UUID generation and preservation across endpoints.
    - [`auth.rs`](./tests/sdk_compat/auth.rs): Bearer token protection on `/v1/*` while keeping `/health` and `/metrics` public.
    - [`flows.rs`](./tests/sdk_compat/flows.rs): End-to-end Python SDK usage patterns.
    - [`contract_client.rs`](./tests/sdk_compat/contract_client.rs): Per-call headers, metadata merging, model overrides.
    - [`contract_types.rs`](./tests/sdk_compat/contract_types.rs): Jev questions and typed answer deserialization.
    - [`contract_errors.rs`](./tests/sdk_compat/contract_errors.rs): Retry header emission and error taxonomy.
    - [`openapi.rs`](./tests/sdk_compat/openapi.rs): OpenAPI specification validation ensuring all paths, methods, and schemas match the live router.
- [`tests/grpc_roundtrip.rs`](./tests/grpc_roundtrip.rs) & [`tests/grpc_roundtrip/`](./tests/grpc_roundtrip/):
  - End-to-end gRPC protocol tests modularized into:
    - [`helpers.rs`](./tests/grpc_roundtrip/helpers.rs): Ephemeral gRPC server and Protobuf fixtures.
    - [`evaluate.rs`](./tests/grpc_roundtrip/evaluate.rs): All-question-type evaluation and state roundtripping.
    - [`errors.rs`](./tests/grpc_roundtrip/errors.rs): Malformed requests, empty questions, unknown models, and request ID metadata.
    - [`auth.rs`](./tests/grpc_roundtrip/auth.rs): gRPC metadata bearer token validation, `x-api-key`, and request ID sanitization.
- [`tests/transport_limits.rs`](./tests/transport_limits.rs):
  - End-to-end integration tests confirming HTTP and gRPC listeners share evaluation budgets while failed authentication uses an independent budget.

## Gotchas & Wire Subtleties

1. **Dual SystemOne Endpoints**:
   Both `/v1/systemone` and `/v1/system_one` must be routed to the same handler for SDK compatibility.
2. **Overload Header Symmetry (529)**:
   When `EngineError::Overloaded` occurs, the API layer emits status 529 and both `Retry-After` (seconds) and `retry-after-ms` (milliseconds) headers in HTTP, or metadata keys `retry-after` and `retry-after-ms` in gRPC.
3. **Public Endpoints**:
   Keep `/health` and `/metrics` outside `auth_layer` for probes and scrapers. `/playground` exposes only inert HTML. Evaluation and model-control routes remain gated.
4. **Playground Is Not in `openapi.yaml`** (deliberate):
   Keep `GET /playground` out of `openapi.yaml` and SDK parity assertions. Wire API changes belong in the spec; playground UI changes do not.
5. **Unofficial Arrow**:
   Keep `POST /v1/arrow` opt-in and outside OpenAPI and SDK parity tests. It uses `/v1/*` middleware and returns JSON errors, never partial streams. See [`docs/ARROW.md`](../../docs/ARROW.md) and `src/arrow.rs` for its contract.

## Verification Commands

```bash
# Validate OpenAPI specification syntax and semantic rules
npx --yes @redocly/cli@1.34.5 lint docs/openapi.yaml

# Check embedded playground request and lifecycle UI regressions (Node 18+)
node --test crates/openkind-api/tests/playground.test.cjs

# Run all API tests (unit tests, grpc roundtrip, and sdk compat)
cargo test -p openkind-api

# Run just the unofficial Arrow endpoint tests
cargo test -p openkind-api --lib arrow

# Run just the SDK compatibility contract
cargo test -p openkind-api --test sdk_compat

# Run gRPC integration tests
cargo test -p openkind-api --test grpc_roundtrip
```
