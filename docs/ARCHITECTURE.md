# openpick — Architecture

> A Jev-compatible, open-source decision-inference engine in Rust.
>
> Wire spec: https://docs.typesafe.ai/api
> Reference client SDK (target): https://docs.typesafe.ai/sdk/python/api

## Overview

`openpick` is a server that takes structured decision questions (Noul,
Choice, Score) and returns inferred answers, plus the surrounding
tracing/metrics/auth surface a public SDK needs. It is wire-compatible
with the proposed `typesafe_sdk` Python client, so any future consumer
can speak to a self-hosted `openpickd` daemon the same way it speaks to
the hosted TypeSafe API.

The repo is split so that the **wire contract** lives in a tiny crate
(`openpick-core`), the **model logic** lives behind a trait
(`openpick-engine`), and the **transport layers** (HTTP, gRPC) live in
`openpick-api`. Anyone can drop in a new model backend (candle, GGUF,
ONNX, a remote provider) by implementing `DecisionEngine`.

## Workspace layout

```
openpick/
├── Cargo.toml                # workspace manifest + shared deps
├── crates/
│   ├── openpick-core/        # Jev wire types (request, response, error)
│   ├── openpick-engine/      # DecisionEngine trait + EngineRegistry
│   ├── openpick-api/         # HTTP (axum) + gRPC (tonic) transport
│   ├── openpick-server/      # openpickd binary
│   ├── openpick-cli/         # openpick binary
│   ├── openpick-runtime/     # hardware / OS abstraction (Phase 2+)
│   ├── openpick-backends/    # candle / GGUF / onnx (Phase 2+)
│   └── openpick-gen-schemas/ # one-shot JSON Schema codegen
├── proto/openpick.proto      # gRPC service definition
├── examples/                 # 8 JSON fixtures from the Jev spec
├── docs/ARCHITECTURE.md      # this file
└── crates/openpick-core/schemas/   # generated JSON Schema docs
```

### Layering

```
┌─────────────────────────────────────────────────────┐
│ openpickd  / openpick cli (binaries)                │
├─────────────────────────────────────────────────────┤
│ openpick-api      (HTTP axum 0.8 + gRPC tonic 0.14) │
│   ├── middleware: request_id, auth, tracing         │
│   ├── error mapping → 400/401/404/422/429/5xx       │
│   └── models: ModelInfo, ModelsResponse (Jev shape) │
├─────────────────────────────────────────────────────┤
│ openpick-engine    (DecisionEngine trait)           │
│   └── MockEngine (Phase 1) — deterministic fake     │
│   └── candle / GGUF / onnx (Phase 2) — TBD          │
├─────────────────────────────────────────────────────┤
│ openpick-core      (wire types only)                │
│   ├── SystemRequest, SystemResponse, Answer         │
│   ├── validate_request() → ValidationError          │
│   └── JSON Schema (jev-v1-request.json, ...)        │
└─────────────────────────────────────────────────────┘
```

Strict dependency direction: `core` ⇐ `engine` ⇐ `api` ⇐ `server/cli`.
None of the upper layers ever reach back. New types land in `core`;
behaviors land in `engine`; transport lives in `api`.

## Crate responsibilities

### `openpick-core`

Jev wire-format types. This crate is the single source of truth for
what a request and response look like on the wire, across both HTTP
and gRPC. Public API:

- `SystemRequest`, `SystemResponse`
- `Question` (Noul, Choice, Score variants), `Answer` (same)
- `State` (string | object | array), `Instructions` (same)
- `ValidationError` + `validate_request(req)`
- `ModelInfo`, `ModelsResponse` (`GET /v1/models` wire shape)

Float widths matter on the wire: `probabilities`, `score`, `noul`,
`confidence` are all `f64`. `0.92f32` round-trips to
`0.9200000166893005` which is a wire-format regression.

### `openpick-engine`

The trait that anything callable from `/v1/systemone` implements:

```rust
#[async_trait]
pub trait DecisionEngine: Send + Sync + 'static {
    fn backend_id(&self) -> &str;
    fn model_metadata(&self) -> openpick_core::ModelInfo;

    async fn evaluate(
        &self,
        req: SystemRequest,
    ) -> Result<SystemResponse, EngineError>;
}
```

`EngineRegistry` maps alias → `Arc<dyn DecisionEngine>`. The
server CLI takes a `--models mock,probe=candle,...` flag and registers
each name in the registry on startup.

Phase 1 ships `MockEngine` — deterministic + slightly jittered — to
exercise the protocol. Phase 2 adds real candle / GGUF / native
backends behind the same trait, targeting the frozen Qwen3.5-4B +
7,683-parameter linear classification head validated in Phase 2B
(87.67% accuracy, ~80.75 ms median decision latency; reference weights
in `research/opendecision_phase2b_20260917T205849Z/frozen_export/`).

### `openpick-api`

Two transports, one business logic:

- **HTTP** (`src/http.rs`) — axum 0.8 router:
  - `POST /v1/systemone`  — the Jev evaluation endpoint
  - `GET /v1/models`     — `{"models":[{name,description,release_date}]}`
  - `GET /health`        — liveness, no auth
  - `GET /metrics`       — Prometheus exporter
- **gRPC** (`src/grpc.rs`) — tonic 0.14, single service `system_one`
  with one `evaluate` RPC. Inherits the request_id header via a
  tonic interceptor.

Middleware (`src/middleware.rs`):

- `request_id_layer` — reads `x-typesafe-request-id` from request if
  present, else mints a UUIDv4 and stamps on response. Runs **outermost**
  so it's stamped even on 401s.
- `auth_layer` — Bearer-token gate on `/v1/*`, env-driven via
  `OPENPICK_API_KEY`. `/health` and `/metrics` are open.
- `TraceLayer` — structured tracing per request.

Error model (`src/error.rs`) maps `ApiError` → `(status, error-envelope, headers)`:

| Status | Variant                 | Headers                        |
|--------|-------------------------|--------------------------------|
| 400    | `BadJson`               | —                              |
| 401    | `Unauthorized`          | `WWW-Authenticate: Bearer`     |
| 404    | `ModelNotFound`         | —                              |
| 422    | `Validation(...)`       | —                              |
| 429    | `RateLimited { retry_after_ms }` | `Retry-After`, `retry-after-ms` |
| 529    | `Overloaded { retry_after_ms }` | `Retry-After`, `retry-after-ms` |
| 5xx    | `Internal`              | —                              |

Every error response includes `x-typesafe-request-id`, including 4xx
and 5xx.

### `openpick-server`

The `openpickd` daemon binary. Entry point for production. Parses CLI
flags via clap, builds the registry, mounts the HTTP router from
`openpick-api::http`, mounts the gRPC service from
`openpick-api::grpc`, listens on both ports concurrently, and shuts
down cleanly on SIGTERM.

### `openpick-cli`

The `openpick` binary. Subcommands:

- `openpick serve`  — same as `openpickd` (left for symmetry)
- `openpick evaluate` — read a JSON request from a file or stdin, hit a
  running daemon, write the response to stdout
- `openpick inspect` — pretty-print a JSON file, validating it against
  the Jev request schema
- `openpick version` — print build metadata

### `openpick-runtime`, `openpick-backends`

Placeholders for Phase 2. `runtime` will own device discovery, VRAM
accounting, and worker pools. `backends` will implement real model
loaders (candle / GGUF / ONNX / remote).

## Wire compatibility

The server matches the Jev HTTP API at https://docs.typesafe.ai/api
**and** the proposed `typesafe_sdk` Python client at
https://docs.typesafe.ai/sdk/python/api. Every endpoint, header,
and error code is covered by `crates/openpick-api/tests/sdk_compat.rs`
(37 tests).

Conformance to the Jev examples (`examples/01..08`) is covered by
`crates/openpick-core/tests/conformance.rs` (23 tests).

## Testing strategy

- **`openpick-core`** — `cargo test` validates every spec example
  round-trips through serde (request → response).
- **`openpick-core`** — `cargo run -p openpick-gen-schemas` regenerates
  `schemas/jev-v1-{request,response}.json` from the Rust types.
- **`openpick-engine`** — `MockEngine` is deterministic with a seeded
  RNG and sorted key iteration.
- **`openpick-api`** — middleware unit tests + `sdk_compat.rs` end-to-end
  tests (axum's `tower::ServiceExt::oneshot` against the real router).
- **`openpick-api`** — `grpc_roundtrip.rs` spins up a real
  `tonic::transport::Server` on a random port and exercises the
  evaluate RPC.
- **`openpick-server`** — manual end-to-end: build, run, curl each
  endpoint with all 8 example fixtures.

Current totals as of last test run: **122 tests passing, 0 failing.**

## Operational notes

- **Auth is opt-in.** Without `OPENPICK_API_KEY` set, `/v1/*` is open
  (useful for local dev). Setting the env var gates the API surface
  with constant-time token comparison.
- **Metrics are optional.** `/metrics` returns 200 with `# not
  installed` until the daemon binary calls
  `install_metrics_recorder()` on startup (it always does).
- **gRPC keeps the same error model.** `R pc_status_to_http()` in
  `grpc.rs` maps `ApiError` → `tonic::Status` codes so a gRPC client
  sees the same taxonomy.
- **Wire types must stay in sync.** Any change to a field in `core`
  is a breaking change for every future SDK caller. Bump the Jev
  version (`JevRequest::SCHEMA_VERSION`) and add a round-trip test
  before merging.

## What's left (Phase 2 & Phase 3)

### Phase 2: Python Model Research (In Progress)
- Multi-seed baseline stabilization and systematic 3-tier batching invariance diagnostic (resolving the ~1.27% padding shift).
- Dynamic candidate scoring for Jev `Choice` ($K \le 255$), `Noul`, and ordinal `Score`.
- Controlled LoRA fine-tuning comparison against the frozen baseline.
- Model scaling (Qwen 2B / 0.8B) and quantization (GGUF / AWQ) to reduce the ~7.85 GiB memory footprint.
- Shared-state prefill and KV-cache branching in Python.

### Phase 3: Rust Engine Implementation (Planned)
- Real model backends: candle for GGUF weights, ONNX runtime, optional
  remote provider passthrough, implementing the validated Qwen architecture.
- Parity test harness: validate Rust engine execution against
  `research/opendecision_phase2b_20260917T205849Z/frozen_export/golden_head_inputs.npz`.
- Request scheduler: batch incoming `system_one` requests and evaluate branched
  questions against a shared model instance and cached state.
- VRAM/device accounting in `openpick-runtime`.
- A Python `openpick` client that issues real RPCs against a
  self-hosted `openpickd` (the SDK compat tests are the contract).
