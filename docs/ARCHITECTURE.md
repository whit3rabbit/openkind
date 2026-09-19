# opendecision — Architecture

> A Jev-compatible, open-source decision-inference engine in Rust.
>
> Wire spec: https://docs.typesafe.ai/api
> Reference client SDK (target): https://docs.typesafe.ai/sdk/python/api

## Overview

`opendecision` is a server that takes structured decision questions (Noul,
Choice, Score) and returns inferred answers, plus the surrounding
tracing/metrics/auth surface a public SDK needs. It is wire-compatible
with the proposed `typesafe_sdk` Python client, so any future consumer
can speak to a self-hosted `opendecisiond` daemon the same way it speaks to
the hosted TypeSafe API.

The repo is split so that the **wire contract** lives in a tiny crate
(`opendecision-core`), the **model logic** lives behind a trait
(`opendecision-engine`), and the **transport layers** (HTTP, gRPC) live in
`opendecision-api`. Anyone can drop in a new model backend (candle, GGUF,
ONNX, a remote provider) by implementing `DecisionEngine`.

## Workspace layout

```
opendecision/
├── Cargo.toml                # workspace manifest + shared deps
├── crates/
│   ├── opendecision-core/        # Jev wire types (request, response, error)
│   ├── opendecision-engine/      # DecisionEngine trait + EngineRegistry
│   ├── opendecision-api/         # HTTP (axum) + gRPC (tonic) transport
│   ├── opendecision-server/      # opendecisiond binary
│   ├── opendecision-cli/         # opendecision binary
│   ├── opendecision-runtime/     # hardware / OS abstraction (Phase 2+)
│   ├── opendecision-backends/    # candle / GGUF / onnx (Phase 2+)
│   └── opendecision-gen-schemas/ # one-shot JSON Schema codegen
├── proto/opendecision.proto      # gRPC service definition
├── examples/                 # 8 JSON fixtures from the Jev spec
├── docs/ARCHITECTURE.md      # this file
└── crates/opendecision-core/schemas/   # generated JSON Schema docs
```

### Layering

```
┌─────────────────────────────────────────────────────┐
│ opendecisiond  / opendecision cli (binaries)                │
├─────────────────────────────────────────────────────┤
│ opendecision-api      (HTTP axum 0.8 + gRPC tonic 0.14) │
│   ├── middleware: request_id, auth, tracing         │
│   ├── error mapping → 400/401/404/422/429/5xx       │
│   └── models: ModelInfo, ModelsResponse (Jev shape) │
├─────────────────────────────────────────────────────┤
│ opendecision-engine    (DecisionEngine trait)           │
│   └── MockEngine (Phase 1) — deterministic fake     │
│   └── validated native backends (Phase 3) — gated    │
├─────────────────────────────────────────────────────┤
│ opendecision-core      (wire types only)                │
│   ├── SystemRequest, SystemResponse, Answer         │
│   ├── validate_request() → ValidationError          │
│   └── JSON Schema (jev-v1-request.json, ...)        │
└─────────────────────────────────────────────────────┘
```

Strict dependency direction: `core` ⇐ `engine` ⇐ `api` ⇐ `server/cli`.
None of the upper layers ever reach back. New types land in `core`;
behaviors land in `engine`; transport lives in `api`.

## Crate responsibilities

### `opendecision-core`

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

### `opendecision-engine`

The trait that anything callable from `/v1/systemone` implements:

```rust
#[async_trait]
pub trait DecisionEngine: Send + Sync + 'static {
    fn backend_id(&self) -> &str;
    fn model_metadata(&self) -> opendecision_core::ModelInfo;

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
exercise the protocol. Phase 2 research now targets the frozen
Qwen3.5-4B + 7,683-parameter last-token linear head. Phase 2C
replicated that direction at 87.00% matched and 88.80% mismatched
accuracy on fresh 1,000-example MultiNLI partitions, but also found
batch-dependent probability and class changes. Real candle / GGUF /
native backends remain a Phase 3 implementation target behind this
same trait, gated on the Phase 2F reference-engine checks.

### `opendecision-api`

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
  `OPENDECISION_API_KEY`. `/health` and `/metrics` are open.
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

### `opendecision-server`

The `opendecisiond` daemon binary. Entry point for production. Parses CLI
flags via clap, builds the registry, mounts the HTTP router from
`opendecision-api::http`, mounts the gRPC service from
`opendecision-api::grpc`, listens on both ports concurrently, and shuts
down cleanly on SIGTERM.

### `opendecision-cli`

The `opendecision` binary. Subcommands:

- `opendecision serve`  — same as `opendecisiond` (left for symmetry)
- `opendecision evaluate` — read a JSON request from a file or stdin, hit a
  running daemon, write the response to stdout
- `opendecision inspect` — pretty-print a JSON file, validating it against
  the Jev request schema
- `opendecision version` — print build metadata

### `opendecision-runtime`, `opendecision-backends`

Placeholders for Phase 3. `runtime` will own device discovery, VRAM
accounting, worker pools, and the shared-state cache. `backends` will
implement real model loaders (candle / GGUF / ONNX / remote) after the
Python reference path and serving acceptance tests are complete.

### Phase 2 research boundary and serving implications

Phase 2E run `20260918T114914072764Z` completed on an NVIDIA L4. The
[expanded result archive](../research/opendecision_phase2e_expanded_20260918T114914072764Z/)
contains the [results README](../research/opendecision_phase2e_expanded_20260918T114914072764Z/README_results.md),
frozen checkpoint and heads, saved raw parity
rows, complete-request timings, component profiles, and process-isolated
memory snapshots. An independent reconstruction of 3,072 probability
distributions and the reported aggregates agreed with the saved results.

The run establishes the following implementation boundaries:

- **FP32 execution reference:** full-prompt batch-four, shared-prefix
  sequential suffixes, and shared-prefix equal-length suffix batches all
  had 0/128 tolerance failures, selected-outcome changes, and answer/review
  changes. Their largest absolute differences were 0.00000928, 0.00000776,
  and 0.00001072 at a 0.005 probability tolerance.
- **Hybrid cache contract:** reusable cache state must be isolated across
  branches, including recurrent and convolution state, not only attention
  keys and values. Repeated branches and reversed candidate order passed the
  saved isolation checks.
- **BF16 is a separate behavior configuration:** BF16 exceeded tolerance in
  78/128 to 80/128 episodes, changed selected outcomes in 10/128 to 14/128,
  and changed answer/review decisions in 7/128 to 11/128. It remains faster
  and smaller, but is not a drop-in replacement for the FP32 reference.
- **Scheduler implication:** for short real requests, full-prompt batching
  was faster at two and four candidates, while cached suffix batching was
  faster at eight and sixteen. At sixteen candidates, cached suffix batching
  was 1.46x faster than full-prompt batch four. Scheduler selection must
  measure prefix length, candidate count, and suffix-length distribution.
- **Long-prefix result:** with 1,024 shared-prefix tokens and eight synthetic
  candidates, FP32 cached suffix batching was approximately 7x faster than
  either full-prompt strategy, with approximately 0.00000185 maximum
  probability difference and no policy-output change.
- **Optimization priority:** model execution consumed 92–94% of FP32 cached
  request time, while cache cloning and expansion consumed 4.6–6.6%. The
  first optimization target is therefore model-call count and suffix-batch
  utilization, not an elaborate cache allocator.

The result supports a Rust reference engine, not production equivalence for
all inputs or task-quality superiority over BF16. It did not benchmark Rust,
Metal, HTTP, concurrent requests, or long-document decision quality. The
Phase 2F work must reproduce the FP32 full-prompt and cached paths before
production runtime and backend integration begins.

## Wire compatibility

The server matches the Jev HTTP API at https://docs.typesafe.ai/api
**and** the proposed `typesafe_sdk` Python client at
https://docs.typesafe.ai/sdk/python/api. Every endpoint, header,
and error code is covered by `crates/opendecision-api/tests/sdk_compat.rs`
(37 tests).

Conformance to the Jev examples (`examples/01..08`) is covered by
`crates/opendecision-core/tests/conformance.rs` (23 tests).

## Testing strategy

- **`opendecision-core`** — `cargo test` validates every spec example
  round-trips through serde (request → response).
- **`opendecision-core`** — `cargo run -p opendecision-gen-schemas` regenerates
  `schemas/jev-v1-{request,response}.json` from the Rust types.
- **`opendecision-engine`** — `MockEngine` is deterministic with a seeded
  RNG and sorted key iteration.
- **`opendecision-api`** — middleware unit tests + `sdk_compat.rs` end-to-end
  tests (axum's `tower::ServiceExt::oneshot` against the real router).
- **`opendecision-api`** — `grpc_roundtrip.rs` spins up a real
  `tonic::transport::Server` on a random port and exercises the
  evaluate RPC.
- **`opendecision-server`** — manual end-to-end: build, run, curl each
  endpoint with all 8 example fixtures.

Current totals as of last test run: **122 tests passing, 0 failing.**

## Operational notes

- **Auth is opt-in.** Without `OPENDECISION_API_KEY` set, `/v1/*` is open
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

### Phase 2: Python Model Research (Phase 2E measured, Phase 2F next)
- Preserve the pinned FP32 full-prompt outputs, frozen heads, policies, exact
  token sequences, and isolated hybrid cache checks as the Rust reference.
- Reproduce full-prompt, cached sequential, and cached batched execution in
  Rust before optimizing. Keep probability, selected-outcome, answer/review,
  candidate-order, and branch-isolation checks attached to each change.
- Optimize suffix-batch utilization, exact-length grouping, and model-forward
  efficiency. Any padded or packed suffix strategy requires its own parity
  evidence.
- Evaluate lower precision as a separate behavior configuration. Matching
  only the top candidate is not sufficient.
- Complete the untested LoRA, model-scaling, quantization, ordinal `Score`,
  shared-state prefill across different questions, and Metal experiments.

### Phase 3: Rust Engine Implementation (Planned, gated on Phase 2F)
- Real model backends: candle for GGUF weights, ONNX runtime, optional
  remote provider passthrough, implementing the validated Qwen architecture.
- Parity test harness: validate Rust engine execution against
  the Phase 2C exported fixtures and the prior NLI golden vectors.
- Request scheduler: use explicit single-example reference checks, then
  batch incoming `system_one` requests with small, length-aware policies
  and evaluate branched questions against a shared model instance only
  after cache isolation is validated.
- VRAM/device accounting in `opendecision-runtime`.
- A Python `opendecision` client that issues real RPCs against a
  self-hosted `opendecisiond` (the SDK compat tests are the contract).
