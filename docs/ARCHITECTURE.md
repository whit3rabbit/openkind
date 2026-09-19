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

`EngineRegistry` maps alias → `Arc<dyn DecisionEngine>`. The server CLI takes a `--models mock,qwen-bridge,probe=candle,...` flag and registers each name in the registry on startup.

Behind this public boundary, internal execution is governed by four explicit contracts:
1. **Decision Specification**: Encapsulates state, question semantics, candidate criteria, missing-option semantics, and truncation rules.
2. **Model / Execution Profile**: Identifies checkpoint, adapter, tokenizer/rendering conventions, feature normalization, heads, precision/arithmetic mode, and supported execution strategies.
3. **Backend Capabilities**: Declares supported primitives (Noul, Choice, Score), tested candidate/context boundaries, cache operations, and device memory requirements.
4. **Evaluation Context**: Carries tenant identity, admission budget, deadline/cancellation token, and tracing identifiers.

While `evaluate()` is an async trait method, internal execution contracts define how underlying GPU/model work is queued, resource-bounded, and cancelled under load.

### Prototype Bridge Architecture (Phase 2H)

To expose architectural, API, and serving gaps before completing a full native Rust inference rewrite, OpenDecision implements an end-to-end prototype bridge:

```text
HTTP Client (typesafe_sdk / curl)
  │
  ▼
opendecisiond (Rust Axum HTTP Service)
  │  - x-typesafe-request-id tracing
  │  - Bearer token authentication
  │  - Rate limiting & deadline propagation
  │  - Jev schema validation (validate_request)
  ▼
BridgeEngine (implements DecisionEngine)
  │  - Request serialization & token estimation
  ▼ (resident IPC / local HTTP)
Persistent Python Reference Worker
  │  - Resident Qwen backbone (Qwen3.5-4B / 2B)
  │  - Feature-conditioned rejection head
  │  - Isolated hybrid state cache
  ▼
Real Decision Probabilities & Token Accounting
  │
  ▼
Jev-Compliant SystemResponse (validated & stamped)
```

This bridge allows immediate validation of live SDK requests, unsupported input rejection, token accounting, deadline cancellation, and response semantics against real Qwen predictions.

### `opendecision-api`

Two transports, one business logic:

- **HTTP** (`src/http.rs`) — axum 0.8 router:
  - `POST /v1/systemone`  — canonical Jev evaluation endpoint
  - `POST /v1/system_one` — SDK compatibility alias
  - `GET /v1/models`     — `{"models":[{name,description,release_date}]}`
  - `GET /health`        — liveness, no auth
  - `GET /metrics`       — Prometheus exporter
- **gRPC** (`src/grpc.rs`) — tonic 0.14, single service `system_one` with one `evaluate` RPC. Inherits request ID and deadline cancellation.

Middleware (`src/middleware.rs`):

- `request_id_layer` — reads `x-typesafe-request-id` from request if present, else mints UUIDv4 and stamps on response. Runs **outermost** so it is stamped even on 401s.
- `auth_layer` — Bearer-token gate on `/v1/*`, env-driven via `OPENDECISION_API_KEY` (with fallback to `TYPESAFE_API_KEY`). `/health` and `/metrics` are open.
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

Every error response includes `x-typesafe-request-id`, including 4xx and 5xx.

### `opendecision-server`

The `opendecisiond` daemon binary. Entry point for production. Parses CLI flags via clap, builds the registry, mounts the HTTP router from `opendecision-api::http`, mounts the gRPC service from `opendecision-api::grpc`, listens on both ports concurrently, and shuts down cleanly on SIGTERM.

### `opendecision-cli`

The `opendecision` binary. Subcommands:

- `opendecision serve`  — runs the daemon
- `opendecision evaluate` — read a JSON request from a file or stdin, hit a running daemon, write response to stdout
- `opendecision inspect` — pretty-print a JSON file, validating it against the Jev request schema
- `opendecision version` — print build metadata

### `opendecision-runtime`, `opendecision-backends`

Production native components for Phase 3. `runtime` manages device discovery, VRAM allocation, worker pools, queue admission, and persistent LRU caches. `backends` implements native drivers (Candle, GGUF/llama.cpp, ONNX) following the Parity Ladder.

---

## Nested Hybrid Cache Architecture (Phase 2I)

Jev models evaluate multiple typed questions independently against a single state. OpenDecision organizes prefix computation into a **two-level nested cache tree**:

```text
[State Prefix] ─── (prefill root once)
       │
       ├──► [Question 1 Suffix] ─── (isolated question state)
       │           │
       │           ├──► [Candidate 1.1 Suffix] ──► Score
       │           └──► [Candidate 1.2 Suffix] ──► Score
       │
       └──► [Question 2 Suffix] ─── (isolated question state)
                   │
                   ├──► [Candidate 2.1 Suffix] ──► Score
                   └──► [Candidate 2.2 Suffix] ──► Score
```

### Hybrid Isolation Invariant
Because Qwen alternates recurrent DeltaNet blocks and attention blocks, **branch isolation must clone recurrent state and convolution state at every branch point, not merely apply an attention mask**. Recurrent streams cannot be concatenated or shared without mutating state across independent questions or candidates.

---

## Wire & API Contract Resolutions

1. **Rejection & `none` Probability**:
   - The TypeSafe wire contract returns probabilities over caller-supplied criteria and recommends callers add an "other" option when needed.
   - OpenDecision does not silently append an unrequested `none` choice or renormalize probabilities without explicit contract. Internally, rejection probability ($P(\text{none}) = 1 - a$) is evaluated via a feature-conditioned applicability head and surfaced either through caller-supplied `none` criteria, a versioned native extension, or an application review trigger.
2. **Candidate Identifiers & Semantic Descriptions**:
   - Machine IDs are kept separate from candidate semantics. Both candidate names and descriptions reach the model; null descriptions are permitted because option names provide semantic meaning.

---

## Testing strategy

- **`opendecision-core`** — `cargo test` validates every spec example round-trips through serde (request → response).
- **`opendecision-core`** — `cargo run -p opendecision-gen-schemas -- --write` regenerates `schemas/jev-v1-{request,response}.json`.
- **`opendecision-engine`** — `MockEngine` is deterministic with a seeded RNG and sorted key iteration.
- **`opendecision-api`** — middleware unit tests + `sdk_compat.rs` end-to-end tests (56 tests covering headers, errors, auth, and wire compatibility).
- **`opendecision-api`** — `grpc_roundtrip.rs` spins up a real `tonic::transport::Server` on a random port and exercises the evaluate RPC.
- **`opendecision-server`** — end-to-end CLI parsing and daemon launch tests.

Current totals at commit HEAD: **195 tests passing, 0 failing.**

---

## Operational notes

- **Auth is opt-in.** Without `OPENDECISION_API_KEY` set, `/v1/*` is open for local development. Setting the env var gates the API surface with constant-time token comparison.
- **Metrics are optional.** `/metrics` returns 200 with `# not installed` until the daemon binary calls `install_metrics_recorder()` on startup.
- **gRPC keeps the same error model.** `rpc_status_to_http()` in `grpc.rs` maps `ApiError` → `tonic::Status` codes so a gRPC client sees the same taxonomy.
- **Wire types must stay in sync.** Any change to a field in `core` is a breaking change for every future SDK caller. Bump `JevRequest::SCHEMA_VERSION` and add a round-trip test before merging.
- **Deployment Hardening**: In production, explicit opt-in is required for non-loopback network binding, metrics exposure, bounded request payload limits, and sensitive input redaction.

---

## Phasing & What's Left

- **Phase 2H (NEXT)**: Contract hardening, criteria review, feature-conditioned rejection head ($P(\text{none}) = 1-a$), and resident Python reference worker bridge.
- **Phase 2I (PLANNED)**: Genuine multi-question execution ($1\text{ state} \to Q\text{ questions} \to K\text{ candidates}$), state-first prompt rendering, and nested hybrid cache branching.
- **Phase 2J (PLANNED)**: Matched adaptation comparison (frozen heads vs. LoRA vs. `Qwen/Qwen3.5-2B-Base`), multi-task supervision, and honest calibration (NLL / cumulative Brier for Score).
- **Phase 3 (PLANNED / GATED ON 2H/2I/2J)**: Production Rust engine implementing the Parity Ladder, native backends (Candle, GGUF/llama.cpp, ONNX), and deployment hardening.
