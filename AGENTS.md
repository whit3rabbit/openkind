# AGENTS.md

> LLM agent navigation briefing and master index for the `opendecision` workspace.
> Every agent working on this codebase should read this document first before inspecting or modifying code.

---

## 🧭 System Orientation & Project Vision

`opendecision` is an **open-source, high-throughput inference engine that speaks the Jev protocol** — the judgment-envelope protocol co-developed by TypeSafe for interfacing with decision-making / System 1 models.

- **Wire & SDK Compatible**: Wire-compatible with TypeSafe's hosted API (`https://api.typesafe.ai`), allowing clients built with `typesafe_sdk` to target either the hosted service or a local `opendecisiond` daemon without code changes.
- **Dual Transports**: Native HTTP/REST (`/v1/systemone`, aliased to `/v1/system_one`) and gRPC (`opendecision.SystemOne/Evaluate`).
- **Extensible Inference**: Multi-backend runtime supporting mock engines, remote fallback providers, GGUF/llama.cpp, ONNX, and Candle.

---

## 📚 Core Documentation Index

Before diving into implementations, consult the relevant repository-level guides:

- [docs/ROADMAP.md](docs/ROADMAP.md) — Implementation roadmap, phasing checklist (Phases 0 through 4), and milestone progress.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Architectural design, layered crate topology, transport protocol comparison, and data flow.
- [docs/RESEARCH.md](docs/RESEARCH.md) — Research dossier, model benchmarks, zero-copy evaluation, and backend alternatives.
- [docs/AGENTS.md](docs/AGENTS.md) — Project origin, wire contract rules, and architectural invariant checklists.
- [README.md](README.md) — Workspace overview, quick start, installation, and CLI usage.

---

## 📦 Workspace Crate Catalog & Agent Briefings

Each crate in `opendecision` maintains its own dedicated `AGENTS.md` specifying crate boundaries, invariant checklists, testing mandates, and modification patterns.

| Crate | Layer | Purpose | Agent Briefing | Crate README |
|---|---|---|---|---|
| [`opendecision-core`](crates/opendecision-core) | Protocol & Types | Zero-dependency Jev data types, strict serde serialization, validation, and wire errors | [crates/opendecision-core/AGENTS.md](crates/opendecision-core/AGENTS.md) | [crates/opendecision-core/README.md](crates/opendecision-core/README.md) |
| [`opendecision-engine`](crates/opendecision-engine) | Execution Engine | Core `Engine` trait, mock decision engine, calibrated scoring, and execution dispatch | [crates/opendecision-engine/AGENTS.md](crates/opendecision-engine/AGENTS.md) | [crates/opendecision-engine/README.md](crates/opendecision-engine/README.md) |
| [`opendecision-api`](crates/opendecision-api) | Transports & Routing | HTTP router (Axum), gRPC service, auth layer, rate limiting, and SDK compatibility suite | [crates/opendecision-api/AGENTS.md](crates/opendecision-api/AGENTS.md) | [crates/opendecision-api/README.md](crates/opendecision-api/README.md) |
| [`opendecision-server`](crates/opendecision-server) | Daemon Binary | `opendecisiond` server binary, configuration parsing (`OpenDecisionConfig`), and dual HTTP/gRPC lifecycle | [crates/opendecision-server/AGENTS.md](crates/opendecision-server/AGENTS.md) | [crates/opendecision-server/README.md](crates/opendecision-server/README.md) |
| [`opendecision-cli`](crates/opendecision-cli) | Operator Tooling | `opendecision` command-line utility (`eval`, `serve`, `validate`, `bench`, `routes`) | [crates/opendecision-cli/AGENTS.md](crates/opendecision-cli/AGENTS.md) | [crates/opendecision-cli/README.md](crates/opendecision-cli/README.md) |
| [`opendecision-runtime`](crates/opendecision-runtime) | Engine Runtime | Runtime abstraction, engine factory, dynamic engine registry, and backend dispatch | [crates/opendecision-runtime/AGENTS.md](crates/opendecision-runtime/AGENTS.md) | [crates/opendecision-runtime/README.md](crates/opendecision-runtime/README.md) |
| [`opendecision-backends`](crates/opendecision-backends) | Backend Drivers | Driver implementations for remote providers, GGUF/llama.cpp, ONNX, and Candle | [crates/opendecision-backends/AGENTS.md](crates/opendecision-backends/AGENTS.md) | [crates/opendecision-backends/README.md](crates/opendecision-backends/README.md) |
| [`opendecision-gen-schemas`](crates/opendecision-gen-schemas) | Schema Generator | Tooling binary for generating JSON Schemas (`jev-v1-request.json`, `jev-v1-response.json`) from Rust structs | [crates/opendecision-gen-schemas/AGENTS.md](crates/opendecision-gen-schemas/AGENTS.md) | [crates/opendecision-gen-schemas/README.md](crates/opendecision-gen-schemas/README.md) |
| [`proto`](proto) | Protobuf & gRPC Definitions | `opendecision.proto` definition, `tonic-prost-build` code generation, and binary wire serialization tests | [proto/AGENTS.md](proto/AGENTS.md) | [proto/README.md](proto/README.md) |

---

## 📋 Schema Architecture & Wire Contracts

`opendecision` maintains dual wire schemas that are strictly synchronized:

### 1. Jev JSON Schema (Draft 2020-12)
- **Canonical Schema Files**:
  - Request: [`crates/opendecision-core/schemas/jev-v1-request.json`](crates/opendecision-core/schemas/jev-v1-request.json)
  - Response: [`crates/opendecision-core/schemas/jev-v1-response.json`](crates/opendecision-core/schemas/jev-v1-response.json)
- **Code Authority**: [`crates/opendecision-core`](crates/opendecision-core) (`SystemRequest`, `SystemResponse`, `Question`, `Answer`).
- **Generation Tool**: [`crates/opendecision-gen-schemas`](crates/opendecision-gen-schemas) (`cargo run -p opendecision-gen-schemas -- --write`).
- **Data Model**:
  - `SystemRequest`:
    - `state`: Polymorphic `JSONContent` — plain string, JSON object, or JSON array.
    - `model`: String identifier of target backend (`"jev-latest"`, `"mock"`, etc.).
    - `questions`: Map of string identifiers to typed questions.
  - Question Types (`"type"` discriminant):
    - `noul`: Instructions (`JSONContent`), optional `criteria` (`{ "true": str, "false": str }`).
    - `choice`: Instructions (`JSONContent`), `criteria` (`map<str, str | null>`).
    - `score`: Instructions (`JSONContent`), `criteria` (`array<str>`, $\ge 2$ ordered rubric levels).
  - `SystemResponse`:
    - `model`: Echoed evaluation model string.
    - `usage`: Token usage object (`input_tokens: uint32`, `output_tokens: uint32`).
    - `answers`: Map of question identifiers to typed answers.
  - Answer Types (`"type"` discriminant):
    - `noul`: `noul: f64` $\in [0.0, 1.0]$. **No `confidence` field per specification**.
    - `choice`: `choice: str`, `probabilities: map<str, f64>` (sums to $1.0$), `confidence: f64` $\in [0.0, 1.0]$.
    - `score`: `score: f64`, `legend: map<str, str>` (indices `"0"`, `"1"`, ...), `probabilities: map<str, f64>`, `confidence: f64` $\in [0.0, 1.0]$.

### 2. Protobuf Schema (`opendecision.proto`)
- **Canonical Schema File**: [`proto/proto/opendecision.proto`](proto/proto/opendecision.proto)
- **Package**: `opendecision`
- **Service**: `SystemOne`
  - `rpc Evaluate(SystemOneRequest) returns (SystemOneResponse)`
- **Wire Parity Invariants**:
  - Floating-point fields MUST be `double` (64-bit IEEE 754), matching `f64` in `opendecision-core`.
  - `state` is represented via `oneof value { string text = 1; Structured structured = 2; }` where `Structured.bytes json` carries UTF-8 JSON.
  - `instructions_json` in questions carries UTF-8 JSON bytes to preserve `string | object | array` polymorphism.
  - `NoulCriteria` uses `string is_true = 1` and `string is_false = 2` to avoid reserved keyword collision.
  - `NoulAnswer` contains only `double noul = 1` (no confidence).
  - Code generation runs at build time via [`proto/build.rs`](proto/build.rs) using `tonic-prost-build`.

### 3. OpenAPI 3.1 Specification (`openapi.yaml`)
- **Canonical Schema File**: [`crates/opendecision-api/openapi.yaml`](crates/opendecision-api/openapi.yaml) (also referenced at [`docs/openapi.yaml`](docs/openapi.yaml))
- **Format**: OpenAPI 3.1.0 (YAML), natively aligned with JSON Schema Draft 2020-12.
- **Coverage**:
  - Routes: `POST /v1/systemone` (canonical), `POST /v1/system_one` (SDK alias), `GET /v1/models`, `GET /health`, `GET /metrics`.
  - Security: `BearerAuth` scheme (token gate on `/v1/*`).
  - Headers: `x-typesafe-request-id` (UUIDv4), `Retry-After` (integer seconds), `retry-after-ms` (integer milliseconds), `WWW-Authenticate: Bearer`.
  - Status Codes: `200 OK`, `400 Bad Request` (`bad_json`), `401 Unauthorized` (`unauthorized`), `404 Not Found` (`unknown_model`), `422 Unprocessable Entity` (`invalid_body`), `429 Too Many Requests` (`rate_limited`), `529 Overloaded` (`overloaded`), `500 Internal Server Error` (`internal_error`).

---


## 🛠️ Key Architectural Constraints & Rules

1. **Path Discipline**:
   - **Never include full machine paths or absolute paths** (e.g. `/Users/...` or `file:///...`) in documentation, code comments, or links.
   - **Always use relative paths** (e.g., `crates/opendecision-core`, `docs/ARCHITECTURE.md`, `../opendecision-engine`). This ensures documentation is portable across development environments, CI runners, and team checkouts.
2. **Wire Format Stability**:
   - `opendecision-core` types MUST preserve 100% compatibility with TypeSafe's Jev JSON schemas and wire format.
   - All floating-point fields must use `f64`.
   - Never remove or re-order enum variants or struct fields without verifying schema conformance (`cargo run -p opendecision-gen-schemas`).
3. **Dependency Flow**:
   - `core` depends on nothing in workspace.
   - `engine` depends only on `core`.
   - `api` depends on `core`, `engine`, and optionally `proto`.
   - `server` and `cli` compose `api`, `runtime`, and `backends`.
   - Circular dependencies are forbidden.
4. **Verification Before Merging**:
   - Always run the full verification battery before submitting changes:
     ```bash
     cargo fmt --check
     cargo clippy --workspace --all-targets -- -D warnings
     cargo test --workspace
     cargo run -p opendecision-gen-schemas -- --write
     ```

