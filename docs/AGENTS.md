# AGENTS.md

> Briefing for any human or LLM agent working on `openpick`. Read this
> before touching the repo.

## Project goal

Build an **open-source inference engine that speaks the Jev protocol**
— *Jev is a judgment-envelope format* co-developed by TypeSafe for
interfacing with decision-making models. Jev's premise (paraphrased
from [Introducing System One Models and Jev][jev-blog]):

> Most AI today is generative and tries to maximize helpfulness. But
> products increasingly need a different signal: a structured
> **judgment** about the user's state, intent, or preferences, returned
> in a strictly-typed envelope the application can act on. Jev is the
> schema. [TypeSafe's `system_one` models][typesafe-api] are the
> implementation.

`openpick` is the open implementation of that idea. It is wire- and
SDK-compatible with TypeSafe's hosted API at https://api.typesafe.ai so
that:

- A future Python or Rust client built against `typesafe_sdk` / Jev
  can target either the hosted service or a self-hosted `openpickd`
  with **no code change**.
- Operators can run inference on their own hardware, swap in custom
  model backends (candle, GGUF, ONNX, a remote provider), and own the
> data flow end-to-end.

The end state is: drop in `openpickd`, point `typesafe_sdk` at it,
keep the app layer untouched.

## Source of truth

| What                                  | Where                                                  |
|---------------------------------------|--------------------------------------------------------|
| Wire-format spec (HTTP)               | https://docs.typesafe.ai/api                           |
| Wire-format spec (Python SDK surface)  | https://docs.typesafe.ai/sdk/python/api                |
| Schema blog post (goal & framing)     | https://typesafe.ai/blog/introducing-system-one-models-and-jev |
| Architecture diagram & phasing        | [docs/ARCHITECTURE.md](./ARCHITECTURE.md)              |
| Generated JSON Schema (request)       | [crates/openpick-core/schemas/jev-v1-request.json](../crates/openpick-core/schemas/jev-v1-request.json) |
| Generated JSON Schema (response)      | [crates/openpick-core/schemas/jev-v1-response.json](../crates/openpick-core/schemas/jev-v1-response.json) |
| Service `.proto`                       | [proto/proto/openpick.proto](../proto/proto/openpick.proto) |
| Example request/response fixtures     | [examples/](../examples/) (`01_noul.json` ... `08_response_score.json`) |
| SDK compatibility contract tests       | [crates/openpick-api/tests/sdk_compat.rs](../crates/openpick-api/tests/sdk_compat.rs) |

If the docs and the code disagree, **the code wins only after a
round-trip test proves it; otherwise the docs win**. The
`tests/sdk_compat.rs` file is the executable contract — every
endpoint, header, and error code from the docs is pinned there.

## Schema Information & Wire Contracts

`openpick` maintains dual wire schemas that are strictly synchronized:

### 1. Jev JSON Schema (Draft 2020-12)
- **Canonical Schema Files**:
  - Request: [`crates/openpick-core/schemas/jev-v1-request.json`](../crates/openpick-core/schemas/jev-v1-request.json)
  - Response: [`crates/openpick-core/schemas/jev-v1-response.json`](../crates/openpick-core/schemas/jev-v1-response.json)
- **Code Authority**: `openpick-core` (`SystemRequest`, `SystemResponse`, `Question`, `Answer`, `State`, `Usage`).
- **Generation Tool**: `openpick-gen-schemas` (`cargo run -p openpick-gen-schemas -- --write`).
- **Request Format**:
  - `state`: Polymorphic `JSONContent` (plain text string, JSON object, or JSON array).
  - `model`: Target backend model identifier (`"jev-latest"`, `"mock"`, etc.).
  - `questions`: Map of string identifiers to typed questions.
- **Question Kinds** (`"type"` discriminant):
  - `noul`: Boolean probability question. `instructions` (`JSONContent`), optional `criteria` (`{"true": str, "false": str}`).
  - `choice`: Categorical choice question. `instructions` (`JSONContent`), `criteria` (`map<str, str | null>`).
  - `score`: Ordered rubric rating. `instructions` (`JSONContent`), `criteria` (`array<str>`, $\ge 2$ ordered rubric levels).
- **Response Format**:
  - `model`: Echoed evaluation model.
  - `usage`: `UsageInfo` (`input_tokens: uint32`, `output_tokens: uint32`).
  - `answers`: Map of question identifiers to typed answers.
- **Answer Kinds** (`"type"` discriminant):
  - `noul`: `noul: f64` $\in [0.0, 1.0]$. **No `confidence` field per specification**.
  - `choice`: `choice: str`, `probabilities: map<str, f64>` (sums to $1.0$), `confidence: f64` $\in [0.0, 1.0]$.
  - `score`: `score: f64`, `legend: map<str, str>` (indices `"0"`, `"1"`, ...), `probabilities: map<str, f64>`, `confidence: f64` $\in [0.0, 1.0]$.

### 2. Protobuf Schema (`openpick.proto`)
- **Canonical Schema File**: [`proto/proto/openpick.proto`](../proto/proto/openpick.proto)
- **Package**: `openpick`
- **Service**: `SystemOne`
  - `rpc Evaluate (SystemOneRequest) returns (SystemOneResponse)`
- **Wire Parity Rules**:
  - Floating-point fields MUST be `double` (64-bit IEEE 754), matching `f64` in `openpick-core`.
  - `state` uses `oneof value { string text = 1; Structured structured = 2; }` where `Structured.bytes json` forwards raw JSON bytes.
  - `instructions_json` uses raw JSON bytes to preserve `string | object | array` polymorphism.
  - `NoulCriteria` uses `string is_true = 1` and `string is_false = 2` to avoid keyword collision with Protobuf/Rust `true`/`false`.
  - `NoulAnswer` contains only `double noul = 1` (no confidence).
  - Code generation runs at build time via `proto/build.rs` using `tonic-prost-build`.

### 3. OpenAPI 3.1 Specification (`openapi.yaml`)
- **Canonical Schema File**: [`crates/openpick-api/openapi.yaml`](../crates/openpick-api/openapi.yaml) (also referenced at [`docs/openapi.yaml`](./openapi.yaml))
- **Format**: OpenAPI 3.1.0 (YAML), natively aligned with JSON Schema Draft 2020-12.
- **Coverage**:
  - Routes: `POST /v1/systemone` (canonical), `POST /v1/system_one` (SDK alias), `GET /v1/models`, `GET /health`, `GET /metrics`.
  - Security: `BearerAuth` scheme (token gate on `/v1/*`).
  - Headers: `x-typesafe-request-id` (UUIDv4), `Retry-After` (integer seconds), `retry-after-ms` (integer milliseconds), `WWW-Authenticate: Bearer`.
  - Status Codes: `200 OK`, `400 Bad Request` (`bad_json`), `401 Unauthorized` (`unauthorized`), `404 Not Found` (`unknown_model`), `422 Unprocessable Entity` (`invalid_body`), `429 Too Many Requests` (`rate_limited`), `529 Overloaded` (`overloaded`), `500 Internal Server Error` (`internal_error`).

## Phasing

Tracked as git history. Each phase ships with passing tests at HEAD.

### Phase 0 — wire contract (DONE)

Lock the schema before the server. Goal: TypeSafe's spec examples
must round-trip through our types without any data loss.

- [x] `openpick-core` crate — `SystemRequest`/`SystemResponse`/`Answer`
      with `f64` (not `f32`) precision on wire values.
- [x] `validate_request()` covering every documented validation rule.
- [x] 23 conformance tests against every Jev spec example.
- [x] `gen-schemas` binary → JSON Schema files for publishing.
- [x] 8 example JSON fixtures in `examples/`.

### Phase 1 — daemon + SDK compatibility (DONE)

A self-hostable server that the future `typesafe_sdk` Python client
can speak to without modification.

- [x] HTTP (axum 0.8): `POST /v1/systemone` (aliased to `/v1/system_one`),
      `GET /v1/models`, `GET /health`, `GET /metrics`.
- [x] gRPC (tonic 0.14): `openpick.system_one.SystemOne`.
- [x] `openpickd` daemon + `openpick` CLI.
- [x] `DecisionEngine` trait + `MockEngine` (deterministic, slightly
      jittered answers for testing).
- [x] **SDK compatibility surface** (the bit the future client
      actually depends on):
  - `x-typesafe-request-id` on every response, including 401s.
  - Bearer auth opt-in via `OPENPICK_API_KEY` with fallback to `TYPESAFE_API_KEY`.
  - `Retry-After` / `retry-after-ms` on 429 / 529.
  - Error envelope `{"error":{"code",message}}` mapped to the
    Python SDK's `TypeSafe{Authentication,RateLimit,BadRequest,...}Error`.
  - `/v1/models` returns `{"models":[{name,description,release_date}]}`.
- [x] 56 SDK compat tests + 22 HTTP unit/integration tests + 8 gRPC
      roundtrip tests + 33 core tests + 14 engine tests + 7 CLI tests + 2 server tests + 2 proto tests = **144 tests passing** at HEAD.

### Phase 2 — model research and serving gates (Phase 2E MEASURED / DONE)

Phase 2E keeps Qwen3.5, the NLI head, and the real-candidate scorer frozen.
Run `20260918T114914072764Z` completed on an NVIDIA L4 with fresh FP32 and
BF16 workers. The [expanded archive](../research/opendecision_phase2e_expanded_20260918T114914072764Z/)
contains the [results README](../research/opendecision_phase2e_expanded_20260918T114914072764Z/README_results.md).
An independent reconstruction of the saved probability
distributions, policy actions, parity counts, and timing aggregates agreed
with the report. The reconstruction validated saved calculations; it did not
rerun Qwen.

- [x] FP32 full-prompt and shared-prefix strategies: 0/128 tolerance
      failures, selected-outcome changes, or answer/review changes at a
      0.005 probability tolerance. Maximum differences were 0.00000928,
      0.00000776, and 0.00001072.
- [x] Hybrid cache isolation: reusable cache state, repeated branches, and
      candidate-order reversal passed the saved checks, including recurrent
      and convolution state isolation.
- [x] FP32 cached suffix batching: 1.46x faster than full-prompt batch four
      at 16 candidates, and approximately 7x faster on the synthetic
      1,024-token shared-prefix benchmark, while preserving tested policy
      behavior.
- [x] BF16 behavior boundary: 10/128 to 14/128 selected-outcome changes and
      7/128 to 11/128 answer/review changes depending on strategy. BF16 is
      faster and smaller, but is not behavior-preserving for this reference.
- [x] Component profile: 92–94% of FP32 cached request time was model
      execution, versus 4.6–6.6% cache cloning and expansion.
- [ ] Shared-state branching across different questions, Rust/Metal/HTTP,
      concurrent requests, long-document decision quality, and a cheaper
      behavior-preserving precision configuration.

### Phase 2F: Rust reference engine and execution optimization (NEXT)

- [ ] Reproduce the pinned FP32 full-prompt, cached sequential, and cached
      batched paths in Rust before optimizing.
- [ ] Optimize suffix-batch utilization, exact-length grouping, and
      model-forward efficiency while retaining probability, selected-outcome,
      answer/review, order, and branch-isolation gates.
- [ ] Evaluate padded or packed suffix strategies with independent parity
      evidence, then evaluate lower precision as a separate configuration.

### Phase 3 — Rust engine and production backends (PLANNED / GATED ON PHASE 2F)

- [ ] `openpick-runtime` — device discovery, VRAM accounting,
      worker pools, and shared-state cache.
- [ ] `openpick-backends` — candle (GGUF), ONNX runtime, optional
      remote-provider passthrough.
- [ ] Request scheduler — batch incoming `system_one` calls only after
      the numerical reference, rejection-policy, and cache-isolation checks
      pass.
- [ ] Cancel tokens / timeout propagation from gRPC deadline headers.
- [ ] A real Python client (`openpick` or `typesafe_sdk`) that exercises
      a self-hosted `openpickd` end-to-end against a logged-in daemon.
- [ ] Eval harness: feed a labeled dataset through a model, capture
      the side-by-side comparison with the hosted API.

## Conventions for agents

### When you change a wire type

1. Edit the Rust struct in `crates/openpick-core/src/`.
2. Run `cargo test -p openpick-core` — fix any broken round-trip.
3. Regenerate the JSON Schema: `cargo run -p openpick-gen-schemas -- --write`.
4. Add a new fixture in `examples/` mirroring the Jev spec example.
5. Add an `sdk_compat.rs` test pinning the new shape.
6. Bump `JevRequest::SCHEMA_VERSION` (or whatever constants surface
   in `core::request`).
7. Update `docs/ARCHITECTURE.md` if the layering changed.

### When you touch the wire format

Any change to a JSON field in `core::request` or `core::response` is
a breaking change for every future SDK caller. Bisect-friendly
phasing: the next non-trivial release must include a "Jev spec
delta" section in the changelog pointing at the diff between
`jev-v1-{request,response}.json` versions.

### When you add a backend

Implement `openpick_engine::DecisionEngine` for your model loader.
Wire it in `openpick-server/src/main.rs` behind a CLI flag. Add at
least one `cargo test -p openpick-engine` round-trip test. Add a
`crates/openpick-api/tests/sdk_compat.rs` case that exercises the
new alias end-to-end via `/v1/systemone`.

### When you don't know what's right

- Read `docs/ARCHITECTURE.md` first.
- Look at `tests/sdk_compat.rs` second — if your change would break
  one of those tests, you are touching the wire contract and you
  must add a new pinned test alongside the fix.
- The TypeSafe HTTP spec (https://docs.typesafe.ai/api) is
  authoritative for HTTP shapes. The Python SDK doc
  (https://docs.typesafe.ai/sdk/python/api) is authoritative for
  client surface conventions (headers, exceptions, retry policy).

## Quick reference

```bash
# Build everything
cargo build --workspace

# Run all tests (144 at HEAD)
cargo test --workspace

# Regenerate JSON Schema
cargo run -p openpick-gen-schemas -- --write

# Start the daemon (mock engine, dev mode)
cargo run -p openpickd -- --http-addr 127.0.0.1:18080 \
                            --grpc-addr 127.0.0.1:19090 \
                            --models mock,jev-latest

# Verify a request against a running daemon
curl -sS -H 'content-type: application/json' \
     -X POST http://127.0.0.1:18080/v1/systemone \
     -d @examples/04_mixed.json | jq
```

[typesafe-api]: https://docs.typesafe.ai/api
[typesafe-sdk]: https://docs.typesafe.ai/sdk/python/api
[jev-blog]: https://typesafe.ai/blog/introducing-system-one-models-and-jev
