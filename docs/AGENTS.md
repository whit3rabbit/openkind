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

- [x] HTTP (axum 0.8): `POST /v1/systemone`, `GET /v1/models`,
      `GET /health`, `GET /metrics`.
- [x] gRPC (tonic 0.14): `openpick.system_one.SystemOne`.
- [x] `openpickd` daemon + `openpick` CLI.
- [x] `DecisionEngine` trait + `MockEngine` (deterministic, slightly
      jittered answers for testing).
- [x] **SDK compatibility surface** (the bit the future client
      actually depends on):
  - `x-typesafe-request-id` on every response, including 401s.
  - Bearer auth opt-in via `OPENPICK_API_KEY`.
  - `Retry-After` / `retry-after-ms` on 429 / 529.
  - Error envelope `{"error":{"code",message}}` mapped to the
    Python SDK's `TypeSafe{Authentication,RateLimit,BadRequest,...}Error`.
  - `/v1/models` returns `{"models":[{name,description,release_date}]}`.
- [x] 37 SDK compat tests + 5 HTTP integration tests + 1 gRPC
      roundtrip test = **82 tests passing** at HEAD.

### Phase 2 — real model backends (NOT STARTED)

Bring the engine from "mock" to "does useful work" without changing
the wire format.

- [ ] `openpick-runtime` — device discovery, VRAM accounting,
      worker pools.
- [ ] `openpick-backends` — candle (GGUF), ONNX runtime, optional
      remote-provider passthrough.
- [ ] Request scheduler — batch incoming `system_one` calls, share
      model instances across requests.
- [ ] Cancel tokens / timeout propagation from gRPC deadline headers.
- [ ] A real Python client (`openpick` or `typesafe_sdk`) that exercises
      a self-hosted `openpickd` end-to-end against a logged-in daemon.
- [ ] Eval harness: feed a labeled dataset through a model, capture
      the side-by-side comparison with the hosted API.

## Conventions for agents

### When you change a wire type

1. Edit the Rust struct in `crates/openpick-core/src/`.
2. Run `cargo test -p openpick-core` — fix any broken round-trip.
3. Regenerate the JSON Schema: `cargo run -p openpick-gen-schemas`.
4. Add a new fixture in `examples/` mirroring the Jev spec example.
5. Add an `sdk_compat.rs` test pinning the new shape.
6. Bump `JevRequest::SCHEMA_VERSION` (or whatever constants surface
   in `core::request`).
7. Update `docs/ARCHITECTURE.md` if the layering changed.

### When you touch the wire format

Any change to a JSON field in `core::request` or `core::response` is
a breaking change for every future SDK caller. Bisect-friendly
phrasing: the next non-trivial release must include a "Jev spec
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

# Run all tests (82+)
cargo test --workspace

# Regenerate JSON Schema
cargo run -p openpick-gen-schemas

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
