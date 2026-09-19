# opendecision

> Open-source, **Jev-compatible decision inference engine** in Rust.
> Drop-in alternative to the hosted TypeSafe API —
> [what is Jev?](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

Jev is a schema and a model class for *fast, structured, type-safe
decisions* instead of *slow, generative chat*. `opendecision` is the
open-source server that speaks that protocol end-to-end: same wire
format, same Python `typesafe_sdk` surface, but you own the hardware
and the model weights.

## Read me first

- [`docs/AGENTS.md`](./docs/AGENTS.md) — goal, source-of-truth links,
  phased roadmap, and agent-facing conventions.
- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — crate layout,
  layering, error mapping, test strategy.

## Status

| Phase                                | Status      | Tests |
|--------------------------------------|-------------|-------|
| **Phase 0** — wire contract          | done        | 43    |
| **Phase 1** — daemon + SDK compat    | done        | 152 (195 workspace total) |
| **Phase 2** — empirical model research (2A–2G done; 2H bridge next) | in progress | — |
| **Phase 3** — production Rust engine & native backends | planned | — |

`cargo test --workspace` runs **195** tests across all crates at commit HEAD. The
SDK-compat contract (`crates/opendecision-api/tests/sdk_compat.rs`) is the executable
specification — every TypeSafe endpoint, header, and error code is
pinned there.

## Layout

```
crates/
├── opendecision-core/        # Jev wire types + validation
├── opendecision-engine/      # DecisionEngine trait + MockEngine
├── opendecision-api/         # HTTP (axum 0.8) + gRPC (tonic 0.14)
├── opendecision-server/      # opendecisiond daemon
├── opendecision-cli/         # opendecision CLI
├── opendecision-client/      # async Rust client SDK (typesafe_sdk counterpart)
├── opendecision-runtime/     # device discovery & VRAM accounting
├── opendecision-backends/    # native drivers (candle / GGUF / onnx)
└── opendecision-gen-schemas/ # JSON Schema codegen
proto/                    # opendecision.proto
docs/                     # AGENTS.md, ARCHITECTURE.md, ROADMAP.md, RESEARCH.md
examples/                 # 8 Jev spec example fixtures
```

## Quickstart

```bash
# Build & test
cargo build --workspace
cargo test --workspace     # 195 tests at HEAD

# Regenerate JSON Schema
cargo run -p opendecision-gen-schemas -- --write

# Run the daemon (mock engine, dev mode, no auth)
cargo run -p opendecisiond -- \
    --http-addr 127.0.0.1:18080 \
    --grpc-addr 127.0.0.1:19090 \
    --models mock,jev-latest

# Hit it with a Jev example
curl -sS -H 'content-type: application/json' \
     -X POST http://127.0.0.1:18080/v1/systemone \
     -d @examples/04_mixed.json | jq
```

Enable auth by setting `OPENDECISION_API_KEY=***` and passing
`Authorization: Bearer ***`. See `docs/AGENTS.md` for the full
SDK-compat surface and the wire-format conventions.
