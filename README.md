# openpick

> Open-source, **Jev-compatible decision inference engine** in Rust.
> Drop-in alternative to the hosted TypeSafe API —
> [what is Jev?](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

Jev is a schema and a model class for *fast, structured, type-safe
decisions* instead of *slow, generative chat*. `openpick` is the
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
| **Phase 0** — wire contract          | done        | 33    |
| **Phase 1** — daemon + SDK compat    | done        | 89 (122 total) |
| **Phase 2** — real model backends    | in progress | —     |

`cargo test --workspace` runs **122+** tests across all crates. The
SDK-compat contract (`tests/sdk_compat.rs`) is the executable
specification — every TypeSafe endpoint, header, and error code is
pinned there.

## Layout

```
crates/
├── openpick-core/        # Jev wire types + validation
├── openpick-engine/      # DecisionEngine trait + MockEngine
├── openpick-api/         # HTTP (axum 0.8) + gRPC (tonic 0.14)
├── openpick-server/      # openpickd daemon
├── openpick-cli/         # openpick CLI
├── openpick-runtime/     # (Phase 2) device/VRAM accounting
├── openpick-backends/    # (Phase 2) candle / GGUF / onnx
└── openpick-gen-schemas/ # JSON Schema codegen
proto/                    # openpick.proto
docs/                     # AGENTS.md, ARCHITECTURE.md
examples/                 # 8 Jev spec example fixtures
```

## Quickstart

```bash
# Build & test
cargo build --workspace
cargo test --workspace     # 122+ tests

# Regenerate JSON Schema
cargo run -p openpick-gen-schemas

# Run the daemon (mock engine, dev mode, no auth)
cargo run -p openpickd -- \
    --http-addr 127.0.0.1:18080 \
    --grpc-addr 127.0.0.1:19090 \
    --models mock,jev-latest

# Hit it with a Jev example
curl -sS -H 'content-type: application/json' \
     -X POST http://127.0.0.1:18080/v1/systemone \
     -d @examples/04_mixed.json | jq
```

Enable auth by setting `OPENPICK_API_KEY=***` and passing
`Authorization: Bearer ***`. See `docs/AGENTS.md` for the full
SDK-compat surface and the wire-format conventions.
