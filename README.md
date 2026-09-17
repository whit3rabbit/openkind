# openpick

Open-source, Jev-compatible decision inference engine.

Implements the [TypeSafe Jev API contract](https://docs.typesafe.ai/api) (the `/v1/systemone` protocol). Phase 0 is the schema and request/response types — pinned to the Jev spec verbatim so existing TypeSafe clients work unmodified.

## Phase 0 (this commit)

- `openpick-core` — protocol types, validation, JSON schema generation, conformance tests.
- No networking, no model. Just the schema, validated against the official Jev examples.

## Layout

```
crates/
├── openpick-core/       # Phase 0: types + validation + JSON schemas
├── openpick-api/        # Phase 1: axum + tonic
├── openpick-server/     # Phase 1: openpickd daemon
├── openpick-engine/     # Phase 2: scheduler + DecisionEngine trait
├── openpick-runtime/    # Phase 2: tensor + device abstraction
├── openpick-backends/   # Phase 2: candle / GGUF / native backends
└── openpick-cli/        # Phase 1: `openpick serve | evaluate | inspect`
```

## Conformance

`cargo test -p openpick-core` runs the schema through every example request/response pair from docs.typesafe.ai/api and asserts round-trip equality.
