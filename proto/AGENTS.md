# AGENTS.md — openkind-proto

> LLM developer guide for `openkind-proto`. Read this before modifying the Protobuf definitions or build script.

## Crate Purpose & Boundaries

`openkind-proto` contains the Protocol Buffers definition ([`proto/proto/openkind.proto`](proto/openkind.proto)) and build-time code generator (`build.rs`) for the `openkind.SystemOne` gRPC service.

It generates the Rust structs and client/server service traits consumed by `openkind-api::grpc`.

The field tags and message layout live in [`proto/openkind.proto`](proto/openkind.proto). Use that file as the schema source; this guide records only cross-language invariants and change hazards.

### Critical Invariants

1. **Protobuf $\leftrightarrow$ Core Semantic Parity**:
   - `openkind.proto` must maintain exact semantic parity with the JSON wire format in `openkind-core`.
   - `state` uses a `oneof` supporting `string text = 1` or `Structured structured = 2` (raw JSON bytes).
   - `instructions_json` is serialized JSON bytes to preserve the `string | object | array` polymorphism of the Jev spec.
   - Criteria strings for Noul use `string is_true = 1` and `string is_false = 2` to avoid keyword collisions with Protobuf/Rust `true`/`false`.
   - Float fields (`noul`, `score`, `confidence`, `probabilities` values) MUST be `double` (64-bit IEEE 754), matching `openkind-core`'s `f64`. **Never use `float`**.
2. **Build-Time Compilation (`build.rs`)**:
   - Compiles during `cargo build` using `tonic-prost-build` and `prost-build`.
   - Generated code is written to `OUT_DIR` and re-exported via `src/lib.rs` (`tonic::include_proto!("openkind")`).

## Critical Gotchas & Rules

1. **Synchronized gRPC Conversions**:
   Whenever a message in `proto/proto/openkind.proto` changes, the bidirectional conversion functions in `crates/openkind-api/src/grpc.rs` (`pb_state_to_core`, `pb_questions_to_core`, `core_to_pb_response`) must be updated to keep HTTP and gRPC behavior identical.
2. **Wire Precision and Answer Shape**:
   Keep Protobuf floats as `double`. `NoulAnswer` has no confidence field.

## Step-by-Step Change Protocol

1. Edit `proto/proto/openkind.proto`. Ensure doc comments cross-reference the canonical Jev API (`https://docs.typesafe.ai/api`) and Python SDK (`https://docs.typesafe.ai/sdk/python/api`).
2. Regenerate bindings with the build command under Verification Commands.
3. Update conversions in `crates/openkind-api/src/grpc.rs`.
4. Keep wire parity with OpenAPI (`crates/openkind-api/openapi.yaml` / `docs/openapi.yaml`).
5. Run the gRPC integration command in the [API guide](../crates/openkind-api/AGENTS.md).

## Verification Commands

```bash
# Build proto crate and regenerate bindings
cargo build -p openkind-proto

```
