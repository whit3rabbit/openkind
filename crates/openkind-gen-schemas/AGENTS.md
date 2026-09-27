# AGENTS.md — openkind-gen-schemas

> LLM developer guide for `openkind-gen-schemas`. Read this before modifying schema generation.

## Crate Purpose & Boundaries

`openkind-gen-schemas` is a dedicated utility binary that generates official Draft 2020-12 JSON Schemas for `openkind_core::SystemRequest` and `openkind_core::SystemResponse`.

It uses `schemars` to inspect canonical Rust structs in `openkind-core` and writes JSON to stdout or updates committed schema files in place.

## Critical Invariants

1. **Schema Synchronization**:
   The committed JSON Schemas in `crates/openkind-core/schemas/` must exactly match the output derived from the Rust types.
2. **Never Hand-Edit Schemas**:
   Always modify the Rust struct definitions or docstrings in `openkind-core`, then use the [root verification command](../../AGENTS.md#verification) to regenerate them. Hand-edits will be overwritten or cause CI test failures in `tests/schema_sync.rs`.

## When to Run This Tool

You MUST run this binary whenever any struct, field, or doc comment changes in:
- `crates/openkind-core/src/request.rs`
- `crates/openkind-core/src/response.rs`
- `crates/openkind-core/src/question.rs`
- `crates/openkind-core/src/answer.rs`
- `crates/openkind-core/src/state.rs`

## Generated Schema Files

The generator writes these committed schema files:

- `crates/openkind-core/schemas/jev-v1-request.json`
- `crates/openkind-core/schemas/jev-v1-response.json`

## Verification Commands

```bash
# Run the schema synchronization test
cargo test -p openkind-gen-schemas

# Run core schema conformance tests
cargo test -p openkind-core --test conformance json_schema_generates_for_request_and_response
```
