# AGENTS.md — opendecision-gen-schemas

> LLM developer guide for `opendecision-gen-schemas`. Read this before modifying schema generation.

## Crate Purpose & Boundaries

`opendecision-gen-schemas` is a dedicated utility binary that generates official Draft 2020-12 JSON Schemas for `opendecision_core::SystemRequest` and `opendecision_core::SystemResponse`.

It uses `schemars` to inspect canonical Rust structs in `opendecision-core` and writes JSON to stdout or updates committed schema files in place.

## Critical Invariants

1. **Schema Synchronization**:
   The committed JSON Schemas in `crates/opendecision-core/schemas/` must exactly match the output derived from the Rust types.
2. **Never Hand-Edit Schemas**:
   Always modify the Rust struct definitions or docstrings in `opendecision-core`, then run the generator with `--write`. Hand-edits will be overwritten or cause CI test failures in `tests/schema_sync.rs`.

## When to Run This Tool

You MUST run this binary whenever any struct, field, or doc comment changes in:
- `crates/opendecision-core/src/request.rs`
- `crates/opendecision-core/src/response.rs`
- `crates/opendecision-core/src/question.rs`
- `crates/opendecision-core/src/answer.rs`
- `crates/opendecision-core/src/state.rs`

## How to Regenerate Schemas

Run the binary with `--write` to update the committed schema files in-place:

```bash
# Regenerate and write schemas directly to crates/opendecision-core/schemas/
cargo run -p opendecision-gen-schemas -- --write

# Verify diff on committed schemas:
git diff crates/opendecision-core/schemas/
```

The committed schema files are:
- `crates/opendecision-core/schemas/jev-v1-request.json`
- `crates/opendecision-core/schemas/jev-v1-response.json`

## Verification Commands

```bash
# Run the schema synchronization test
cargo test -p opendecision-gen-schemas

# Run core schema conformance tests
cargo test -p opendecision-core --test conformance json_schema_generates_for_request_and_response
```
