# AGENTS.md — opendecision-gen-schemas

> LLM developer guide for `opendecision-gen-schemas`. Read this before modifying schema generation.

## Crate Purpose & Boundaries

`opendecision-gen-schemas` is a dedicated utility binary that generates official draft-2020-12 JSON Schemas for `opendecision_core::SystemRequest` and `opendecision_core::SystemResponse`.

It uses `schemars` to inspect the canonical Rust structs in `opendecision-core` and writes JSON to stdout.

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
cargo test -p opendecision-core --test conformance json_schema_generates_for_request_and_response
```
