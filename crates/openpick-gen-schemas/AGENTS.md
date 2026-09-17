# AGENTS.md — openpick-gen-schemas

> LLM developer guide for `openpick-gen-schemas`. Read this before modifying schema generation.

## Crate Purpose & Boundaries

`openpick-gen-schemas` is a dedicated utility binary that generates official draft-2020-12 JSON Schemas for `openpick_core::SystemRequest` and `openpick_core::SystemResponse`.

It uses `schemars` to inspect the canonical Rust structs in `openpick-core` and writes JSON to stdout.

## When to Run This Tool

You MUST run this binary whenever any struct, field, or doc comment changes in:
- `crates/openpick-core/src/request.rs`
- `crates/openpick-core/src/response.rs`
- `crates/openpick-core/src/question.rs`
- `crates/openpick-core/src/answer.rs`
- `crates/openpick-core/src/state.rs`

## How to Regenerate Schemas

Run the binary with `--write` to update the committed schema files in-place:

```bash
# Regenerate and write schemas directly to crates/openpick-core/schemas/
cargo run -p openpick-gen-schemas -- --write

# Verify diff on committed schemas:
git diff crates/openpick-core/schemas/
```

The committed schema files are:
- `crates/openpick-core/schemas/jev-v1-request.json`
- `crates/openpick-core/schemas/jev-v1-response.json`

## Verification Commands

```bash
cargo test -p openpick-core --test conformance json_schema_generates_for_request_and_response
```
