# AGENTS.md — openpick-cli

> LLM developer guide for `openpick-cli`. Read this before modifying the `openpick` CLI client.

## Crate Purpose & Boundaries

`openpick-cli` provides the `openpick` command-line utility. It is designed for:
- Developers validating request payloads offline without a running server (`inspect`).
- Operators sending test evaluations to a running daemon via HTTP (`evaluate`).
- Scripts and tooling checking the wire API compatibility version (`version`).

### Critical Invariants

1. **Self-Contained Client**:
   - Uses `reqwest` for issuing HTTP POST calls to `/v1/systemone`.
   - Does NOT depend on `openpick-api` or `openpick-server`. It only depends on `openpick-core` (for types and validation) and `openpick-engine`.
2. **Deterministic Exit Codes**:
   - `inspect`: Returns exit code `0` when `validate_request` succeeds. Returns non-zero on I/O failure, JSON parse errors, or validation errors.
   - `evaluate`: Returns exit code `0` on HTTP 2xx. Exits with code `1` if the server returns any non-2xx status code.
3. **Stdio Contract**:
   - The `--pretty` flag formats response JSON using `serde_json::to_string_pretty`.

## Key Files & Types

- [`src/main.rs`](./src/main.rs):
  - `Cli`: Root Clap parser.
  - `Commands`:
    - `Inspect { file }`: Validates a request JSON file against `openpick_core::validate_request`.
    - `Evaluate { file, server, pretty }`: POSTs the raw JSON to `{server}/v1/systemone`.
    - `Version`: Prints `openpick_core::api_version()`.

## Verification Commands

```bash
cargo test -p openpick-cli
```
