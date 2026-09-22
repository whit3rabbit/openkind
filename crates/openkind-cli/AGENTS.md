# AGENTS.md — openkind-cli

> LLM developer guide for `openkind-cli`. Read this before modifying the `openkind` CLI client.

## Crate Purpose & Boundaries

`openkind-cli` provides the `openkind` command-line utility. It is designed for:
- Developers validating request payloads offline without a running server (`inspect`).
- Operators sending test evaluations to a running daemon via HTTP (`evaluate`).
- Launching the standalone `openkindd` server binary (`serve`).
- Scripts and tooling checking the wire API compatibility version (`version`).

### Critical Invariants

1. **Self-Contained Client Dependencies**:
   - Uses `reqwest` for issuing HTTP POST calls to `/v1/systemone`.
   - Does NOT depend on `openkind-api` or `openkind-server`. It only depends on `openkind-core` (for types and validation) and `openkind-engine`.
2. **Deterministic Exit Codes**:
   - `inspect`: Returns exit code `0` when `validate_request` succeeds. Returns `1` on validation or JSON parse error, and `2` on file I/O error.
   - `evaluate`: Returns exit code `0` on HTTP 2xx. Exits with code `1` if the server returns any non-2xx status code.
3. **Input Guard**:
   - `MAX_CLI_INPUT_BYTES` (64 MiB) bounds file reading to prevent unbounded memory allocation on corrupt input files.
4. **Stdio Contract**:
   - The `--pretty` flag formats response JSON using `serde_json::to_string_pretty`.

## Key Files & Types

- [`src/main.rs`](./src/main.rs): Minimal CLI entrypoint and subcommand dispatch.
- [`src/args.rs`](./src/args.rs):
  - `Cli`: Root Clap parser.
  - `Commands`:
    - `Inspect { file }`: Validates a request JSON file against `openkind_core::validate_request`.
    - `Evaluate { file, server, api_key, pretty }`: POSTs the raw JSON to `{server}/v1/systemone`.
    - `Serve { http_addr, grpc_addr, models, api_key }`: Launches `openkindd`.
    - `Version`: Prints `openkind_core::api_version()`.
- [`src/inspect.rs`](./src/inspect.rs): `cmd_inspect` and input file validation bounds.
- [`src/evaluate.rs`](./src/evaluate.rs): `cmd_evaluate` and `cmd_evaluate_async` HTTP execution.
- [`src/serve.rs`](./src/serve.rs): `cmd_serve` process execution delegating to `openkindd`.
- [`src/tests.rs`](./src/tests.rs): Parser, inspect, and input bounds unit tests.

## Critical Gotchas & Rules

1. **No Direct Model Execution**:
   `openkind-cli` does not load neural model weights directly. For evaluation, it targets a running `openkindd` instance via HTTP. For offline validation, it evaluates request structure and JSON schema rules only.
2. **Standard Output Cleanliness**:
   In `evaluate` and `inspect`, only JSON output (or formatted errors) should reach stdout so downstream scripts can pipe results to tools like `jq`.

## Verification Commands

```bash
cargo check -p openkind-cli
cargo test -p openkind-cli
```
