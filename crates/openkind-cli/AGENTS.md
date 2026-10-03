# AGENTS.md — openkind-cli

> LLM developer guide for `openkind-cli`. Read this before modifying the `openkind` CLI client.

## Crate Purpose & Boundaries

`openkind-cli` provides the `openkind` command-line utility. It is designed for:
- Developers validating request payloads offline without a running server (`inspect`).
- Operators sending test evaluations to a running daemon via HTTP (`evaluate`).
- Launching the standalone `openkindd` server binary (`serve`).
- Discovering and installing curated models (`catalog`, `pull`) and managing
  local installations without network access (`list`, `show`, `rm`).
- Checking daemon health and registered model aliases (`status`).
- Scripts and tooling checking the wire API compatibility version (`version`).

### Critical Invariants

1. **Self-Contained Client Dependencies**:
   - Uses `reqwest` for issuing HTTP POST calls to `/v1/systemone`.
   - Does NOT depend on `openkind-api` or `openkind-server`. It declares
     `openkind-core`, `openkind-engine`, and `openkind-model-store` as workspace
     dependencies; the model store owns discovery and local installations.
2. **Deterministic Exit Codes**:
   - `inspect`: Returns exit code `0` when `validate_request` succeeds. Returns `1` on validation, JSON parse, or file I/O errors (all propagate as `anyhow` errors out of `main`); exit code `2` is reserved for Clap usage errors.
   - `evaluate`: Returns exit code `0` for a valid successful response. Exits with code `1` for non-2xx responses, malformed JSON, or answers that fail validation against the request.
3. **Input Guard**:
   - `MAX_CLI_INPUT_BYTES` (32 MB) bounds file reading to prevent unbounded memory allocation on corrupt input files.
4. **Stdio Contract**:
   - Successful `evaluate` responses are JSON on stdout by default. HTTP status and error details go to stderr.
   - `--pretty` formats JSON using `serde_json::to_string_pretty`; `--format text` opts into typed answer rows.
   - Pull progress stays on stderr. Text output uses TTY detection and honors `NO_COLOR`.

## Key Files & Types

- [`src/main.rs`](./src/main.rs): Minimal CLI entrypoint and subcommand dispatch.
- [`src/lib.rs`](./src/lib.rs): Benchmarkable library seam exposing parsed-command
  dispatch and pure request parse/validation.
- [`src/args.rs`](./src/args.rs):
  - `Cli`: Root Clap parser with a streaming fast path (`fast_parse` using `SmallVec`) skipping Clap machinery on common unambiguous invocations for sub-millisecond CLI startup.
  - `Commands`:
    - `Inspect { file }`: Validates a request JSON file against `openkind_core::validate_request`.
    - `Evaluate { file, server, api_key, pretty, format, verbose }`: POSTs the raw JSON to `{server}/v1/systemone`.
    - `Serve { ... }`: Launches `openkindd`, forwarding explicit installed
      models and the shared model store directory.
    - `Playground { ... }`: Connects to a healthy daemon or starts and supervises one with the playground enabled. It waits for `/health`, opens the browser or prints the URL, and forwards shutdown signals (`SIGINT`/`SIGTERM` on Unix; on Windows the daemon observes the same console Ctrl-C event and the supervisor reaps its real exit status).
    - `Catalog`, `Pull`, `List`, `Show`, `Rm`: Curated discovery and local
      installation management, with JSON output flags for read commands.
    - `Status { server, api_key, watch }`: Checks `/health` and lists aliases from `/v1/models`. `--watch` refreshes the view in a Bubble Tea terminal program.
    - `Doctor { json, cuda_device, rocm_device, onnx_runtime }`: Reports paired
      daemon readiness without loading models or opening listeners.
    - `Version`: Prints `openkind_core::api_version()`.
- [`src/inspect.rs`](./src/inspect.rs): `cmd_inspect` and input file validation bounds.
- [`src/evaluate.rs`](./src/evaluate.rs): `cmd_evaluate` and `cmd_evaluate_async` HTTP execution.
- [`src/output.rs`](./src/output.rs): Shared text tables, color policy, and pull progress rendering.
- `status --watch` uses `bubbletea-rs` with Lipgloss styles. Plain `status` stays one-shot.
- [`src/status.rs`](./src/status.rs): One-shot health and model alias inspection, plus the live `--watch` screen.
- [`src/serve.rs`](./src/serve.rs): `cmd_serve` process execution delegating to `openkindd`, replacing the CLI process on Unix so shutdown signals reach the daemon directly (on Windows the wrapper blocks on the daemon; both observe the shared console Ctrl-C event).
- [`src/daemon.rs`](./src/daemon.rs): Shared executable resolution for doctor,
  serve, and playground: `OPENKINDD_BINARY`, sibling executable, then `PATH`.
- [`src/playground.rs`](./src/playground.rs): `cmd_playground` — health probe,
  daemon spawn/supervise, and the per-platform browser opener.
- [`src/models.rs`](./src/models.rs): Online catalog and pull commands, plus
  offline installation reads and removal. `catalog` formats curated models into
  a table showing name, aliases, profile, status, token context limit, and description.
  See the [registry guide](../../docs/MODEL_REGISTRY.md) for public mirror ownership.
- [`src/tests.rs`](./src/tests.rs): Parser, inspect, and input bounds unit tests.
- [`benches/cli.rs`](./benches/cli.rs): Criterion argument-parsing and in-memory
  inspect benchmarks for 1, 8, and 32 questions.

## Critical Gotchas & Rules

1. **No Direct Model Execution**:
   The CLI never loads neural weights. `evaluate` sends requests to `openkindd`; `inspect` checks request structure and schema only. Only `pull` contacts model hosts. `list`, `show`, and `rm` use local files.
2. **Standard Output Cleanliness**:
   Successful `evaluate` calls emit JSON to stdout by default. `--format text` opts into human-readable rows. HTTP errors and pull progress go to stderr. `inspect` retains its concise validation summary.

## Verification Commands

```bash
cargo check -p openkind-cli
cargo test -p openkind-cli
cargo test -p openkind-cli --bench cli
cargo bench -p openkind-cli --bench cli -- --noplot
```
