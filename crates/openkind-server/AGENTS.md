# AGENTS.md — openkind-server

> LLM developer guide for `openkind-server`. Read this before modifying the `openkindd` daemon binary, CLI arguments, or server lifecycle.

## Crate Purpose & Boundaries

`openkind-server` is the binary crate that produces `openkindd`, the standalone inference daemon. It brings together:
- Configuration parsing (`clap` with environment variable fallbacks for all flags).
- Observability initialization (`tracing_subscriber::fmt` with `EnvFilter`).
- Prometheus metrics recorder installation (`openkind_api::http::install_metrics_recorder`).
- Model engine instantiation and registration into `EngineRegistry` (supporting both `MockEngine` and `Qwen35DecisionEngine`).
- Concurrent HTTP (`axum::serve`) and gRPC (`tonic::transport::Server`) listeners.
- Graceful shutdown orchestration on Unix `SIGINT` (Ctrl-C) or `SIGTERM`.

### Critical Invariants

1. **Dual Protocol Listener Support**:
   - HTTP and gRPC bind to separate sockets (`--http-addr` default `0.0.0.0:8080`, `--grpc-addr` default `0.0.0.0:9090`).
   - The literal `--grpc-addr 0` (also `off`, `none`, or `disabled`) disables gRPC. A normal `host:0` address requests an ephemeral port and is not the disable sentinel.
2. **Graceful Shutdown**:
   - Both HTTP and gRPC listener tasks share a shutdown signal future that listens for Unix `SIGTERM` and `SIGINT`.
   - On signal receipt, active in-flight inference requests complete before the process exits.
3. **Environment Variable Parity**:
   - Every CLI flag has an identical environment variable fallback (e.g. `--http-addr` / `OPENKIND_HTTP_ADDR`, `--models` / `OPENKIND_MODELS`, `--qwen35-bundle-root` / `OPENKIND_QWEN35_BUNDLE_ROOT`).
4. **Metrics Recorder Initialization**:
   - The daemon MUST call `openkind_api::http::install_metrics_recorder()` once on startup before binding HTTP routes so that `/metrics` serves live counters.

## Key Files & Types

- [`src/main.rs`](./src/main.rs):
  - `Args`: Clap argument parser defining:
    - Server endpoints: `--http-addr`, `--grpc-addr`, `--api-key`, `--log-filter`.
    - Model aliases: `--models`, `--qwen35-aliases`.
    - Native Qwen3.5 parameters: `--qwen35-bundle-root`, `--qwen35-checkpoint-root`, `--qwen35-tokenizer`, `--qwen35-concurrency`, `--qwen35-queue`, `--qwen35-max-tensor-bytes`, `--qwen35-max-process-bytes`, `--qwen35-scratch-bytes`, `--qwen35-allocator-headroom-bytes`, `--qwen35-execution` (diagnostic plan override: `auto` default; bypasses the profitability policy only — admission ceilings and backend capabilities still apply).
  - `main()`: Orchestrates logging, auth, Prometheus recorder, engine registration, and concurrent listener tasks via `tokio::join!`.
  - `shutdown_signal()`: Future selecting on `tokio::signal::ctrl_c()` and Unix `SIGTERM`.

## Engine Registration Architecture

`openkindd` populates `EngineRegistry` dynamically based on `--models`:
1. **Native Engine Path**: If any alias in `--models` is listed in `--qwen35-aliases` (default `qwen35-native`):
   - Validates that `--qwen35-bundle-root`, `--qwen35-checkpoint-root`, and `--qwen35-tokenizer` are provided.
   - Instantiates `SchedulerConfig::for_pinned_profile` with `BackendCapabilities::per_lane()`.
   - Optionally attaches a process-memory envelope if `--qwen35-max-process-bytes` is configured. The engine refreshes peak RSS after model load and immediately before each request, then divides remaining headroom across the concurrency limit.
   - Loads `Qwen35DecisionEngine` with configured concurrency and queue semaphores.
   - Registers the shared engine under each matching alias.
2. **Mock Engine Path**: Any alias not matching `--qwen35-aliases` registers an instance of `MockEngine`.

## Critical Gotchas & Rules

1. **Native Path Fail-Closed**:
   If a user requests a native alias in `--models` but omits any of the required path flags (`bundle-root`, `checkpoint-root`, `tokenizer`), the daemon fails fast on startup with a contextual error.
2. **Metrics Recorder Single-Init**:
   `install_metrics_recorder()` panics if called more than once in the same process. It must be called strictly once during initialization.
3. **Graceful Drain**:
   When orchestrating shutdown, drop guards ensure server tasks drain in-flight evaluations. Do not call `std::process::exit(0)` abruptly from signal handlers.
4. **Cancellation Does Not Free Native Capacity Early**:
   `Qwen35DecisionEngine` moves queue and execution permits into the blocking model task. If an HTTP or gRPC caller disconnects, replacement work remains blocked until the native task actually finishes.

## Verification Commands

```bash
cargo check -p openkind-server
cargo test -p openkind-server
```
