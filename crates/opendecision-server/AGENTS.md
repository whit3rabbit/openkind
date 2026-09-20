# AGENTS.md — opendecision-server

> LLM developer guide for `opendecision-server`. Read this before modifying the `opendecisiond` daemon binary, CLI arguments, or server lifecycle.

## Crate Purpose & Boundaries

`opendecision-server` is the binary crate that produces `opendecisiond`, the standalone daemon. It brings together:
- Configuration parsing (`clap` with environment variable fallbacks).
- Observability (`tracing_subscriber::fmt` with `EnvFilter`).
- Model initialization and registry population (`opendecision_engine::EngineRegistry`).
- Concurrently binding HTTP (`axum::serve`) and gRPC (`tonic::transport::Server`) listeners.
- Graceful shutdown orchestration on `SIGINT` (Ctrl-C) or `SIGTERM`.

### Critical Invariants

1. **Dual Protocol Listener Support**:
   - HTTP and gRPC bind to separate sockets (`--http-addr` default `0.0.0.0:8080`, `--grpc-addr` default `0.0.0.0:9090`).
   - If `--grpc-addr` has port 0, gRPC is cleanly disabled.
2. **Graceful Shutdown**:
   - Both HTTP and gRPC tasks MUST share a graceful shutdown signal future that catches both Unix `SIGTERM` and `SIGINT`.
   - On shutdown signal, in-flight inference requests complete before the process terminates.
3. **Environment Variable Parity**:
   - Every CLI flag has an environment variable fallback (`OPENDECISION_HTTP_ADDR`, `OPENDECISION_GRPC_ADDR`, `OPENDECISION_MODELS`, `OPENDECISION_API_KEY`, `RUST_LOG`).
4. **Metrics Recorder Initialization**:
   - The daemon MUST call `opendecision_api::http::install_metrics_recorder()` on startup before starting the server so that `/metrics` serves live counters.

## Key Files & Types

- [`src/main.rs`](./src/main.rs):
  - `Args`: Clap argument parser defining `--http-addr`, `--grpc-addr`, `--models`, `--api-key`, and `--log-filter`.
  - `main()`: Orchestrates tracing, auth configuration, registry creation, Prometheus registration, and concurrent task spawning (`tokio::join!`).
  - `shutdown_signal()`: Future selecting on `ctrl_c()` and Unix `terminate()`.

## Common Tasks

### Registering New Model Backends on Startup
The selected Qwen profile now passes tokenizer, renderer, CPU backbone, and
policy parity, but it is not ready for daemon registration. Wait for the
backend-neutral `BranchableState`, sequential/batched Q/K parity, explicit
capability limits, and native semantic-none wire decision before registration:
1. In `main.rs`, inspect `--models` aliases.
2. Instantiate the appropriate backend struct (or mock) depending on the configuration.
3. Register the engine into `EngineRegistry` under the designated alias before passing to `AppState`.

## Verification Commands

```bash
cargo test -p opendecision-server
```
