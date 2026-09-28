# AGENTS.md — openkind-server

> LLM developer guide for `openkind-server`. Read this before modifying the `openkindd` daemon binary, CLI arguments, or server lifecycle.

## Crate Purpose & Boundaries

`openkind-server` is the binary crate that produces `openkindd`, the standalone inference daemon. It brings together:
- Configuration parsing (`clap` with environment variable fallbacks for all flags).
- Observability initialization (`tracing_subscriber::fmt` with `EnvFilter`).
- Prometheus metrics recorder installation (`openkind_api::http::install_metrics_recorder`).
- Model engine instantiation and registration into `EngineRegistry` (supporting both `MockEngine` and `Qwen35DecisionEngine`).
- Explicit startup loading of verified installations from `openkind-model-store`.
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
  - `main()`: Daemon entrypoint; orchestrates logging, auth, Prometheus recorder, engine registration (Qwen 3.5, installed models, surveyed families, router-script and winnow composites, and mock), and concurrent listener tasks via `tokio::join!`.
  - `shutdown_signal()`: Future selecting on `tokio::signal::ctrl_c()` and Unix `SIGTERM`.
- [`src/lib.rs`](./src/lib.rs): Library re-exports for daemon and benchmark integration.
- [`src/args.rs`](./src/args.rs): Clap argument parser defining:
    - Server endpoints: `--http-addr`, `--grpc-addr`, `--api-key`, `--rate-limit-rpm` (`OPENKIND_RATE_LIMIT_RPM`, default 120; `0` disables), `--playground` (`OPENKIND_PLAYGROUND`, `on`/`off`, default `off`; adds the `GET /playground` UI and gated local model controls — see [`openkind-api`](../openkind-api/AGENTS.md)), `--log-filter`.
    - Model aliases: `--models`, `--qwen35-aliases`, `--installed-models`, and `--models-dir`.
    - Surveyed-family configuration: flattened `family_args: FamilyArgs`.
    - Native Qwen3.5 parameters: `--qwen35-bundle-root`, `--qwen35-checkpoint-root`, `--qwen35-tokenizer`, `--qwen35-backend` / `OPENKIND_QWEN35_BACKEND` (`native-cpu` default, or feature-gated `mlx-fp32` on macOS arm64), `--qwen35-concurrency`, `--qwen35-queue`, `--qwen35-timeout-ms` (queue-inclusive, default 600000), `--qwen35-max-tensor-bytes`, `--qwen35-max-process-bytes`, `--qwen35-scratch-bytes`, `--qwen35-allocator-headroom-bytes`, `--qwen35-execution` (diagnostic plan override: `auto` default; bypasses the profitability policy only — admission ceilings and backend capabilities still apply).
- [`src/playground.rs`](./src/playground.rs): Explicit load/unload of supported local installations and mock aliases. Blocking verification runs outside async workers; mutations serialize across clients. Installation guards stay alive until shutdown, and startup native/composite engines require restart.
- [`src/families.rs`](./src/families.rs):
  - `FamilyArgs`: Flags and environment variables for surveyed-family loaders (`--decoder-letter-aliases`, `--decoder-letter-model-root`, `--encoder-nli-aliases`, `--encoder-nli-model-root`, `--encoder-instruct-label-aliases`, `--encoder-instruct-label-model-root`, `--kev-aliases`, `--kev-model-root`, `--kev-base-root`, `--decoder-llm-aliases`, `--decoder-llm-model-root`, `--schema-scorer-aliases`, `--schema-scorer-model-root`, `--router-script-aliases`, `--router-script-rules`, `--qwen3guard-aliases`, `--qwen3guard-model-root`, `--winnow-aliases`, `--winnow-model-root`, `--winnow-adapter`, `--winnow-siblings`).
  - `FamilyAdmission`: Concurrency, queue, and timeout parameters for family engines (`--family-concurrency`, `--family-queue`, `--family-timeout-ms`).
  - Fail-closed validation for duplicate or missing artifact configurations.
- [`benches/server.rs`](./benches/server.rs): Criterion coverage for the complete
  authenticated in-memory Axum path with rate limiting disabled and MockEngine.

## Engine Registration Architecture

`openkindd` populates `EngineRegistry` from `--models` and `--installed-models`:
1. **Native Engine Path**: If any alias in `--models` is listed in `--qwen35-aliases` (default `qwen35-native`):
   - Validates that `--qwen35-bundle-root`, `--qwen35-checkpoint-root`, and `--qwen35-tokenizer` are provided.
   - Instantiates `SchedulerConfig::for_pinned_profile` with `BackendCapabilities::per_lane()`. The MLX loader may advertise vectorized forward only for FP32 `ReferenceOps` when `--qwen35-execution nested-batched` is explicitly forced; automatic scheduling stays per-lane pending measured performance. MLX server startup requires the `mlx` crate feature and is available only on macOS arm64.
   - Optionally attaches a process-memory envelope if `--qwen35-max-process-bytes` is configured. The engine refreshes peak RSS after model load and immediately before each request, then divides remaining headroom across the concurrency limit.
   - Loads `Qwen35DecisionEngine` with configured concurrency and queue semaphores.
   - Registers the shared engine under each matching alias.
2. **Installed Model Path**: Each name in `--installed-models` is read from
   `--models-dir` or `OPENKIND_MODELS_DIR`, verified, matched to a compiled-in
   loader, and registered under its immutable name. A serving lock remains
   held until shutdown; missing profiles and alias collisions fail startup.
   The public catalog is never fetched during daemon startup. See the
   [registry guide](../../docs/MODEL_REGISTRY.md) for the mirror and sync flow.
3. **Surveyed-Family Engine Path**: If any alias in `--models` matches `--decoder-letter-aliases`, `--encoder-nli-aliases`, `--encoder-instruct-label-aliases`, `--kev-aliases`, `--decoder-llm-aliases`, `--schema-scorer-aliases`, or `--qwen3guard-aliases`:
   - Validates that the corresponding `--<family>-model-root` is provided (fails fast on startup if omitted).
   - Loads the respective engine adapter from `openkind_backends::families` with bounded `FamilyLimits`.
   - Registers the shared engine under each matching alias.
4. **Router-Script Composite Path**: If any alias in `--models` matches `--router-script-aliases`:
   - Parses the routing rule table (`--router-script-rules`).
   - Resolves sibling engine references registered under `--models`.
   - Registers `RouterScriptEngine` dispatching across the sibling engines.
5. **Winnow Composite Path**: If any alias in `--models` matches `--winnow-aliases`:
   - Loads the pinned adapter from `--winnow-adapter` and resolves the sibling engines named by `--winnow-siblings` (`A=<alias>,B=<alias>`).
   - Registers `WinnowEngine` dispatching across the two sibling engines.
6. **Mock Engine Path**: Any alias in `--models` not matching native, surveyed-family, router-script, or winnow configurations registers an instance of `MockEngine`.

## Critical Gotchas & Rules

1. **Native Path Fail-Closed**:
   If a user requests a native alias in `--models` but omits any of the required path flags (`bundle-root`, `checkpoint-root`, `tokenizer`), the daemon fails fast on startup with a contextual error.
2. **Metrics Recorder Single-Init**:
   `install_metrics_recorder()` is idempotent — a second call returns `Ok(())` if a recorder is already installed — but it should still be called exactly once during initialization, before binding HTTP routes.
3. **Graceful Drain**:
   When orchestrating shutdown, drop guards ensure server tasks drain in-flight evaluations. Do not call `std::process::exit(0)` abruptly from signal handlers.
4. **Cancellation Does Not Free Native Capacity Early**:
   `Qwen35DecisionEngine` moves queue and execution permits into the blocking model task. If an HTTP or gRPC caller disconnects, replacement work remains blocked until cooperative cancellation reaches a CPU decoder-layer boundary and the native task exits. `--qwen35-timeout-ms` is a queue-inclusive deadline (default 600000 ms).
5. **Telemetry Privacy**:
   Native metrics use fixed outcome labels and durations only. HTTP spans exclude headers. Never log request IDs, auth headers, state/question/candidate content, token IDs, content fingerprints, or input digests.
6. **Service Evidence Boundary**:
   The release-mode native CPU load/soak evidence is recorded in [`docs/verification/native-service-gate/2026-09-22/`](../../docs/verification/native-service-gate/2026-09-22/README.md). Its fixture has no reviewed labels and does not establish model quality, Metal behavior, or product-release promotion; use [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for the canonical methodology.
7. **Family Path Fail-Closed**:
   Requesting a surveyed-family alias without its required `--<family>-model-root` or assigning the same alias to multiple families fails fast on daemon startup with a descriptive configuration error.

## Verification Commands

```bash
cargo check -p openkind-server
cargo test -p openkind-server
cargo test -p openkind-server --bench server
cargo bench -p openkind-server --bench server -- --noplot
```
