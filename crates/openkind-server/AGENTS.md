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
    - Server addresses and auth, plus rate limits (`OPENKIND_RATE_LIMIT_RPM`, default 120; `0` disables). `--playground` and `--arrow` are opt-in. See the [API guide](../openkind-api/AGENTS.md) and [Arrow guide](../../docs/ARROW.md).
    - Model aliases: `--models`, `--qwen35-aliases`, `--installed-models`, and `--models-dir`.
    - Surveyed-family configuration: flattened `family_args: FamilyArgs`.
    - Native Qwen uses bundle/checkpoint/tokenizer paths, backend selection, concurrency and queue limits, and a queue-inclusive timeout (default 600000 ms). Memory ceilings and `--qwen35-execution` configure admission and diagnostic scheduling.
- [`src/playground.rs`](./src/playground.rs): Explicit load/unload of supported local installations and mock aliases. Blocking verification runs outside async workers; mutations serialize across clients. Installation guards stay alive until shutdown, and startup native/composite engines require restart.
- [`src/installed.rs`](./src/installed.rs): Loads catalog models for startup and the playground. `installed_kind` validates manifest identity against compiled loaders. Winnow binds installed `decoder-logit-letter` to A and `encoder-nli` to B. Missing siblings fall back to aliases or fail closed.
- [`src/proxy.rs`](./src/proxy.rs): Proxy-cache service (only when `--proxy-cache-upstream` is set):
  - Flags cover upstream, model aliases, encoder, backend, data directory, credentials, timeout, agreement, text storage, and bootstrap thresholds. Each flag has an `OPENKIND_PROXY_CACHE_*` environment alias.
  - `ProxyService` implements the [`openkind-api`](../openkind-api/AGENTS.md) `SystemProxy` hook. It groups choice questions, embeds state once, routes through the [`openkind-backends`](../openkind-backends/AGENTS.md) manager, and forwards other requests with the caller's bearer key. Responses include cache headers.
  - Internal errors fail open to upstream. Unverified keys are forwarded and trusted only after a parsed answer. The daemon keeps only salted key hashes in memory.
  - Encoder resolution is fail-closed: a missing installation errors with the `openkind pull` instruction (regular or MLX profile). The daemon never downloads during startup (model-store invariant).
- [`src/families.rs`](./src/families.rs):
  - `FamilyArgs` holds aliases and model roots for surveyed families and composite routers. MLX-capable family backends are feature-gated and macOS arm64 only.
  - `FamilyAdmission`: Concurrency, queue, and timeout parameters for family engines (`--family-concurrency`, `--family-queue`, `--family-timeout-ms`).
  - Fail-closed validation for duplicate or missing artifact configurations.
- [`benches/server.rs`](./benches/server.rs): Criterion coverage for the complete
  authenticated in-memory Axum path with rate limiting disabled and MockEngine.

## Engine Registration Architecture

`openkindd` populates `EngineRegistry` from `--models` and `--installed-models`:
1. **Native Engine Path**: If any alias in `--models` is listed in `--qwen35-aliases` (default `qwen35-native`):
   - Validates that `--qwen35-bundle-root`, `--qwen35-checkpoint-root`, and `--qwen35-tokenizer` are provided.
   - Defaults to per-lane execution. MLX advertises vectorization only for explicitly forced nested-batched FP32 `ReferenceOps`. Automatic scheduling stays per-lane pending measured performance. MLX requires the feature on macOS arm64.
   - Optionally attaches a process-memory envelope if `--qwen35-max-process-bytes` is configured. The engine refreshes peak RSS after model load and immediately before each request, then divides remaining headroom across the concurrency limit.
   - Loads `Qwen35DecisionEngine` with configured concurrency and queue semaphores.
   - Registers the shared engine under each matching alias.
2. **Installed Model Path**: The daemon verifies each installation against a compiled loader and registers its immutable name. It holds a serving lock until shutdown. Missing profiles and alias collisions fail startup. Startup never fetches the public catalog.
3. **Surveyed-Family Engine Path**: If any alias in `--models` matches `--decoder-letter-aliases`, `--encoder-nli-aliases`, `--encoder-instruct-label-aliases`, `--kev-aliases`, `--decoder-llm-aliases`, `--schema-scorer-aliases`, `--qwen3guard-aliases`, `--decoder-logit-qwen35-aliases`, `--laya-english-aliases`, `--laya-multilingual-aliases`, or `--laya-typed-decisions-aliases`:
   - Validates that the corresponding `--<family>-model-root` is provided (fails fast on startup if omitted).
   - Loads the family adapter with bounded `FamilyLimits`. MLX-capable families default to `native-cpu`. Requesting `mlx-fp32` without the daemon feature fails startup.
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
   A disconnect does not release native permits. The blocking task owns queue and execution permits until cancellation exits at a CPU layer boundary. The queue-inclusive timeout defaults to 600000 ms.
5. **Telemetry Privacy**:
   Native metrics use fixed outcome labels and durations only. HTTP spans exclude headers. Never log request IDs, auth headers, state/question/candidate content, token IDs, content fingerprints, or input digests.
6. **Service Evidence Boundary**:
   Service load and soak evidence does not establish model quality, Metal behavior, or release readiness. Use [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for methodology and current evidence.
7. **Family Path Fail-Closed**:
   Fail startup if a surveyed-family alias lacks its model root or appears in multiple families.

## Verification Commands

```bash
cargo check -p openkind-server
cargo test -p openkind-server
cargo test -p openkind-server --bench server
cargo bench -p openkind-server --bench server -- --noplot
```
