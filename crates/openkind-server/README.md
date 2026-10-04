# openkind-server

> The `openkindd` production inference daemon.

`openkind-server` provides the `openkindd` daemon binary that powers self-hosted `openkind` deployments. It sets up structured tracing, initializes the `EngineRegistry`, mounts both HTTP (axum) and gRPC (tonic) listeners, registers Prometheus metrics, and manages graceful shutdown on Ctrl-C (`SIGINT`, plus `SIGTERM` on Unix).

## Installation & Build

### Homebrew (macOS & Linux)

```bash
brew install whit3rabbit/tap/openkind
```

### Cargo

```bash
cargo install --locked openkind-server
```

### Build from source

```bash
cargo build --release -p openkind-server
# Binary produced at target/release/openkindd
```

## Command-Line Usage

```bash
openkindd [OPTIONS]
```

### Options

| Flag | Environment Variable | Default | Description |
|---|---|---|---|
| `--http-addr <ADDR>` | `OPENKIND_HTTP_ADDR` | `127.0.0.1:8080` | Address to bind HTTP server on |
| `--grpc-addr <ADDR>` | `OPENKIND_GRPC_ADDR` | `127.0.0.1:9090` | Address to bind gRPC server on (port 0 or literal `0`/`off`/`none`/`disabled` disables gRPC) |
| `--models <ALIASES>` | `OPENKIND_MODELS` | `mock,jev-latest` | Comma-separated list of model aliases to register |
| `--qwen35-aliases <ALIASES>` | `OPENKIND_QWEN35_ALIASES` | `qwen35-native` | Aliases in `--models` that use the native Qwen engine |
| `--qwen35-bundle-root <PATH>` | `OPENKIND_QWEN35_BUNDLE_ROOT` | *(none)* | Offline selected-profile bundle root required by native aliases |
| `--qwen35-checkpoint-root <PATH>` | `OPENKIND_QWEN35_CHECKPOINT_ROOT` | *(none)* | Offline pinned checkpoint root required by native aliases |
| `--qwen35-tokenizer <PATH>` | `OPENKIND_QWEN35_TOKENIZER` | *(none)* | Digest-locked tokenizer JSON required by native aliases |
| `--qwen35-concurrency <N>` | `OPENKIND_QWEN35_CONCURRENCY` | `1` | Maximum concurrent native evaluations |
| `--qwen35-queue <N>` | `OPENKIND_QWEN35_QUEUE` | `2` | Additional native requests allowed to wait |
| `--qwen35-max-tensor-bytes <N>` | `OPENKIND_QWEN35_MAX_TENSOR_BYTES` | *(none)* | Optional continuation tensor-payload ceiling per request |
| `--qwen35-max-process-bytes <N>` | `OPENKIND_QWEN35_MAX_PROCESS_BYTES` | *(none)* | Optional process-memory admission ceiling |
| `--api-key <TOKEN>` | `OPENKIND_API_KEY` | *(none)* | Optional bearer token required for `/v1/*` and gRPC |
| `--log-filter <FILTER>` | `RUST_LOG` | `info` | Tracing filter (e.g. `info`, `debug`, `openkind=trace`) |

### Optional API keys

Authentication is off when no key is configured. `--api-key <TOKEN>` or
`OPENKIND_API_KEY` enables checking on `/v1/*` and gRPC. `/health` and
`/metrics` stay public. Clients must send the configured key as a bearer token.

Generate a key with the CLI and supply it before starting the daemon:

```bash
export OPENKIND_API_KEY="$(openkind keygen)"
openkindd --http-addr 127.0.0.1:18080 --grpc-addr 0 --models mock
```

Configured keys must be nonempty visible ASCII without whitespace. Invalid
values fail startup without printing the secret. Keys are not persisted.
The [CLI guide](../openkind-cli/README.md#keygen) covers generation and
PowerShell; the [environment reference](../../docs/ENV.md) covers aliases and
conflicting settings.

## Example

```bash
# Start openkindd locally with mock backends
cargo run -p openkind-server -- \
    --http-addr 127.0.0.1:18080 \
    --grpc-addr 127.0.0.1:19090 \
    --models mock,jev-latest

# Query health
curl http://127.0.0.1:18080/health

# Evaluate a request
curl -sS -H 'content-type: application/json' \
     -X POST http://127.0.0.1:18080/v1/systemone \
     -d @examples/01_noul.json | jq
```

Native registration is opt-in and fail-closed. Add a Qwen alias to `--models`
and supply all three offline artifact paths. Choice requests must reserve the
literal `__none__` option for native semantic-none mass. Choice and Score
confidence use normalized entropy; internal review policy is not serialized as
semantic none. When a process-memory ceiling is configured, the engine refreshes
the loaded process high-water mark before each evaluation and conservatively
splits remaining headroom across the configured concurrency limit.

## Testing

```bash
cargo test -p openkind-server
```
