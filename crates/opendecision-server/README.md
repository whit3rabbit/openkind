# opendecision-server

> The `opendecisiond` production inference daemon.

`opendecision-server` provides the `opendecisiond` daemon binary that powers self-hosted `opendecision` deployments. It sets up structured tracing, initializes the `EngineRegistry`, mounts both HTTP (axum) and gRPC (tonic) listeners, registers Prometheus metrics, and manages graceful shutdown on `SIGINT` or `SIGTERM`.

## Installation & Build

Build the daemon binary:

```bash
cargo build --release -p opendecision-server
# Binary produced at target/release/opendecisiond
```

## Command-Line Usage

```bash
opendecisiond [OPTIONS]
```

### Options

| Flag | Environment Variable | Default | Description |
|---|---|---|---|
| `--http-addr <ADDR>` | `OPENDECISION_HTTP_ADDR` | `0.0.0.0:8080` | Address to bind HTTP server on |
| `--grpc-addr <ADDR>` | `OPENDECISION_GRPC_ADDR` | `0.0.0.0:9090` | Address to bind gRPC server on (set port 0 to disable) |
| `--models <ALIASES>` | `OPENDECISION_MODELS` | `mock,jev-latest` | Comma-separated list of model aliases to register |
| `--qwen35-aliases <ALIASES>` | `OPENDECISION_QWEN35_ALIASES` | `qwen35-native` | Aliases in `--models` that use the native Qwen engine |
| `--qwen35-bundle-root <PATH>` | `OPENDECISION_QWEN35_BUNDLE_ROOT` | *(none)* | Offline selected-profile bundle root required by native aliases |
| `--qwen35-checkpoint-root <PATH>` | `OPENDECISION_QWEN35_CHECKPOINT_ROOT` | *(none)* | Offline pinned checkpoint root required by native aliases |
| `--qwen35-tokenizer <PATH>` | `OPENDECISION_QWEN35_TOKENIZER` | *(none)* | Digest-locked tokenizer JSON required by native aliases |
| `--qwen35-concurrency <N>` | `OPENDECISION_QWEN35_CONCURRENCY` | `1` | Maximum concurrent native evaluations |
| `--qwen35-queue <N>` | `OPENDECISION_QWEN35_QUEUE` | `2` | Additional native requests allowed to wait |
| `--qwen35-max-tensor-bytes <N>` | `OPENDECISION_QWEN35_MAX_TENSOR_BYTES` | *(none)* | Optional continuation tensor-payload ceiling per request |
| `--qwen35-max-process-bytes <N>` | `OPENDECISION_QWEN35_MAX_PROCESS_BYTES` | *(none)* | Optional process-memory admission ceiling |
| `--api-key <TOKEN>` | `OPENDECISION_API_KEY` | *(none)* | Optional bearer token required for `/v1/*` routes |
| `--log-filter <FILTER>` | `RUST_LOG` | `info` | Tracing filter (e.g. `info`, `debug`, `opendecision=trace`) |

## Example

```bash
# Start opendecisiond locally with mock backends
cargo run -p opendecisiond -- \
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
cargo test -p opendecision-server
```
