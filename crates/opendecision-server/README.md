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

## Testing

```bash
cargo test -p opendecision-server
```
