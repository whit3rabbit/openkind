# openpick-server

> The `openpickd` production inference daemon.

`openpick-server` provides the `openpickd` daemon binary that powers self-hosted `openpick` deployments. It sets up structured tracing, initializes the `EngineRegistry`, mounts both HTTP (axum) and gRPC (tonic) listeners, registers Prometheus metrics, and manages graceful shutdown on `SIGINT` or `SIGTERM`.

## Installation & Build

Build the daemon binary:

```bash
cargo build --release -p openpick-server
# Binary produced at target/release/openpickd
```

## Command-Line Usage

```bash
openpickd [OPTIONS]
```

### Options

| Flag | Environment Variable | Default | Description |
|---|---|---|---|
| `--http-addr <ADDR>` | `OPENPICK_HTTP_ADDR` | `0.0.0.0:8080` | Address to bind HTTP server on |
| `--grpc-addr <ADDR>` | `OPENPICK_GRPC_ADDR` | `0.0.0.0:9090` | Address to bind gRPC server on (set port 0 to disable) |
| `--models <ALIASES>` | `OPENPICK_MODELS` | `mock,jev-latest` | Comma-separated list of model aliases to register |
| `--api-key <TOKEN>` | `OPENPICK_API_KEY` | *(none)* | Optional bearer token required for `/v1/*` routes |
| `--log-filter <FILTER>` | `RUST_LOG` | `info` | Tracing filter (e.g. `info`, `debug`, `openpick=trace`) |

## Example

```bash
# Start openpickd locally with mock backends
cargo run -p openpickd -- \
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
cargo test -p openpick-server
```
