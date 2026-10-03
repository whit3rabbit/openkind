# openkind-api

> Dual HTTP (axum) and gRPC (tonic) protocol layer for `openkind`.

`openkind-api` implements the network endpoints that mirror the hosted TypeSafe Jev API and Python SDK client expectations. It handles routing, middleware, authentication, request-id propagation, rate limiting, payload limits, error envelopes, and Prometheus metrics. Both transports call `openkind_engine::dispatch` and return typed decisions; neither generates text.

This crate is a workspace library with no binary. The [`openkindd` daemon](../openkind-server/README.md) mounts it.

## Quickstart

Run the daemon (default models include `mock`; HTTP binds `0.0.0.0:8080`), then probe it:

```bash
cargo install --locked openkind-server
openkindd
curl -s http://127.0.0.1:8080/health
# {"status":"ok"}
```

A first evaluation, no API key required unless one is configured:

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "I was charged twice. Please help.",
    "model": "mock",
    "questions": {
      "billing": { "type": "noul", "instructions": "Is this about billing?" }
    }
  }'
```

The response reports `answers.billing.noul` as a probability in `[0, 1]`.

## HTTP endpoints (axum 0.8)

| Route | Purpose |
|---|---|
| `POST /v1/systemone` (aliased to `/v1/system_one`) | Evaluates a `SystemRequest` and returns a `SystemResponse`. |
| `GET /v1/models` | Lists models with name, description, and release date. |
| `GET /health` | Liveness probe returning `{"status":"ok"}`. Open, no auth. |
| `GET /metrics` | Prometheus text-format scrape. Open, no auth. |
| `GET /playground` | Embedded web UI. Opt-in: `openkindd --playground on`. |
| `POST /v1/arrow` | Unofficial bulk Arrow IPC endpoint, outside the TypeSafe wire contract. Opt-in: `openkindd --arrow on`; still behind the `/v1` auth gate and rate limiter. |

`/playground` and `/v1/arrow` are excluded from the OpenAPI specification and SDK parity tests by design. See [`docs/ARROW.md`](../../docs/ARROW.md) for the Arrow contract.

## gRPC (tonic 0.14)

- `openkind.SystemOne/Evaluate` is the binary RPC equivalent of `POST /v1/systemone`.
- Bearer credentials, or `x-api-key` metadata, are checked per call when an API key is configured. Missing or invalid credentials return `UNAUTHENTICATED` (HTTP 401 equivalent).
- Engine and validation failures map symmetrically to HTTP: `INVALID_ARGUMENT` to 422, `NOT_FOUND` to 404, `UNAVAILABLE` to 529 overloaded, `DEADLINE_EXCEEDED` to 504, and `INTERNAL` to 500.
- Every response and error status carries `x-typesafe-request-id` metadata, matching the HTTP header.

## Middleware and cross-cutting behavior

1. **Request ID tracking (`x-typesafe-request-id`)**: the outermost layer stamps every response, including 401s, 429s, and 5xx errors. An inbound header from a proxy is honored only if it is non-empty, at most 128 characters, and limited to ASCII letters, digits, `.`, `-`, and `_`; otherwise a fresh UUIDv4 is minted.
2. **Bearer token authentication**: enabled via `OPENKIND_API_KEY` or `TYPESAFE_API_KEY` (deprecated `OPENDECISION_API_KEY` and `OPENPICK_API_KEY` fallbacks). `/v1/*` routes are gated; `/health`, `/metrics`, and the playground HTML shell stay open for probes and scrapers. Tokens are compared as SHA-256 digests in constant time.
3. **Rate limiting**: a per-IP fixed-window limiter emits 429 with `Retry-After` and `retry-after-ms`. The daemon enables it by default at 120 requests per minute. Set `--rate-limit-rpm 0` to disable it.
4. **Payload limit**: request bodies are capped at 16 MB by default, returning 413 `payload_too_large` beyond that.
5. **Tracing**: `TraceLayer` spans exclude request headers, so credentials and caller-supplied request IDs never enter telemetry.
6. **Proxy-cache headers**: in daemon proxy mode, proxied `/v1/systemone` responses carry `x-openkind-cache` (and optionally `x-openkind-cache-detail`).

## Error taxonomy

Internal errors map to TypeSafe JSON envelopes of the form `{"error":{"code":...,"message":...}}`:

- `400 Bad Request` (`bad_json`)
- `401 Unauthorized` (`unauthorized`, includes `WWW-Authenticate: Bearer`)
- `404 Not Found` (`unknown_model`)
- `413 Payload Too Large` (`payload_too_large`)
- `422 Unprocessable Entity` (`invalid_body`)
- `429 Too Many Requests` (`rate_limited`) and `529 Overloaded` (`overloaded`), both with `Retry-After` and `retry-after-ms`
- `500 Internal Server Error` (`internal_error` / `backend_error`)
- `502 Bad Gateway` (`bad_gateway`)
- `504 Gateway Timeout` (`deadline_exceeded`)

## OpenAPI specification

[`openapi.yaml`](./openapi.yaml) is the canonical OpenAPI 3.1.0 specification for the wire HTTP endpoints, mirrored at `docs/openapi.yaml` in the repository root.

## Testing

```bash
cargo test -p openkind-api
```

- The 61-test `sdk_compat` suite pins wire and header compatibility with the TypeSafe Python SDK: `cargo test -p openkind-api --test sdk_compat`.
- gRPC roundtrip integration tests: `cargo test -p openkind-api --test grpc_roundtrip`.
- Arrow endpoint tests: `cargo test -p openkind-api --lib arrow`.
- Playground UI regressions (Node 18+): `node --test crates/openkind-api/tests/playground.test.cjs` from the repository root.

## License

MIT — see [../../LICENSE](../../LICENSE). The workspace Cargo manifest declares `MIT OR Apache-2.0`.
