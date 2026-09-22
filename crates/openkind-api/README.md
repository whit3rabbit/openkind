# openkind-api

> Dual HTTP (axum) and gRPC (tonic) protocol layer for `openkind`.

`openkind-api` implements the network endpoints that mirror the hosted TypeSafe Jev API and Python SDK client expectations. It handles routing, middleware, authentication, request-id propagation, error envelopes, and Prometheus metrics.

## Supported Protocols & Endpoints

### HTTP (axum 0.8)
- `POST /v1/systemone` (aliased to `/v1/system_one`): Evaluates a `SystemRequest` and returns a `SystemResponse`.
- `GET /v1/models`: Returns `{ "models": [ ... ] }` listing available models, their descriptions, and release dates.
- `GET /health`: Liveness probe (open, no auth required).
- `GET /metrics`: Prometheus text scrape endpoint (open, no auth required).

### OpenAPI Specification
- Canonical OpenAPI 3.1.0 specification: [`openapi.yaml`](./openapi.yaml) (also referenced at `docs/openapi.yaml`).

### gRPC (tonic 0.14)
- `openkind.system_one.SystemOne/Evaluate`: High-performance binary RPC equivalent of `POST /v1/systemone`.

## Middleware & Cross-Cutting Behaviors

1. **Request ID Tracking (`x-typesafe-request-id`)**:
   - Outermost layer stamps every response (including 401s, 4xx, and 5xx errors) with a unique UUIDv4.
   - Honors inbound `x-typesafe-request-id` headers when supplied by proxies.
2. **Bearer Token Authentication**:
   - Optional auth layer configured via `AuthConfig`, `OPENKIND_API_KEY`, or `TYPESAFE_API_KEY`.
   - Protects `/v1/*` routes with constant-time token comparison.
   - `/health` and `/metrics` remain open for scrapers and orchestrators.
3. **Error Taxonomy**:
   Maps internal errors into standard TypeSafe HTTP responses with structured JSON envelopes:
   - `400 Bad Request` (`bad_json`)
   - `401 Unauthorized` (`unauthorized`, includes `WWW-Authenticate: Bearer`)
   - `404 Not Found` (`unknown_model`)
   - `422 Unprocessable Entity` (`invalid_body`)
   - `429 Too Many Requests` (`rate_limited`, includes `Retry-After` and `retry-after-ms`)
   - `529 Overloaded` (`overloaded`, includes `Retry-After` and `retry-after-ms`)
   - `500 Internal Server Error` (`internal_error` / `backend_error`)

## Testing

```bash
cargo test -p openkind-api
```

This includes the 56-test `sdk_compat` suite verifying wire and header compatibility with the TypeSafe Python SDK, and gRPC roundtrip integration tests.

