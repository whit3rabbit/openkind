# AGENTS.md — openkind-client

> LLM developer guide for `openkind-client`. Read this before modifying client behavior, retry logic, or error mapping.

## Crate Purpose & Boundaries

`openkind-client` is the async Rust SDK for the SystemOne HTTP API. It is the Rust counterpart of the `typesafe_sdk` Python SDK: same endpoints, same retry taxonomy, same error envelope handling, same environment-variable resolution — usable against a local `openkindd` daemon, `https://api.typesafe.ai`, and supported providers (OpenRouter System One and Cloudflare Workers AI via [`ClientBuilder`](src/client/builder.rs)).

### Dependency Rules

- **Runtime dependency on `openkind-core` only** (plus tokio/reqwest/serde stack). The client must never depend on `openkind-api`, `openkind-engine`, `openkind-server`, or `openkind-cli`.
- `openkind-api` and `openkind-engine` appear **only in `[dev-dependencies]`**, used by the integration tests to boot a real server in-process.
- Orphan-rule consequence: `impl From<&str> for State` is impossible (both types foreign to this crate), which is why [`IntoState`](src/client/options.rs) exists as a local conversion trait. Do not replace it with serialization tricks.

## Module Map

| Module | Responsibility |
|---|---|
| [`src/client/`](src/client/) | Modular client implementation: `mod.rs` (re-exports), [`options.rs`](src/client/options.rs) (`RequestOptions`, `IntoState`), [`builder.rs`](src/client/builder.rs) (`ClientBuilder`, provider presets), [`core.rs`](src/client/core.rs) (`Client`, `Health`, request-bound response validation), [`http1.rs`](src/client/http1.rs) (socket-level HTTP/1.1 transport, chunked stream splitting, connection reuse), [`transport.rs`](src/client/transport.rs) (retry send loop in `send_json`), and [`tests.rs`](src/client/tests.rs) |
| [`src/error/`](src/error/) | Modular error taxonomy: `mod.rs` (re-exports, `Error` enum including `InvalidResponse`), [`api_error.rs`](src/error/api_error.rs) (`ApiError`, `ApiErrorKind`), [`envelope.rs`](src/error/envelope.rs) (lenient error-envelope extraction), [`retry_after.rs`](src/error/retry_after.rs) (`parse_retry_after`), and [`tests.rs`](src/error/tests.rs) |
| [`src/retry/`](src/retry/) | Modular retry policy: `mod.rs` (re-exports), [`policy.rs`](src/retry/policy.rs) (`RetryPolicy`, backoff computation, jitter), and [`tests.rs`](src/retry/tests.rs) |
| [`src/question.rs`](src/question.rs) | Ergonomic `Question` / `State` constructors mirroring the Python SDK's `Noul`/`Choice`/`Score` sugar |

## Critical Invariants

1. **Python SDK Parity for Defaults**:
   `RetryPolicy::default()` must match tenacity defaults in the Python SDK: 2 retries, 0.5s initial backoff, 5s max backoff, 0.25 jitter, retrying `{408, 429, 5xx}` (529 included via 5xx), 30s total budget. Pinned by `tests::defaults_match_python_sdk`.
2. **`retry-after-ms` Precedence**:
   When both headers are present, `retry-after-ms` (milliseconds) wins over `Retry-After` (seconds/HTTP-date). The openkind server emits both on 429/529.
3. **Retry Counting**:
   `retries` counts completed retries; the `x-typesafe-retry-count` header carries this value (first retry = `1`, mirroring Python SDK) and the initial attempt sends no header.
4. **Budget Semantics**:
   Mirror tenacity `stop_before_delay` — if `elapsed + next_delay >= total_timeout`, return the last error immediately instead of sleeping. Do not start a retry that cannot complete.
5. **Never Retry Non-Transient Failures**:
   4xx errors (except 408/429), deserialization errors, and configuration failures must surface immediately without retrying. `RetryPolicy::is_retryable` decides transport and API retries. A decoded 2xx response that fails request-bound validation becomes nonretryable `Error::InvalidResponse` after the send loop.
6. **Wire Types from Core**:
   Wire types come strictly from `openkind-core` — re-exported, never redefined here.

## Critical Gotchas & Pitfalls

1. **Rust Orphan Rule & `IntoState`**:
   Because both `&str` and `openkind_core::State` are foreign types, standard `From`/`Into` cannot be implemented in `openkind-client`. Always use the [`IntoState`](src/client/options.rs) trait for ergonomic conversion.
2. **Protected Header Precedence**:
   Protected headers (`authorization`, `accept`, `user-agent`, `x-typesafe-sdk`, `x-typesafe-runtime`, `x-typesafe-retry-count`, `content-type`) cannot be overridden by user defaults or per-call options.
3. **Strict Answer Tag Decoding**:
   Unknown answer `"type"` tags fail decoding with an error (intentional strict divergence from Python SDK's silent drop). Known tags must still match the originating question type, and Choice options must match the caller's criteria.
4. **Timing-Sensitive Tests**:
   Integration tests must use millisecond-scale delays and generous wall-clock bounds; never assert exact sleep durations.

## Testing Mandate & Parity Mapping

- `tests/sdk_parity_retry.rs` (modularized into `tests/sdk_parity_retry/`): `status.rs`, `overrides.rs`, `delays.rs`, `exhaustion.rs`, and `concurrency.rs`.
- `tests/sdk_parity_errors.rs`, `tests/sdk_parity_wire.rs`, `tests/sdk_parity_config.rs`: Ports of the TypeSafe Python SDK's test suite; [`PARITY.md`](PARITY.md) is the authoritative file-by-file mapping.
- `tests/live_server.rs`: Real `openkind-api` server over TCP verifying wire conformance across all question types, auth, 404/422 envelopes, and the real rate limiter.
- `tests/provider_surfaces.rs`: Provider routing checks for OpenRouter metadata/cost handling and Cloudflare Workers AI endpoints.
- `tests/retry_behavior.rs`: Deterministic stub server for attempt counting, retry headers, precedence, and budget stops.
- `tests/response_validation.rs`: Stub server checks for valid 2xx responses and nonretryable invalid responses.
- `benches/client.rs`: Criterion benchmarks over a warmed localhost connection
  to the authenticated Axum/MockEngine stack, with retries disabled.

## Verification Commands

```bash
cargo check -p openkind-client
cargo test -p openkind-client
cargo test -p openkind-client --bench client
cargo bench -p openkind-client --bench client -- --noplot
```
