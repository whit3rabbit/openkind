# AGENTS.md — opendecision-client

> LLM developer guide for `opendecision-client`. Read this before modifying client behavior, retry logic, or error mapping.

## Crate Purpose & Boundaries

`opendecision-client` is the async Rust SDK for the SystemOne HTTP API. It is the Rust counterpart of the `typesafe_sdk` Python SDK: same endpoints, same retry taxonomy, same error envelope handling, same environment-variable resolution — usable against both a local `opendecisiond` daemon and `https://api.typesafe.ai`.

### Dependency Rules

- **Runtime dependency on `opendecision-core` only** (plus tokio/reqwest/serde stack). The client must never depend on `opendecision-api`, `opendecision-engine`, `opendecision-server`, or `opendecision-cli`.
- `opendecision-api` and `opendecision-engine` appear **only in `[dev-dependencies]`**, used by the integration tests to boot a real server in-process.
- Orphan-rule consequence: `impl From<&str> for State` is impossible (both foreign), which is why [`IntoState`](src/client/options.rs) exists as a local conversion trait. Do not replace it with serialization tricks.

## Module Map

| Module | Responsibility |
|---|---|
| [`src/client/`](src/client/) | Modular client implementation: `mod.rs` (re-exports), [`options.rs`](src/client/options.rs) (`RequestOptions`, `IntoState`), [`builder.rs`](src/client/builder.rs) (`ClientBuilder`), [`core.rs`](src/client/core.rs) (`Client`, `Health`), [`transport.rs`](src/client/transport.rs) (retry send loop in `send_json`), and [`tests.rs`](src/client/tests.rs) |
| [`src/error/`](src/error/) | Modular error taxonomy: `mod.rs` (re-exports, `Error` enum), [`api_error.rs`](src/error/api_error.rs) (`ApiError`, `ApiErrorKind`), [`envelope.rs`](src/error/envelope.rs) (lenient error-envelope extraction), [`retry_after.rs`](src/error/retry_after.rs) (`parse_retry_after`), and [`tests.rs`](src/error/tests.rs) |
| [`src/retry/`](src/retry/) | Modular retry policy: `mod.rs` (re-exports), [`policy.rs`](src/retry/policy.rs) (`RetryPolicy`, backoff computation, jitter), and [`tests.rs`](src/retry/tests.rs) |
| [`src/question.rs`](src/question.rs) | Ergonomic `Question` / `State` constructors mirroring the Python SDK's `Noul`/`Choice`/`Score` sugar |

## Critical Invariants

1. **Python SDK parity for defaults**: `RetryPolicy::default()` must keep matching tenacity defaults in the Python SDK — 2 retries, 0.5s initial, 5s max backoff, 0.25 jitter, retry `{408, 429, 5xx}` (529 included via 5xx), 30s total budget. `tests::defaults_match_python_sdk` pins this.
2. **`retry-after-ms` precedence**: when both headers are present, `retry-after-ms` (milliseconds) wins over `Retry-After` (seconds/HTTP-date). The opendecision server emits both on 429/529.
3. **Retry counting**: `retries` is the number of completed retries; the `x-typesafe-retry-count` header carries this value (first retry = `1`, mirroring the Python SDK) and the initial attempt sends no header.
4. **Budget semantics**: mirror tenacity `stop_before_delay` — if `elapsed + next_delay >= total_timeout`, return the last error instead of sleeping. Do not start a retry that cannot finish.
5. **Never retry non-transient failures**: 4xx (except 408/429), decode failures, and config errors must surface immediately. `RetryPolicy::is_retryable` is the single decision point.
6. **Wire types come from `opendecision-core`** — re-exported, never redefined here. Any schema change flows from the core crate automatically.
7. **Server compatibility is pinned by `crates/opendecision-api/tests/sdk_compat.rs`** (server-side) and `crates/opendecision-client/tests/live_server.rs` (client-side). If either suite breaks, you have introduced a client-facing regression — fix the cause, do not loosen the assertion.

## Testing Mandate

```bash
cargo test -p opendecision-client
```

- `tests/sdk_parity_retry.rs` (modularized into [`tests/sdk_parity_retry/`](tests/sdk_parity_retry/): `status.rs`, `overrides.rs`, `delays.rs`, `exhaustion.rs`, and `concurrency.rs`), `tests/sdk_parity_errors.rs`, `tests/sdk_parity_wire.rs`, `tests/sdk_parity_config.rs` — ports of the TypeSafe Python SDK's test suite; [`PARITY.md`](PARITY.md) is the authoritative file-by-file mapping (ported / covered / N/A / divergence). Keep it in sync when porting new tests.
- `tests/common/mod.rs` — the `MockTransport`-style stub: per-request handler closure plus a captured-request handle (`x-call` marker, `x-typesafe-retry-count`, parsed body, headers).
- `tests/live_server.rs` — real `opendecision-api` server over TCP (`into_make_service_with_connect_info` is required for the per-IP rate limiter to fire). Covers wire conformance for all three question types, auth, 404/422 error envelopes, and the real rate limiter.
- `tests/retry_behavior.rs` — deterministic stub server for attempt counting, retry-count headers, `Retry-After` precedence, budget stops, connection classification, per-call overrides.
- Timing-sensitive tests must use millisecond-scale delays and generous wall-clock assertions; never assert exact sleeps (the Python suite asserts mocked sleep arguments; the ports assert elapsed bounds instead).

## Behavior Contracts Pinned by the Parity Suite

Do not change these without updating the Python SDK mapping (`PARITY.md`) and the TypeSafe SDK reference:

- `RetryPolicy::default()` equals the Python tenacity defaults (2 retries, 0.5s→5s, 0.25 jitter, `{408, 429} ∪ 5xx`, 30s budget).
- Empty `Retry-After` parses as zero; past HTTP-dates clamp to zero; `retry-after-ms` invalid values fall through to `Retry-After`; overflow parses as absent.
- Protected headers (`authorization`, `accept`, `user-agent`, `x-typesafe-sdk`, `x-typesafe-runtime`, `x-typesafe-retry-count`, and `content-type` on bodied requests) cannot be overridden by user defaults or per-call extras; other per-call extras replace same-name defaults.
- Retry counting: initial attempt sends no header; retry N sends `x-typesafe-retry-count: N`.
- The last HTTP response (not the first) is the error surfaced after exhausted retries.
- Unknown answer `"type"` tags fail decoding (strict divergence from the Python SDK's silent drop).

## Verification

Run the full workspace battery from the root before merging (see root `AGENTS.md`):

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p opendecision-gen-schemas -- --write
```
