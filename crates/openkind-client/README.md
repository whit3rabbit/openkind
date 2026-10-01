# openkind-client

Async Rust client for the `openkindd` daemon and Jev's TypeSafe System One HTTP API. The common request subset also works through OpenRouter's System One endpoint and Cloudflare Workers AI. See the [field-by-field compatibility matrix](../../docs/JEV_COMPATIBILITY.md) for provider differences and current limits.

Built on the shared `openkind-core` wire types (`SystemRequest`, `SystemResponse`, `Question`, `Answer`, ...), so request/response conformance with `jev-v1-request.json` / `jev-v1-response.json` is structural, not hand-maintained. Retry, rate-limit, and error conventions mirror the `typesafe_sdk` Python SDK.

Successful evaluation responses are also checked against the submitted request: answer IDs and types must match, and a `Choice` selection and its probability keys must come from that question's criteria. A model decision does not authorize the caller's next action.

Answer shapes mirror their questions. A `noul` answer carries a single probability and has no confidence field, while `choice` and `score` answers carry `confidence` plus per-key `probabilities`. When the target is a local `openkindd` daemon running a native model profile, `Choice` questions must also include a non-empty `__none__` option — a backend requirement, not a global Jev rule.

## Installation

This is a library crate. For an application beside the cloned `openkind` directory, add it as a path dependency:

```toml
[dependencies]
openkind-client = { path = "../openkind/crates/openkind-client" }
```

## Quick start

```rust
use openkind_client::{question, Client};

#[tokio::main]
async fn main() -> Result<(), openkind_client::Error> {
    let client = Client::builder()
        .api_key("dev-key")
        .base_url("http://127.0.0.1:18080") // default: https://api.typesafe.ai
        .build()?;

    let response = client
        .system_one(
            "I was charged twice. Please help.",
            [
                ("billing", question::noul("Is this about billing?")),
                ("tone", question::choice("What is the tone?", [("calm", None), ("angry", None)])),
                ("severity", question::score("How severe?", ["low", "medium", "high"])),
            ],
        )
        .await?;

    println!("answers: {:?}", response.answers);
    Ok(())
}
```

For full control, build an `openkind_client::SystemRequest` and call `client.evaluate(request)`.

## Provider setup

```rust
use openkind_client::Client;

# fn clients() -> Result<(), openkind_client::Error> {
let openrouter = Client::builder()
    .api_key("openrouter-key")
    .base_url("https://openrouter.ai/api")
    .default_model("jev-1.13")
    .build()?;

let cloudflare = Client::builder()
    .api_key("cloudflare-api-token")
    .cloudflare_account("accountid")
    .build()?;
# let _ = (openrouter, cloudflare);
# Ok(())
# }
```

Both use `system_one` or `evaluate`. OpenRouter requires the `/api` base URL,
since the client appends `/v1/systemone`. It adds `id`, `provider`, and
`usage.cost` to responses. The current Rust `SystemResponse` decodes the
decisions but discards those extra fields.

OpenRouter's `/api/v1/models` has a
different response shape, so `list_models` is not supported there, and
`/health` is an OpenKind extension that OpenRouter does not expose.

Cloudflare mode sends `{ "model": "typesafe/jev", "input": { "state": ..., "questions": ... } }`
to the account's `/ai/run` endpoint, then reads the Jev response from `result`.
A `success: false` envelope is rejected even when the embedded result would
otherwise decode.

`list_models` and `health` are unavailable in this mode. An
explicit `base_url` can override the Cloudflare host for a proxy or local test
server. It must end at the account prefix because the client appends `/ai/run`.

## Rate limiting, overload, and retries

The server rejects excess traffic with `429 rate_limited` and reports saturation with `529 overloaded`. Both carry `retry-after-ms` (preferred) and `Retry-After` headers. The client:

- waits exactly as long as the server asked (HTTP-date `Retry-After` values are also parsed).
- falls back to exponential backoff with subtractive jitter for other transient failures (`408`, `5xx`, connection errors, per-attempt timeouts, and `429`/`529` responses without a delay header).
- stops before a retry that would exceed the call's total time budget, returning the last error.

Defaults mirror the Python SDK's `RetryPolicy`: 2 retries, 0.5s→5s backoff, 25% jitter, 30s total budget. Retried requests send `x-typesafe-retry-count`. Tune or disable:

```rust
use std::time::Duration;
use openkind_client::{Client, RetryPolicy};

let client = Client::builder()
    .api_key("dev-key")
    .retry(
        RetryPolicy::new()
            .max_retries(5)
            .backoff_initial(Duration::from_millis(100))
            .backoff_max(Duration::from_secs(2))
            .total_timeout(Some(Duration::from_secs(15))),
    )
    .build()?;
```

Pass `RetryPolicy::new().max_retries(0)` to disable retries, or override per call via `RequestOptions` with the `*_with` method variants.

## Errors

All failures surface as `openkind_client::Error`:

| Variant | Meaning |
|---|---|
| `Error::Api(ApiError)` | Non-success response: `status`, `code` (`rate_limited`, `unknown_model`, ...), `message`, `request_id` (from `x-typesafe-request-id`), `retry_after` |
| `Error::Connection` | Could not reach / read from the server |
| `Error::Timeout` | Per-attempt timeout exceeded |
| `Error::ResponseTooLarge` | Response exceeded the fixed 8 MiB body limit |
| `Error::Decode` | 2xx body did not match the expected wire type |
| `Error::InvalidResponse` | 2xx body decoded but violated the submitted question contract. Never retried |
| `Error::Config` | Invalid client configuration (missing API key, bad URL, ...) |

Successful and error response bodies are both limited to 8 MiB
(`MAX_RESPONSE_BODY_SIZE`). The limit is enforced while streaming the body, so
it also applies when `Content-Length` is absent or inaccurate.

`ApiError::kind()` classifies by status into the Python SDK taxonomy: `BadRequest` (400), `Authentication` (401), `PermissionDenied` (403), `NotFound` (404), `PayloadTooLarge` (413), `UnprocessableEntity` (422), `RateLimit` (429), `Overloaded` (529), `InternalServer` (5xx), and `Other` for anything else.

```rust
match client.evaluate(request).await {
    Ok(response) => { /* ... */ }
    Err(err) if err.kind() == Some(openkind_client::ApiErrorKind::RateLimit) => {
        eprintln!("back off for {:?}", err.retry_after().unwrap());
    }
    Err(err) => {
        eprintln!("failed: {err} (request_id={:?})", err.request_id());
    }
}
```

## Environment variables

| Builder option | Primary | Python-SDK fallback |
|---|---|---|
| `api_key` | `OPENKIND_API_KEY` | `TYPESAFE_API_KEY` |
| `base_url` | `OPENKIND_BASE_URL` | `TYPESAFE_BASE_URL` |
| `default_model` | `OPENKIND_DEFAULT_MODEL` | `TYPESAFE_DEFAULT_MODEL` |

Explicit builder values win, and empty/whitespace env values are ignored.

The API
key is required and must be printable ASCII without whitespace. Values are
trimmed, and an explicit builder key is used as-is even when invalid rather
than falling back to the environment.

Cloudflare mode uses its account URL and `typesafe/jev` unless `base_url` or
`default_model` is set explicitly on the builder (its environment fallbacks do
not apply).

## Endpoints

| Method | Client API | Path |
|---|---|---|
| `POST` | `evaluate` / `evaluate_with` / `system_one` / `system_one_with` | `/v1/systemone` |
| `POST` in Cloudflare mode | Same evaluation methods | `/ai/run` |
| `GET` | `list_models` / `list_models_with` | `/v1/models` |
| `GET` | `health` / `health_with` | `/health` (unauthenticated, OpenKind daemon extension) |

Per-call overrides (`timeout`, `retry`, extra headers) are available on every `*_with` variant through [`RequestOptions`](src/client/options.rs).

## Testing

```bash
cargo test -p openkind-client
```

- Unit tests: `Retry-After` parsing (ms/seconds/HTTP-date, incl. the Python SDK's edge matrix), backoff caps/jitter/extremes, error-envelope extraction, status classification, env-var precedence.
- `tests/sdk_parity_*.rs`: ports of the TypeSafe Python SDK's own test suite (`tests/test_retry.py`, `test_clients.py`, `test_errors.py`, `test_config.py`, `test_responses.py`, `test_questions.py`, `test_types.py`) — status matrices, attempt counts, retry-count header sequences, exact wire bodies, header-precedence rules. [PARITY.md](PARITY.md) maps every Python test file to its Rust counterpart.
- `tests/live_server.rs`: boots a real `openkind-api` server in-process (including the real rate limiter) and verifies the full client↔server contract — the client-side mirror of the API crate's `sdk_compat.rs` suite.
- `tests/provider_surfaces.rs`: provider routing checks for OpenRouter metadata handling and Cloudflare request wrapping, `result` unwrapping, and failure-envelope rejection.
- `tests/retry_behavior.rs`: deterministic stub server for retry mechanics (attempt counts, retry-count headers, `Retry-After` precedence, budget stops, connection errors).
- `tests/response_validation.rs`: request-bound answer validation against a stub server.

## Boundaries

- The only workspace crate in the runtime dependency set is `openkind-core`. The remaining runtime dependencies are the HTTP/async stack (reqwest, tokio, bytes) and support crates (serde, serde_json, thiserror, tracing, fastrand). Test-only dev-dependencies add `openkind-api`, `openkind-engine`, axum, and criterion.
- HTTP only. gRPC (`openkind.SystemOne/Evaluate`) is not covered.

## License

MIT — the Cargo manifest declares `MIT OR Apache-2.0`. See [../../LICENSE](../../LICENSE).
