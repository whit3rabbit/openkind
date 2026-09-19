# AGENTS.md — opendecision-engine

> LLM developer guide for `opendecision-engine`. Read this before modifying the engine trait, registry, or mock engine.

## Crate Purpose & Boundaries

`opendecision-engine` sits between transport layers (`opendecision-api`) and model execution. It defines:
- The `DecisionEngine` trait that all current and future model backends implement.
- `EngineRegistry`: Model alias mapping and dispatch router.
- `dispatch()`: The unified orchestration pipeline (validation, token usage calculation, telemetry).
- `MockEngine`: Deterministic fake engine for testing without loading neural weights.

### Critical Invariants

1. **Runtime Agnostic**:
   - `DecisionEngine` does not know whether requests arrived via HTTP or gRPC. It strictly accepts `opendecision_core::SystemRequest` and returns `opendecision_core::SystemResponse`.
   - Never import HTTP or gRPC transport types (`axum`, `tonic`, headers, status codes) into `opendecision-engine`.
2. **Deterministic Registry Resolution**:
   - `EngineRegistry::models()` must always return model alias keys in **sorted order**.
   - `EngineRegistry::list_models()` must override the internal backend's `ModelInfo.name` with the **registered alias** (the client-facing identifier).
3. **Usage Accounting Guarantee**:
   - `dispatch()` guarantees that `resp.usage.input_tokens` is populated (fallback to `engine.estimate_input_tokens`) and `resp.usage.output_tokens` is computed via `estimate_output_tokens`.

## Key Files & Types

- [`src/lib.rs`](./src/lib.rs):
  - `pub trait DecisionEngine: Send + Sync`:
    - `backend_id(&self) -> &str`
    - `model_metadata(&self) -> ModelInfo`
    - `async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse>`
    - `fn estimate_input_tokens(&self, req: &SystemRequest) -> u32`
  - `pub struct EngineRegistry`: Stores `HashMap<String, Arc<dyn DecisionEngine>>`.
  - `pub async fn dispatch(req, registry) -> EngineResult<SystemResponse>`: Emits telemetry metrics `opendecision_requests_total`, `opendecision_responses_total`, and `opendecision_request_duration_ms`.
  - `enum EngineError`:
    - `Invalid(ValidationError)` (mapped to HTTP 422 by API layer)
    - `UnknownModel(String)` (mapped to HTTP 404)
    - `Backend { backend, message }` (mapped to HTTP 500)
- [`src/mock.rs`](./src/mock.rs):
  - `MockEngine`: Seeds RNG using `(question_id, instructions)` hash to guarantee determinism across requests.
  - Generates valid distributions over choices and rubrics.

## How to Add a New Backend (Phase 2)

1. Implement `DecisionEngine` for your backend struct in `crates/opendecision-backends`.
2. Ensure `backend_id()` returns a unique identifier (e.g. `"qwen-3.5-4b-candle"`).
3. Override `model_metadata()` with release date and description.
4. Wire it into the `EngineRegistry` on startup in `crates/opendecision-server/src/main.rs`.
5. Write roundtrip tests in `opendecision-engine` and `crates/opendecision-api/tests/sdk_compat.rs`.

## Verification Commands

```bash
cargo test -p opendecision-engine
```
