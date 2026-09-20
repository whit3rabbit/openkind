# AGENTS.md — opendecision-engine

> LLM developer guide for `opendecision-engine`. Read this before modifying the engine trait, registry, or mock engine.

## Crate Purpose & Boundaries

`opendecision-engine` sits between transport layers (`opendecision-api`) and model execution. It defines:
- The `DecisionEngine` trait that all current and future model backends implement.
- The immutable `ModelExecutionProfile` contract for pinned model, renderer, readout, calibration, policy, and parity identities.
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

- [`src/lib.rs`](./src/lib.rs): Public API facade and module re-exports.
- [`src/engine.rs`](./src/engine.rs):
  - `pub trait DecisionEngine: Send + Sync`:
    - `backend_id(&self) -> &str`
    - `model_metadata(&self) -> ModelInfo`
    - `async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse>`
    - `fn estimate_input_tokens(&self, req: &SystemRequest) -> u32`
  - `estimate_output_tokens(answers)` calculation helper.
- [`src/registry.rs`](./src/registry.rs):
  - `pub struct EngineRegistry`: Thread-safe registry storing `HashMap<String, Arc<dyn DecisionEngine>>` with deterministic sorted model listing.
- [`src/dispatch.rs`](./src/dispatch.rs):
  - `pub async fn dispatch(req, registry) -> EngineResult<SystemResponse>`: Emits telemetry metrics `opendecision_requests_total`, `opendecision_responses_total`, and `opendecision_request_duration_ms`.
- [`src/error.rs`](./src/error.rs):
  - `enum EngineError`:
    - `Invalid(ValidationError)` (mapped to HTTP 422 by API layer)
    - `UnknownModel(String)` (mapped to HTTP 404)
    - `Backend { backend, message }` (mapped to HTTP 500)
  - `EngineResult<T>` type alias.
- [`src/mock.rs`](./src/mock.rs):
  - `MockEngine`: Seeds RNG using `(question_id, instructions)` hash to guarantee determinism across requests.
  - Generates valid distributions over choices and rubrics.
- [`src/profile.rs`](./src/profile.rs):
  - `ModelExecutionProfile`: Immutable provenance, execution semantics, and numerical parity contract.
  - This metadata does not alter `DecisionEngine` or expose a native model through the wire API.
- [`src/tests.rs`](./src/tests.rs): Comprehensive unit tests covering registry lookups, dispatch pipeline, and token usage accounting.

## Native Backend Integration Order

The fitted head/probability algebra, exact tokenizer/state-first rendering, and
correctness-first CPU backbone/continuation gates are complete for profile
`a047d6802c3f06f085b8`.

1. ~~Lift the Qwen-specific cached state into the backend-neutral
   `BranchableState` contract and prove fork/gather isolation.~~ Complete for
   the CPU path (Phase 3.4).
2. ~~Prove sequential nested `state → question → candidate` execution.~~
   Complete for the CPU path (Phase 3.5).
3. ~~Prove batched Q/K execution against the sequential baseline.~~ Complete
   for the CPU path (Phases 3.6/3.7).
4. ~~Measure the adaptive scheduler crossover on the named Mac.~~ Complete
   for the warm-process CPU path (Phase 3.8). High-cardinality stress and
   repeatability precede production promotion.
5. Only then implement `DecisionEngine`, register the backend, and add
   wire-level mappings.

## Verification Commands

```bash
cargo test -p opendecision-engine
```
