# AGENTS.md — opendecision-engine

> LLM developer guide for `opendecision-engine`. Read this before modifying the engine trait, registry, or mock engine.

## Crate Purpose & Boundaries

`opendecision-engine` sits between transport layers (`opendecision-api`) and concrete model backends (`opendecision-backends`). It defines:
- The `DecisionEngine` trait that all model backends implement.
- `EngineRegistry`: Thread-safe model alias routing and deterministic listing.
- `dispatch()`: Unified evaluation orchestration (request validation, telemetry, and token usage calculation).
- `EngineError`: Transport-neutral error taxonomy.
- `MockEngine`: Deterministic pseudo-random engine for testing without loading model weights.
- `ModelExecutionProfile`: Immutable contract for model provenance, execution semantics, and numerical tolerances.

### Critical Invariants

1. **Runtime & Transport Agnostic**:
   - `DecisionEngine` operates purely on `opendecision_core::SystemRequest` and `opendecision_core::SystemResponse`.
   - **Never import transport types** (`axum`, `tonic`, HTTP status codes, headers) into `opendecision-engine`. Transport mapping belongs in `opendecision-api`.
2. **Deterministic Registry Resolution**:
   - `EngineRegistry::models()` must return registered model aliases in **lexicographically sorted order**.
   - `EngineRegistry::list_models()` must override the internal backend's `ModelInfo.name` with the **registered alias** (the client-facing identifier requested by operators).
3. **Usage Accounting Guarantee**:
   - `dispatch()` guarantees that `resp.usage.input_tokens` is populated (falling back to `engine.estimate_input_tokens(req)`) and `resp.usage.output_tokens` is calculated via `estimate_output_tokens(&resp.answers)`.

## Key Files & Types

- [`src/lib.rs`](./src/lib.rs): Crate facade, error re-exports, and public interfaces.
- [`src/engine.rs`](./src/engine.rs):
  - `pub trait DecisionEngine: Send + Sync`:
    - `fn backend_id(&self) -> &str`: Unique internal engine identity (e.g. `"mock"`, `"qwen35-native-cpu"`).
    - `fn model_metadata(&self) -> ModelInfo`: Descriptive metadata returned by `/v1/models`.
    - `async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse>`: Async execution entry point.
    - `fn estimate_input_tokens(&self, req: &SystemRequest) -> u32`: Heuristic or tokenizer-based input token estimate.
- [`src/registry.rs`](./src/registry.rs):
  - `pub struct EngineRegistry`: Stores `HashMap<String, Arc<dyn DecisionEngine>>`; registry construction/mutation happens before it is wrapped in shared app state, and query results are sorted deterministically.
- [`src/dispatch.rs`](./src/dispatch.rs):
  - `pub async fn dispatch(req, registry) -> EngineResult<SystemResponse>`:
    - Validates request using `opendecision_core::validate_request`.
    - Looks up model alias in `registry`.
    - Records metrics: `opendecision_requests_total`, `opendecision_responses_total`, and `opendecision_request_duration_ms`.
    - Executes `engine.evaluate()` and ensures token usage is populated.
    - `estimate_output_tokens(resp)`: Calculates fallback output token counts across answer types.
- [`src/error.rs`](./src/error.rs):
  - `enum EngineError`:
    - `Invalid(ValidationError)`: Input validation failure (mapped to HTTP 422).
    - `UnknownModel(String)`: Unregistered model alias (mapped to HTTP 404).
    - `Overloaded { backend, retry_after_ms }`: Concurrency/admission limit exceeded (mapped to HTTP 529 / retry headers).
    - `Unsupported { backend, message }`: Unsupported feature for backend (mapped to HTTP 422).
    - `Backend { backend, message }`: Internal execution failure (mapped to HTTP 500).
  - `EngineResult<T>` type alias.
- [`src/mock.rs`](./src/mock.rs):
  - `MockEngine`: Deterministic fake engine. Seeds PRNG with `(question_id, instructions)` hash to return reproducible distributions over choices and rubrics.
- [`src/profile.rs`](./src/profile.rs):
  - `ModelExecutionProfile`: Profile identity, bundle SHA-256, calibration temperature, and numerical parity tolerances.
  - `ProbabilitySpace` (on `ExecutionSemantics`): the declared meaning of returned probabilities — `ConditionalOnOfferedOptions` or `OfferedOptionsPlusSemanticNone`. Adapters must fail explicitly on a space they do not implement; silently dropping none mass and renormalizing is the drift this declaration prevents.
- [`src/tests.rs`](./src/tests.rs): Unit tests for registry operations, dispatch pipeline, and token usage accounting.

## Gotchas & Architectural Rules

1. **Alias vs Internal Backend ID**:
   Operators register engines under user-facing aliases (e.g. `qwen35-native`, `default`, `fast`). An engine's internal `backend_id()` is diagnostic (e.g. `qwen35-native-cpu`), but client requests and `/v1/models` must reflect the registered alias name.
2. **Error Category Symmetries**:
   When adding or modifying `EngineError` variants, ensure symmetric mapping exists in both `opendecision-api::http` (`ApiError`) and `opendecision-api::grpc` (`tonic::Status`).
3. **Deterministic Mocking**:
   Tests that need predictable answer distributions should use `MockEngine`. Because it hashes question identifiers, identical questions receive identical answers across runs.

## Verification Commands

```bash
cargo test -p opendecision-engine
```
