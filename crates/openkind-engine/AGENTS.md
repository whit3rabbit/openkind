# AGENTS.md — openkind-engine

> LLM developer guide for `openkind-engine`. Read this before modifying the engine trait, registry, or mock engine.

## Crate Purpose & Boundaries

`openkind-engine` sits between transport layers (`openkind-api`) and concrete model backends (`openkind-backends`). It defines:
- The `DecisionEngine` trait that all model backends implement.
- `EngineRegistry`: Thread-safe model alias routing and deterministic listing.
- `dispatch()`: Unified evaluation orchestration (request and response validation, telemetry, and token usage calculation).
- `EngineError`: Transport-neutral error taxonomy.
- `MockEngine`: Deterministic pseudo-random engine for testing without loading model weights.
- `ModelExecutionProfile`: Immutable contract for model provenance, execution semantics, and numerical tolerances.

### Critical Invariants

1. **Runtime & Transport Agnostic**:
   - `DecisionEngine` operates purely on `openkind_core::SystemRequest` and `openkind_core::SystemResponse`.
   - **Never import transport types** (`axum`, `tonic`, HTTP status codes, headers) into `openkind-engine`. Transport mapping belongs in `openkind-api`.
2. **Deterministic Registry Resolution**:
   - `EngineRegistry::models()` must return registered model aliases in **lexicographically sorted order**.
   - `EngineRegistry::list_models()` must override the internal backend's `ModelInfo.name` with the **registered alias** (the client-facing identifier requested by operators).
3. **Usage Accounting Guarantee**:
   - `dispatch()` guarantees that `resp.usage.input_tokens` is populated (falling back to `engine.estimate_input_tokens(req)`) and `resp.usage.output_tokens` is calculated via `estimate_output_tokens(&resp.answers)`.
4. **Backend Response Contract**:
   - `dispatch()` captures request-bound answer expectations before passing the request to the engine. Missing,
     extra, wrong-type, or out-of-list backend answers are `EngineError::BackendValidation` faults carrying the original validation error.

## Key Files & Types

- [`src/lib.rs`](./src/lib.rs): Crate facade, error re-exports, and public interfaces.
- [`src/engine.rs`](./src/engine.rs):
  - `pub trait DecisionEngine: Send + Sync`:
    - `fn backend_id(&self) -> &str`: Unique internal engine identity (e.g. `"mock"`, `"qwen35-native-cpu"`).
    - `fn model_metadata(&self) -> ModelInfo`: Descriptive metadata returned by `/v1/models`.
    - `async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse>`: Async execution entry point.
    - `fn estimate_input_tokens(&self, req: &SystemRequest) -> u32`: Heuristic or tokenizer-based input token estimate.
- [`src/registry.rs`](./src/registry.rs):
  - `pub struct EngineRegistry`: Shares a lock-protected alias map across clones. Startup registration uses `register`; explicit lifecycle controls use `register_if_absent` and `unregister`. Lookups clone engine handles before releasing the lock, so accepted requests survive unloading. Query results are sorted deterministically.
- [`src/dispatch.rs`](./src/dispatch.rs):
  - `pub async fn dispatch(req, registry) -> EngineResult<SystemResponse>`:
    - Captures and validates `openkind_core::ResponseContract` from the request.
    - Looks up model alias in `registry`.
    - Records metrics: `openkind_requests_total`, `openkind_responses_total`, `openkind_request_duration_ms` (fractional milliseconds), and `openkind_request_outcomes_total` (fixed outcomes, including cancellation).
    - Executes `engine.evaluate()`, validates its response against the contract, and ensures token usage is populated.
    - `estimate_output_tokens(resp)`: Calculates fallback output token counts across answer types.
- [`src/error.rs`](./src/error.rs):
  - `enum EngineError`:
    - `Invalid(ValidationError)`: Input validation failure (mapped to HTTP 422).
    - `UnknownModel(String)`: Unregistered model alias (mapped to HTTP 404).
    - `Overloaded { backend, retry_after_ms }`: Concurrency/admission limit exceeded (mapped to HTTP 529 / retry headers).
    - `Unsupported { backend, message }`: Unsupported feature for backend (mapped to HTTP 422).
    - `DeadlineExceeded { backend, timeout_ms }`: Queue-inclusive evaluation deadline elapsed (mapped to HTTP 504).
    - `BackendValidation { backend, source }`: Backend response contract failure with the original validation error (HTTP 500 / gRPC Internal).
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
   When adding or modifying `EngineError` variants, ensure symmetric mapping exists in both `openkind-api::http` (`ApiError`) and `openkind-api::grpc` (`tonic::Status`).
3. **Deterministic Mocking**:
   Tests that need predictable answer distributions should use `MockEngine`. Because it hashes question identifiers, identical questions receive identical answers across runs.
4. **EngineRegistry Is Not an Artifact Loader**:
   `EngineRegistry` maps aliases to initialized `Arc<dyn DecisionEngine>` values. Callers such as `openkindd` verify artifacts and load engines before registration.

## Verification Commands

```bash
cargo test -p openkind-engine
```
