# openkind-engine

> `DecisionEngine` dispatch, mock inference, and immutable execution-profile contracts for `openkind`.

`openkind-engine` decouples the HTTP and gRPC transport layers from model implementations. It defines the `DecisionEngine` trait every backend implements, the `EngineRegistry` that routes model aliases to engines, the shared `dispatch()` evaluation path, the deterministic `MockEngine`, and the immutable `ModelExecutionProfile` contract.

The crate depends on `openkind-core` and no transport stack: it imports no axum or tonic types. Model execution itself lives in [`openkind-backends`](../openkind-backends/README.md), which implements `DecisionEngine` and consumes the profile contracts below.

## Getting started

The transport layers call `dispatch` with any registered engine. The same path works in tests and tools:

```rust
use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::{NoulQuestion, Question, State, SystemRequest};
use openkind_engine::{dispatch, EngineRegistry, MockEngine};

#[tokio::main]
async fn main() -> Result<(), openkind_engine::EngineError> {
    let mut registry = EngineRegistry::new();
    registry.register("mock", Arc::new(MockEngine::new()));

    let mut questions = HashMap::new();
    questions.insert(
        "is_urgent".into(),
        Question::Noul(NoulQuestion {
            instructions: serde_json::json!("Is this urgent?"),
            criteria: None,
        }),
    );
    let request = SystemRequest {
        state: State::Text("Server down in production!".into()),
        model: "mock".into(),
        questions,
    };

    let response = dispatch(request, &registry).await?;
    println!("{:?}", response.answers);
    Ok(())
}
```

The example needs `tokio` (with the `macros` and `rt-multi-thread` features) and `serde_json` alongside this crate as a path dependency.

## Key components

```
HTTP / gRPC (openkind-api)
         │
         ▼
openkind_engine::dispatch(req, &registry)
         │
         ├── registry.get(&req.model) ──► Arc<dyn DecisionEngine>
         ├── ResponseContract::from_request(&req)
         ├── engine.estimate_input_tokens(&req)
         ├── engine.evaluate(req).await
         └── contract.validate(&resp), fill missing Usage counts
```

- **`DecisionEngine` trait**:
  ```rust
  #[async_trait]
  pub trait DecisionEngine: Send + Sync {
      // Required.
      fn backend_id(&self) -> &str;
      async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse>;

      // Provided defaults: `/v1/models` metadata and a rough `bytes / 4`
      // input estimate.
      fn model_metadata(&self) -> ModelInfo;
      fn estimate_input_tokens(&self, req: &SystemRequest) -> u32;
  }
  ```
- **`EngineRegistry`**:
  Maps public model aliases (e.g. `"jev-latest"`, `"mock"`) to `Arc<dyn DecisionEngine>`. `register` replaces an alias, while `register_if_absent` and `unregister` drive load and unload. Lookups clone the engine handle, so accepted requests keep executing through an unload. `list_models()` overrides each engine's `ModelInfo.name` with the registered alias.
- **`dispatch()`**:
  Resolves the alias, validates the request through `ResponseContract::from_request`, records request metrics, evaluates, and re-validates the engine response against the request's contract. A contract-violating response is an `EngineError::Backend` fault, not a client-facing validation error. Token counts fall back to `estimate_input_tokens` and a per-answer output estimator only when the backend leaves them unset.
- **`MockEngine`**:
  Deterministic `Noul`, `Choice`, and `Score` answers without loading weights. Each answer is seeded from the question ID and structurally hashed instructions, so repeated requests reproduce identical distributions. Probabilities sum to 1.0, and confidence stays within [0.0, 1.0].
- **`ModelExecutionProfile`**:
  Immutable contract binding a reference-bundle source (pinned repository revision and SHA-256), a backbone revision, execution semantics (renderer, fitted head, rejection method, and the declared `ProbabilitySpace`), and the numerical policy (calibration temperature, policy threshold, parity tolerances). Fields are validated at construction. `openkind-backends` builds real engines from a selected profile; the `DecisionEngine` trait itself carries no profile state.
- **`EngineError`**:
  Transport-neutral taxonomy of `Invalid`, `UnknownModel`, `Unsupported`, `Overloaded`, `DeadlineExceeded`, and `Backend` variants. The API layer maps each onto HTTP and gRPC statuses.

## Testing

```bash
cargo test -p openkind-engine
```

Unit tests cover registry ordering and lifecycle, dispatch validation and token fallbacks, mock determinism and distribution guarantees, and profile validation.

## License

See the [MIT license](../../LICENSE). Cargo metadata declares `MIT OR Apache-2.0`.
