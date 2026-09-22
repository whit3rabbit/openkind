# openkind-engine

> Decision-inference trait abstractions, execution pipeline, and mock engine for `openkind`.

`openkind-engine` decouples the HTTP/gRPC transport layers from model implementations. It exposes the core `DecisionEngine` trait, the `EngineRegistry` for dispatching to registered model aliases, token estimation heuristics, the deterministic `MockEngine`, and immutable model-execution profile metadata.

## Architecture

```
HTTP (axum) / gRPC (tonic)
         │
         ▼
openkind_engine::dispatch(req, registry)
         │
         ├── validate_request(&req)
         ├── registry.get(model_alias)
         ├── engine.estimate_input_tokens(&req)
         ├── engine.evaluate(req) ──► Arc<dyn DecisionEngine>
         └── estimate_output_tokens(&resp)
```

## Key Components

- **`DecisionEngine` trait**:
  ```rust
  #[async_trait]
  pub trait DecisionEngine: Send + Sync {
      fn backend_id(&self) -> &str;
      fn model_metadata(&self) -> ModelInfo;
      async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse>;
      fn estimate_input_tokens(&self, req: &SystemRequest) -> u32;
  }
  ```
- **`EngineRegistry`**:
  Maps model aliases (e.g. `"jev-latest"`, `"mock"`) to `Arc<dyn DecisionEngine>`. Allows multiple public model aliases to route to specific backend implementations.
- **`dispatch()`**:
  Coordinates validation, telemetry, token accounting, and evaluation. Populates input and output token counts in the `Usage` block.
- **`MockEngine`**:
  A deterministic mock implementation that generates valid `Noul`, `Choice`, and `Score` answers seeded from question IDs and instructions. Guarantees probability distributions summing to 1.0, confidence within $[0.0, 1.0]$, and reproducible outputs without loading neural weights.
- **`ModelExecutionProfile`**:
  Binds a profile and reference bundle to an immutable backbone revision, renderer, fitted head, rejection method, calibration temperature, policy threshold, and parity tolerances. The selected profile is loaded by `openkind-backends`; the `DecisionEngine` trait remains unchanged.

## Testing

```bash
cargo test -p openkind-engine
```
