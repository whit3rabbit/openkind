# AGENTS.md — openpick-backends

> LLM developer guide for `openpick-backends`. Read this before implementing model loaders and execution engines.

## Crate Purpose & Boundaries

`openpick-backends` will contain the concrete model implementations of `openpick_engine::DecisionEngine`.

It bridges neural network runtimes (e.g. `candle`, ONNX Runtime, or remote inference APIs) to the `SystemRequest` / `SystemResponse` wire contract.

### Invariants & Non-Autoregressive Execution Model

1. **No Language Generation Loop**:
   - Jev is a **decision engine**, not a text generator.
   - The execution path must NOT generate token strings autoregressively.
   - Outputs are scalar logits projected into the finite candidate set supplied in the request criteria, followed by softmax (Choice/Score) or sigmoid (Noul).
2. **Schema Validity is Deterministic**:
   - JSON responses are serialized directly by host code from floating-point vectors and chosen keys. The model never outputs raw JSON text.
3. **Trait Implementation**:
   - Every backend struct must implement `openpick_engine::DecisionEngine`:
     - `backend_id()`: Canonical internal ID.
     - `model_metadata()`: Metadata returned by `/v1/models`.
     - `evaluate()`: Async forward pass returning `SystemResponse`.
     - `estimate_input_tokens()`: Tokenizer-based or heuristic token estimation.

## Planned Backends (Phase 2)

- **`CandleBackend`**:
  - Uses `candle-core` and `candle-nn`.
  - Loads GGUF quantized weights (e.g. Qwen 3.5 4B).
  - Implements the parallel decision head described in `docs/RESEARCH.md`.
- **`OnnxBackend`**:
  - Optional ONNX runtime execution for cross-platform deployments.
- **`RemoteBackend`**:
  - Passthrough backend proxying requests to hosted TypeSafe or external endpoints while maintaining local trait compatibility.

## Roadmap Status

Phase 2 placeholder. Gated on Phase 2B Python exploration and benchmark findings (`docs/ROADMAP.md`).

## Verification Commands

```bash
cargo check -p openpick-backends
cargo test -p openpick-backends
```
