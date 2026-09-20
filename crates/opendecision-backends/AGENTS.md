# AGENTS.md — opendecision-backends

> LLM developer guide for `opendecision-backends`. Read this before implementing model loaders and execution engines.

## Crate Purpose & Boundaries

`opendecision-backends` contains model-facing loaders, parity implementations, and future concrete `DecisionEngine` implementations.

It bridges neural network runtimes (e.g. `candle`, ONNX Runtime, or remote inference APIs) to the `SystemRequest` / `SystemResponse` wire contract.

### Invariants & Non-Autoregressive Execution Model

1. **No Language Generation Loop**:
   - Jev is a **decision engine**, not a text generator.
   - The execution path must NOT generate token strings autoregressively.
   - Outputs are scalar candidate logits projected directly into a calibrated finite distribution.
2. **Schema Validity is Deterministic**:
   - JSON responses are serialized directly by host code from floating-point vectors and chosen keys. The model never outputs raw JSON text.
3. **Trait Implementation**:
   - Every backend struct must implement `opendecision_engine::DecisionEngine`:
     - `backend_id()`: Canonical internal ID.
     - `model_metadata()`: Metadata returned by `/v1/models`.
     - `evaluate()`: Async forward pass returning `SystemResponse`.
     - `estimate_input_tokens()`: Tokenizer-based or heuristic token estimation.

## Qwen 3.5 Readout Scope

- [`src/qwen35`](./src/qwen35) loads the pinned `a047d6802c3f06f085b8` metadata and score-summary head fail closed:
  - [`src/qwen35/manifest.rs`](./src/qwen35/manifest.rs): Offline manifest parsing and bundle SHA-256 verification.
  - [`src/qwen35/head/`](./src/qwen35/head/): Modular score-summary readout:
    - [`types.rs`](./src/qwen35/head/types.rs): `ScoreSummaryHead`, `HeadWeights`, `HeadShape`, `CandidateLogits`, and `HeadEvaluationResult`.
    - [`tensors.rs`](./src/qwen35/head/tensors.rs): SafeTensor loading and f64 matrix/vector extraction.
    - [`math.rs`](./src/qwen35/head/math.rs): Numerically stable vector operations, normalization, and stable softmax.
    - [`evaluation.rs`](./src/qwen35/head/evaluation.rs): Projection, rejection, temperature calibration, and distribution evaluation.
    - [`tests.rs`](./src/qwen35/head/tests.rs): Unit tests for math, weight extraction, and head evaluation.
- Head evaluation uses f64 host algebra for normalization, projection, rejection, calibration, and stable softmax.
- Vendored tests replay the exported features. Builds and tests do not download model assets.
- This module does not implement tokenization, Qwen execution, Candle, server registration, or native-none wire mapping.

## Qwen State Contract

The complete branchable state for Qwen 3.5 includes attention KV, DeltaNet recurrent state, and convolution state. Attention masking or KV-only cloning cannot isolate branches.

## Verification Commands

```bash
cargo check -p opendecision-backends
cargo test -p opendecision-backends
```
