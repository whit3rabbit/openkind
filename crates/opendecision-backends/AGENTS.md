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

## Qwen 3.5 parity scope

- [`src/qwen35`](./src/qwen35) loads the pinned `a047d6802c3f06f085b8` contracts fail closed:
  - [`src/qwen35/profile.rs`](./src/qwen35/profile.rs): Offline manifest, profile, and bundle validation.
  - [`src/qwen35/head/`](./src/qwen35/head/): Modular score-summary readout:
    - [`types.rs`](./src/qwen35/head/types.rs): `ScoreSummaryHead`, `HeadWeights`, `HeadShape`, `CandidateLogits`, and `HeadEvaluationResult`.
    - [`tensors.rs`](./src/qwen35/head/tensors.rs): SafeTensor loading and f64 matrix/vector extraction.
    - [`math.rs`](./src/qwen35/head/math.rs): Numerically stable vector operations, normalization, and stable softmax.
    - [`evaluation.rs`](./src/qwen35/head/evaluation.rs): Projection, rejection, temperature calibration, and distribution evaluation.
    - [`tests.rs`](./src/qwen35/head/tests.rs): Unit tests for math, weight extraction, and head evaluation.
  - [`src/qwen35/tokenizer.rs`](./src/qwen35/tokenizer.rs): Digest-locked offline tokenizer and exact state-first segmented rendering.
  - [`src/qwen35/backbone/`](./src/qwen35/backbone/): Phase 3B contract validation, exact checkpoint embedding, 32-layer Candle CPU execution, final RMSNorm, and complete Qwen continuation state.
- Head evaluation uses f64 host algebra for normalization, projection, rejection, calibration, and stable softmax.
- Offline tests replay the exported features, exact token IDs, and the pinned diagnostic embedding row. Builds and tests do not download model assets.
- `BackboneReference` verifies 47 FP32 vectors and reports max-absolute, RMS, and cosine diagnostics for future native stages.
- `Qwen35Embedding` verifies the immutable checkpoint config, shard index, first-shard size/digest, BF16 tensor layout, token bounds, finite values, and exact BF16-to-FP32 widening.
- `Qwen35Backbone` verifies the second shard, executes 24 DeltaNet and 8 full-attention layers plus final RMSNorm in FP32, and exposes immutable Qwen-specific cached continuation.
- This module does not implement Metal, backend-neutral `BranchableState`, batched fork/gather, server registration, or native-none wire mapping.

## Qwen State Contract

The complete branchable state for Qwen 3.5 includes attention KV, DeltaNet recurrent state, and convolution state. Attention masking or KV-only cloning cannot isolate branches.

Full-sequence and cached-continuation fixture parity now pass on the CPU path. The next state task is to lift `BackboneState` into the backend-neutral `BranchableState` contract with profile identity, stable fingerprints, fork, batched fork, and gather/select semantics.

The named M4 Max checkpoint records maximum candidate-feature error
`1.0300e-04`, maximum probability delta `4.5869e-06`, zero argmax or policy
changes, exact root-state byte accounting, and exact native cached-versus-full
candidate equality. These are correctness-fixture results, not Metal or
production-throughput evidence.

## Verification Commands

```bash
cargo check -p opendecision-backends
cargo test -p opendecision-backends
```
