# AGENTS.md — opendecision-backends

> LLM developer guide for `opendecision-backends`. Read this before implementing model loaders and execution engines.

## Crate Purpose & Boundaries

`opendecision-backends` contains model-facing loaders, parity implementations, and future concrete `DecisionEngine` implementations.

It bridges neural network runtimes (e.g. `candle`, ONNX Runtime, or remote inference APIs) to the `SystemRequest` / `SystemResponse` wire contract.

- [`src/branch`](./src/branch): the backend-neutral `BranchableState` /
  `BranchBatch` contract: profile-bound `StateIdentity`, structural and
  strict `StateFingerprint`s, `StorageBreakdown` byte accounting, `StateError`,
  and the fork/batch/select/gather traits. The contract does not register a
  backend or map state onto the Jev wire format.

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
  - [`src/qwen35/backbone/branch.rs`](./src/qwen35/backbone/branch.rs): `BranchableState`/`BranchBatch` for `BackboneState` with `Qwen35BranchBatch`, storage breakdown, structural and strict fingerprints.
- Head evaluation uses f64 host algebra for normalization, projection, rejection, calibration, and stable softmax.
- Offline tests replay the exported features, exact token IDs, and the pinned diagnostic embedding row. Builds and tests do not download model assets.
- `BackboneReference` verifies 47 FP32 vectors and reports max-absolute, RMS, and cosine diagnostics for future native stages.
- `Qwen35Embedding` verifies the immutable checkpoint config, shard index, first-shard size/digest, BF16 tensor layout, token bounds, finite values, and exact BF16-to-FP32 widening.
- `Qwen35Backbone` verifies the second shard, executes 24 DeltaNet and 8 full-attention layers plus final RMSNorm in FP32, and exposes immutable Qwen-specific cached continuation.
- This module does not implement Metal, sequential or batched nested Q/K execution, server registration, or native-none wire mapping.

## Qwen State Contract

The complete branchable state for Qwen 3.5 includes attention KV, DeltaNet recurrent state, and convolution state. Attention masking or KV-only cloning cannot isolate branches.

`BackboneState` now carries the pinned profile/model/tokenizer/renderer/arithmetic identity (`pinned_state_identity()`), process-local branch lineage, logical position, and every continuation tensor. `fork_one`, `fork_batch`, `select`, and `gather` deep-copy all three tensor families; the structural fingerprint hashes identity, lineage root, position, and tensor layout so scheduling never hashes tens of MiB, while `strict_fingerprint` hashes exact little-endian tensor bytes as the cross-process replay identity. `storage_breakdown()` reports exact attention-KV, recurrent, convolution, and metadata bytes; the tensor total equals the Phase 3B `root_cache_bytes` fixture (`59,899,904` at position 98).

The named M4 Max checkpoint records maximum candidate-feature error
`1.0300e-04`, maximum probability delta `4.5869e-06`, zero argmax or policy
changes, exact root-state byte accounting, and exact native cached-versus-full
candidate equality. These are correctness-fixture results, not Metal or
production-throughput evidence. CPU native parity does not imply Metal or
accelerated parity. The next state task is sequential nested
`state → question → candidate` execution (Phase 3.5).

## Verification Commands

```bash
cargo check -p opendecision-backends
cargo test -p opendecision-backends
```
