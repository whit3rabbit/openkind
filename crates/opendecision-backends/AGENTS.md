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
  - [`src/qwen35/backbone/nested.rs`](./src/qwen35/backbone/nested.rs): Phase 3.5 sequential nested execution (`run_sequential_nested`, `SequentialNestedExecutor`, `Qwen35Backbone::evaluate_nested`): one immutable prefill, per-question forks, per-candidate forks, fail-closed position/immutability checks.
  - [`src/qwen35/backbone/batched.rs`](./src/qwen35/backbone/batched.rs): Phase 3.6/3.7 breadth-first batched Q/K execution (`run_batched_questions`, `run_batched_candidates`, `run_batched_nested`, `Qwen35Backbone::evaluate_batched_nested`): `fork_batch` question lanes, per-question candidate fan-outs, fail-closed root/sibling/position checks, and fan-out byte accounting.
- Head evaluation uses f64 host algebra for normalization, projection, rejection, calibration, and stable softmax.
- Offline tests replay the exported features, exact token IDs, and the pinned diagnostic embedding row. Builds and tests do not download model assets.
- `BackboneReference` verifies 47 FP32 vectors and reports max-absolute, RMS, and cosine diagnostics for future native stages.
- `Qwen35Embedding` verifies the immutable checkpoint config, shard index, first-shard size/digest, BF16 tensor layout, token bounds, finite values, and exact BF16-to-FP32 widening.
- `Qwen35Backbone` verifies the second shard, executes 24 DeltaNet and 8 full-attention layers plus final RMSNorm in FP32, and exposes immutable Qwen-specific cached continuation.
- This module does not implement Metal, vectorized suffix kernels or the adaptive scheduler, server registration, or native-none wire mapping.

## Qwen State Contract

The complete branchable state for Qwen 3.5 includes attention KV, DeltaNet recurrent state, and convolution state. Attention masking or KV-only cloning cannot isolate branches.

`BackboneState` now carries the pinned profile/model/tokenizer/renderer/arithmetic identity (`pinned_state_identity()`), process-local branch lineage, logical position, and every continuation tensor. `fork_one`, `fork_batch`, `select`, and `gather` deep-copy all three tensor families; the structural fingerprint hashes identity, lineage root, position, and tensor layout so scheduling never hashes tens of MiB, while `strict_fingerprint` hashes exact little-endian tensor bytes as the cross-process replay identity. `storage_breakdown()` reports exact attention-KV, recurrent, convolution, and metadata bytes; the tensor total equals the Phase 3B `root_cache_bytes` fixture (`59,899,904` at position 98).

The named M4 Max checkpoint records maximum candidate-feature error
`1.0300e-04`, maximum probability delta `4.5869e-06`, zero argmax or policy
changes, exact root-state byte accounting, and exact native cached-versus-full
candidate equality. The Phase 3.5 nested gate (`qwen35_nested_parity`) prefills
each fixture case once and executes all four Phase 3B questions and 10
candidates through immutable forks with maximum probability delta `4.5869e-06`,
zero argmax or policy changes, root content identical to an independent
prefill after all fork work, exact replay and sibling-order determinism, and
exact cached-versus-full feature and state equality (`0.0`). The Phase 3.6/3.7
batched gate (`qwen35_batched_parity`) fans the same fixtures through
`fork_batch` question lanes (case 0: exactly `3 × 59,899,904 = 179,699,712`
root bytes) and per-question candidate lanes; every batched feature and strict
state fingerprint equals the sequential baseline exactly (`0.0`), the head
reaches the same `4.5869e-06` maximum probability delta with zero argmax,
zero policy, and zero cross-strategy decision changes, and fan-out byte
accounting is exact. These are
correctness-fixture results, not Metal or
production-throughput evidence. CPU native parity does not imply Metal or
accelerated parity. The next execution task is the adaptive scheduler and
target-Mac performance measurements (Phase 3.8).

## Verification Commands

```bash
cargo check -p opendecision-backends
cargo test -p opendecision-backends
```
