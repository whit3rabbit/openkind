# AGENTS.md — opendecision-backends

> LLM developer guide for `opendecision-backends`. Read this before implementing model loaders, execution backends, or numerical algebra.

## Crate Purpose & Boundaries

`opendecision-backends` contains concrete model execution engines, safetensors loaders, and numerical readouts bridging deep-learning runtimes (e.g. `candle`, FP32 CPU kernels) to the `opendecision_engine::DecisionEngine` wire contract.

It is responsible for:
- Deterministic score-summary readout math (projection, rejection, calibration, stable softmax).
- Offline, digest-locked tokenizer and checkpoint loading.
- Qwen 3.5 hybrid architecture execution (DeltaNet linear attention + causal grouped-query attention).
- Branchable continuation-state implementation (`BranchableState` / `BranchBatch`).
- Multi-lane execution strategies (`repeated_full`, `nested_sequential`, `nested_batched`) and adaptive scheduling.
- `Qwen35DecisionEngine`: Wire adapter implementing `opendecision_engine::DecisionEngine`.

### Invariants & Non-Autoregressive Execution Model

1. **No Autoregressive Text Generation**:
   - `opendecision` is a **decision engine**, not a text generator.
   - Forward execution computes final hidden-state representations for candidate suffixes and evaluates them through the score-summary readout head.
   - The execution loop never generates output tokens autoregressively.
2. **Deterministic Schema Emission**:
   - Host Rust code formats and validates JSON responses directly from floating-point candidate distributions. The model never outputs raw JSON text.
3. **Offline & Fail-Closed Loading**:
   - Builds and tests must never download model assets from the internet.
   - Checkpoint shards, config manifests, and tokenizer JSON files are digest-checked and fail closed on missing files or hash mismatches.
4. **Complete Hybrid State Isolation**:
   - Qwen 3.5 branch state includes attention KV, DeltaNet recurrent state, and convolution state.
   - Branching or cloning state must isolate all three tensor families; KV-only cloning fails to isolate DeltaNet and convolution state.

## Key Modules & Files

- [`src/branch/mod.rs`](./src/branch/mod.rs):
  - Compatibility re-export of the runtime-owned branch-state contract ([`opendecision_runtime::branch`](../opendecision-runtime/src/branch/mod.rs)).
- [`src/qwen35/mod.rs`](./src/qwen35/mod.rs): Root facade for the pinned Qwen 3.5 reference engine.
- [`src/qwen35/profile.rs`](./src/qwen35/profile.rs):
  - Validates profile `a047d6802c3f06f085b8`, bundle SHA-256, safetensors manifests, and numerical tolerances.
- [`src/qwen35/engine.rs`](./src/qwen35/engine.rs):
  - `Qwen35DecisionEngine`: Implements `opendecision_engine::DecisionEngine`.
  - `Qwen35EngineConfig`: File paths, concurrency bounds, admission queue limits, and scheduler configuration.
  - Concurrency management: Bounded by `execution_slots` and `admission_slots` semaphores; CPU forward passes run off-thread via `tokio::task::spawn_blocking`.
  - `SEMANTIC_NONE_OPTION = "__none__"`: Reserved Choice criteria key exposing semantic-none mass.
- [`src/qwen35/head/`](./src/qwen35/head/): Score-summary rejection readout:
  - [`types.rs`](./src/qwen35/head/types.rs): `PrimitiveKind`, `PolicyAction`, `HeadEvaluation`, and feature-width/profile constants.
  - [`tensors.rs`](./src/qwen35/head/tensors.rs): Safetensors extraction and f64 conversion.
  - [`math.rs`](./src/qwen35/head/math.rs): Numerically stable vector algebra, normalization, and stable softmax.
  - [`evaluation.rs`](./src/qwen35/head/evaluation.rs): Projection, rejection, temperature calibration, and distribution evaluation.
- [`src/qwen35/tokenizer.rs`](./src/qwen35/tokenizer.rs):
  - `Qwen35Tokenizer`: Digest-locked offline tokenizer loading.
  - `encode_state_first()`: Segmented encoding producing shared root IDs, question IDs, and candidate suffix IDs.
- [`src/qwen35/backbone/`](./src/qwen35/backbone/):
  - [`embedding.rs`](./src/qwen35/backbone/embedding.rs) & [`embedding/layout.rs`](./src/qwen35/backbone/embedding/layout.rs): Checkpoint embedding lookup, index validation, and BF16-to-FP32 widening.
  - [`layer0.rs`](./src/qwen35/backbone/layer0.rs), [`layer0/linear_attention.rs`](./src/qwen35/backbone/layer0/linear_attention.rs), [`layer0/full_attention.rs`](./src/qwen35/backbone/layer0/full_attention.rs): 24 DeltaNet recurrent layers, 8 grouped-query attention layers, and final RMSNorm in FP32.
  - [`branch.rs`](./src/qwen35/backbone/branch.rs): `BranchableState` and `BranchBatch` implementations for `BackboneState` and `Qwen35BranchBatch`.
  - [`persistence.rs`](./src/qwen35/backbone/persistence.rs): Atomic versioned pinned-state snapshot, envelope digest, identity/layout validation, and strict restored-content gate.
  - [`nested.rs`](./src/qwen35/backbone/nested.rs): Sequential nested execution (`run_sequential_nested`, `SequentialNestedExecutor`).
  - [`batched.rs`](./src/qwen35/backbone/batched.rs) & [`batched/types.rs`](./src/qwen35/backbone/batched/types.rs): Breadth-first batched question/candidate execution (`run_batched_questions`, `run_batched_candidates`, `run_batched_nested`).
  - [`strategy.rs`](./src/qwen35/backbone/strategy.rs) & [`strategy/policy.rs`](./src/qwen35/backbone/strategy/policy.rs): Strategy dispatcher (`run_strategy`, `choose_strategy`, `SchedulerConfig`).
  - [`reference.rs`](./src/qwen35/backbone/reference.rs): Golden vector validation for diagnostic stages.

## Critical Gotchas & Rules

1. **DeltaNet + Conv + KV State Isolation**:
   In Qwen 3.5, causal attention masking alone does NOT prevent crosstalk between questions in a batch. Recurrent DeltaNet state and 1D convolution state carry historical activations forward. Forking a branch requires deep-copying all three state tensor groups.
2. **Explicit Semantic None (`SEMANTIC_NONE_OPTION`)**:
   In native Choice questions, semantic-none mass is explicitly managed via `__none__`. The adapter requires this criteria key with a non-empty description, removes it from candidate texts sent to the model backbone, and restores the calibrated rejection mass to the response probabilities map.
3. **Entropy-Based Confidence Calculation**:
   The engine computes confidence using normalized distribution entropy ($1.0 - H / \ln(N)$), clamped to $[0.0, 1.0]$. It is NOT simply the maximum/top probability.
4. **Offline Parity Tests**:
   Offline tests replay golden feature fixtures and pinned embedding rows. Any modification to backbone math, layer logic, or head evaluation must pass the parity test suites without network access.
5. **Batching Claims**:
   `fork_batch` proves state isolation and lane topology. The current CPU executor still advances each lane separately. Do not claim compute batching unless `BackendCapabilities` advertises vectorized question and candidate forward.
6. **Memory Names**:
   `tensor_storage_bytes()` excludes metadata, allocator overhead, mapped weights, Candle objects, and forward scratch. Admission must use a process-memory envelope for those costs.
7. **Cancellation Holds Admission**:
   Queue and execution permits are owned by the blocking native task. A cancelled caller must not release them while the model forward continues in the background. Cooperative preemption remains a separate future capability.
8. **Persisted State Is Pinned**:
   Persist and restore only the selected execution identity and exact Qwen layer layout. Restored states receive fresh process-local lineage and must reproduce the stored `ContentFingerprint`; `SchedulingFingerprint` is never serialized as content evidence.

## Verification Commands

```bash
# Check compilation across all targets
cargo check -p opendecision-backends

# Run backends unit and parity test suites
cargo test -p opendecision-backends
```
