# AGENTS.md — openkind-backends

> LLM developer guide for `openkind-backends`. Read this before implementing model loaders, execution backends, or numerical algebra.

## Crate Purpose & Boundaries

`openkind-backends` contains concrete model execution engines, safetensors loaders, and numerical readouts bridging deep-learning runtimes (e.g. `candle`, FP32 CPU kernels) to the `openkind_engine::DecisionEngine` wire contract.

It is responsible for:
- Deterministic score-summary readout math (projection, rejection, calibration, stable softmax).
- Offline, digest-locked tokenizer and checkpoint loading.
- Qwen 3.5 hybrid architecture execution (DeltaNet linear attention + causal grouped-query attention).
- Branchable continuation-state implementation (`BranchableState` / `BranchBatch`).
- Multi-lane execution strategies (`repeated_full`, `nested_sequential`, `nested_batched`) and adaptive scheduling.
- `Qwen35DecisionEngine`: Wire adapter implementing `openkind_engine::DecisionEngine`.

### Invariants & Non-Autoregressive Execution Model

1. **No Autoregressive Text Generation**:
   - `openkind` is a **decision engine**, not a text generator.
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

### Model acquisition (explicit opt-in)

Model downloads are an operator step, never a build or test side effect. Use
the Hugging Face CLI to place artifacts outside the repository, then pass the
resulting directory to the parity examples. The pinned base checkpoint is the
only frozen parity target:

```bash
# Run this only if the `hf` command is not already installed.
python3 -m pip install --user --upgrade huggingface_hub
MODEL_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/openkind"

hf download Qwen/Qwen3.5-4B-Base \
    --revision 1001bb4d826a52d1f399e183466143f4da7b741b \
    --local-dir "$MODEL_CACHE_DIR/qwen35-4b-base-1001bb4d826a52d1f399e183466143f4da7b741b"
```

For the separately identified MLX-community compatibility comparison:

```bash
hf download mlx-community/Qwen3.5-4B-MLX-bf16 \
    --revision 475632ded9a95863da4e4b235ab9ccbc5d3cc6bf \
    --local-dir "$MODEL_CACHE_DIR/mlx-community-qwen35-4b-mlx-bf16-475632d"
```

The MLX loader validates the local config, tokenizer, index, shard sizes, and
hashes. The community export is not a replacement for the pinned target: its
source model and conversion differ, so successful loading does not establish
parity. Keep these downloads out of tests and CI, which must remain offline.

## Key Modules & Files

- [`src/branch/mod.rs`](./src/branch/mod.rs):
  - Compatibility re-export of the runtime-owned branch-state contract ([`openkind_runtime::branch`](../openkind-runtime/src/branch/mod.rs)).
- [`src/qwen35/mod.rs`](./src/qwen35/mod.rs): Root facade for the pinned Qwen 3.5 reference engine.
- [`src/qwen35/profile.rs`](./src/qwen35/profile.rs):
  - Validates profile `a047d6802c3f06f085b8`, bundle SHA-256, safetensors manifests, and numerical tolerances.
- [`src/qwen35/engine/`](./src/qwen35/engine/): Direct Jev wire adapter for the native reference engine:
  - [`mod.rs`](./src/qwen35/engine/mod.rs): `Qwen35DecisionEngine` (implements `openkind_engine::DecisionEngine`), `Qwen35EngineConfig`, semaphore admission, off-thread blocking spawn, and `SEMANTIC_NONE_OPTION = "__none__"`.
  - [`backbone.rs`](./src/qwen35/engine/backbone.rs): Execution-backend selection. `Qwen35Backend` (`NativeCpu`, plus `MlxFp32`/`MlxBf16` behind `mlx`) picks the backbone behind the same backend-neutral executor contract; MLX loads re-derive scheduler state-size constants from the loaded model (BF16 states are half the FP32 bytes). Only FP32 `ReferenceOps` with an explicitly forced `NestedBatched` plan advertises the implemented 2–8-lane vectorized forward. The unequal-length frozen batch fixture passes, but automatic scheduling and unsupported shapes remain per-lane until matched performance evidence supports broader capability. BF16 loads run the runtime preflight; pinned-base full and nested probability gates fail at the unchanged tolerance.
  - [`canonical.rs`](./src/qwen35/engine/canonical.rs): Structured wire state canonicalization (`state_text`) with byte-lexicographically sorted object keys at every nesting level.
  - [`eval.rs`](./src/qwen35/engine/eval.rs): Model evaluation pipeline (`evaluate_request`), scheduler decision logging, and engine error mapping.
  - [`mapping.rs`](./src/qwen35/engine/mapping.rs): Question criteria extraction, entropy-based confidence calculation, and distribution answer mapping.
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
  - [`strategy.rs`](./src/qwen35/backbone/strategy.rs) & [`strategy/policy.rs`](./src/qwen35/backbone/strategy/policy.rs): Strategy dispatcher (`run_strategy`, `choose_strategy`, `SchedulerConfig`). `SchedulerConfig::forced_strategy` overrides the plan for diagnostics — bypassing the savings-ratio and vectorized-preference policy only; tensor/process admission still fails closed. Every `StrategyDecision` carries the selected plan, its physical `BatchForwardMode`, and a `forced` flag.
  - [`reference.rs`](./src/qwen35/backbone/reference.rs): Golden vector validation for diagnostic stages.
  - [`identity.rs`](./src/qwen35/identity.rs): `ExecutionIdentity` — profile/renderer/tokenizer/arithmetic identity plus the role-typed token digests of one finalized request (order-sensitive `execution_input_digest` is the reproducibility identity; `semantic_set_digest` is order-independent).
  - [`evidence.rs`](./src/qwen35/evidence.rs): Maps the pinned profile, scheduler config, and one decision onto the backend-neutral `openkind-native-run/v1` records from `openkind-runtime::evidence`.
  - [`engine/canonical.rs` state canonicalization](./src/qwen35/engine/canonical.rs): Structured wire state (`State::Object`/`State::Array`) renders through an explicit canonical serializer with byte-lexicographically sorted object keys at every nesting level; key construction order and the JSON map implementation cannot alter the model input. This is the `state_first` renderer's ordering semantics — no renderer-ID bump.
- [`src/qwen35/mlx/`](./src/qwen35/mlx/): Optional MLX/Metal parity backend (`--features mlx`):
  - [`runtime.rs`](./src/qwen35/mlx/runtime.rs): Process-wide serialized explicit-stream execution, memory telemetry, and toolchain qualification.
  - [`weights/`](./src/qwen35/mlx/weights/): Safetensors checkpoint loader with format detection and key normalization:
    - [`mod.rs`](./src/qwen35/mlx/weights/mod.rs): `MlxWeightStore` and `MlxWeightLoadReport` streaming loader.
    - [`checkpoint.rs`](./src/qwen35/mlx/weights/checkpoint.rs): `MlxCheckpointFormat` detection, size/hash verification, and namespace mapping.
    - [`shard.rs`](./src/qwen35/mlx/weights/shard.rs): Safetensors shard directory parsing and host widening (`widen_bf16`, `read_f32`, `shape_i32`).
  - [`model.rs`](./src/qwen35/mlx/model.rs): `MlxQwen35Backbone` and continuation state container.
  - [`branch_state.rs`](./src/qwen35/mlx/branch_state.rs): `BranchableState` and `BranchBatch` implementation for MLX.
  - [`layers/`](./src/qwen35/mlx/layers/): Decoder blocks decomposed into modular components:
    - [`mod.rs`](./src/qwen35/mlx/layers/mod.rs): Block lifecycle (`MlxDecoderLayer`), continuation states (`MlxLinearState`, `MlxFullState`, `MlxLayerState`), and mixer dispatch.
    - [`linear_attention.rs`](./src/qwen35/mlx/layers/linear_attention.rs): Linear attention forward pass and vectorized per-token `gated_delta_step`.
    - [`gated_delta_kernel.rs`](./src/qwen35/mlx/layers/gated_delta_kernel.rs): Generic masked/vector-gate and packed FP32 `Dk = Dv = 128` custom Metal reduction-tree kernels.
    - [`full_attention.rs`](./src/qwen35/mlx/layers/full_attention.rs): Grouped-query attention, rotary embedding (`apply_rotary`), and per-head normalization.
    - [`ops.rs`](./src/qwen35/mlx/layers/ops.rs): MLX array operations, causal conv windowing, and tensor loading helpers.
    - [`differential_tests/`](./src/qwen35/mlx/layers/differential_tests/): Independent FP32 host reference and differential verification tests.
- Multi-file examples:
  - [`examples/qwen35_mlx_qualify/`](./examples/qwen35_mlx_qualify/): Phase 3M.0 runtime qualification suite (`main.rs`, `gate.rs`, `fp32.rs`, `bf16.rs`, `helpers.rs`).
  - [`examples/qwen35_mlx_full_parity/`](./examples/qwen35_mlx_full_parity/): Phase 3M.2–3M.4 full-sequence parity gate (`main.rs`, `full.rs`, `trace.rs`, `fixtures.rs`).

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
   Queue and execution permits are owned by the blocking native task. A cancelled caller must not release them early; the CPU executor observes cancellation and queue-inclusive deadlines between decoder layers, then releases both permits when it exits. Executors with a non-interruptible forward check before and after that forward, so cancellation latency is bounded by the active backend operation rather than a CPU layer. Metrics may record fixed outcome labels and durations only, never request IDs, input content, token IDs, fingerprints, or digests. The named CPU service evidence is in [`docs/verification/native-service-gate/2026-09-22/`](../../docs/verification/native-service-gate/2026-09-22/README.md) and does not qualify MLX cancellation timing.
8. **Persisted State Is Pinned**:
   Persist and restore only the selected execution identity and exact Qwen layer layout. Restored states receive fresh process-local lineage and must reproduce the stored `ContentFingerprint`; `SchedulingFingerprint` is never serialized as content evidence.
   The checkpoint-gated `qwen35_parity_probe persist-replay` compares every restored candidate feature and full head decision against independent full-sequence forwards. The named-host CPU result is recorded in [`docs/verification/phase3.10-2026-09-22/`](../../docs/verification/phase3.10-2026-09-22/README.md).
9. **MLX Backend (`--features mlx`, macOS arm64)**:
   The MLX parity backend in `src/qwen35/mlx/` is optional. Rules: build with
   `SDKROOT=$(xcrun --show-sdk-path)` (bindgen needs the macOS SDK); all MLX
   work goes through `MlxRuntime::execute`, which holds one process-wide lock
   and installs the same explicit cross-thread GPU stream for every operation;
   evaluation outside that scope is unsound. MLX `conv1d` is true
   convolution (kernel reversed, weight `(C_out, K, C_in/groups)`) — the
   recurrent path implements the causal conv explicitly instead; the
   pinned checkpoint stores `A_log` and `linear_attn.norm.weight` in FP32 and
   the rest in BF16 (the loader handles both). The verified
   `mlx-community/Qwen3.5-4B-MLX-bf16` adapter additionally handles its
   `language_model.model.*` and `vision_tower.*` prefixes, `__metadata__`
   safetensors header, `[C, K, 1]` convolution layout, and BF16
   `linear_attn.norm.weight`; layernorm weights fold `(1 + w)` at load but
   DeltaNet `norm.weight` stays raw; MLX `Array` clone is a
   refcounted handle copy and arrays are immutable values, so branch
   isolation is structural (verified by strict fingerprints); arithmetic
   identities include precision and kernel family
   (`mlx-core-0.32.2/fp32/reference-ops`), so MLX states never mix with
   Candle states. The generic masked/vector-gate and packed FP32 reduction-tree
   kernels are opt-in candidates; `ReferenceOps` remains the default because
   the packed sequence candidate passed FP32 parity but was 14–28% slower on
   the smoke sweep. Native BF16 falls back to `ReferenceOps` because its fused
   candidate failed the frozen model gate. A different Xcode/Metal toolchain is a different runtime —
   re-run `qwen35_mlx_qualify` (3M.0) before trusting any MLX gate after a
   toolchain change; bf16 is a separately gated candidate profile and must
   never be treated as a default-equivalent of the FP32 oracle.
10. **Confine Execution Identity Digests to Offline Evidence**:
    Do not emit raw token digests (`execution_input_digest`, `state_token_digest`, `semantic_set_digest`) in daemon telemetry or debug tracing during runtime evaluation. Raw digests of low-entropy inputs can be guessed offline. Reserve them strictly for explicit offline evidence generation.
11. **Complete Candidate Retention in Memory Estimation**:
    When calculating retention in the execution strategy scheduler, account for all candidate states retained by the chosen strategy. In particular, `repeated_full` retains all candidate states across all questions in its `StrategyOutput`. Always use saturating arithmetic when computing total retained bytes to avoid overflow.
12. **In-Place Checkpoint Verification**:
    Safetensors shards must be verified in-place on their read-only filesystem paths. Never copy or stage multi-gigabyte model weights to `/tmp` or ephemeral directories during model loading or inference.

## Verification Commands

```bash
# Check compilation across all targets
cargo check -p openkind-backends

# Run backends unit and parity test suites
cargo test -p openkind-backends
```

### MLX feature (macOS arm64 only)

```bash
export SDKROOT=$(xcrun --show-sdk-path)

cargo check -p openkind-backends --features mlx
cargo clippy -p openkind-backends --features mlx --all-targets -- -D warnings
cargo test  -p openkind-backends --features mlx

# 3M.0 runtime qualification (offline, no checkpoint): exit 0 qualified,
# 3 = fp32 ok but bf16 blocked, 1 = runtime invalid.
cargo run -p openkind-backends --features mlx --release \
    --example qwen35_mlx_qualify -- --formal

# Checkpoint-gated parity examples (<checkpoint-root> <phase3b-reference-root>
# <head-bundle-root>; --precision fp32|bf16):
cargo run -p openkind-backends --features mlx --release \
    --example qwen35_mlx_full_parity -- --stage full --formal <paths...>
cargo run -p openkind-backends --features mlx --release \
    --example qwen35_mlx_nested_parity -- --formal <paths...>
```

### MLX benchmark dispatch

For warm request throughput, use the benchmark harness rather than the parity
examples. The harness dispatches `qwen35-mlx-fp32` through the same
`Qwen35DecisionEngine` path as the CPU backend:

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --release \
  -p openkind-bench --features mlx -- score <workload.jsonl> \
  --engine qwen35-mlx-fp32 \
  --bundle-root <profile-bundle-dir> \
  --checkpoint-root <pinned-checkpoint-dir> \
  --tokenizer <digest-locked-tokenizer.json> \
  --strategies repeated_full,nested_sequential,nested_batched,choose_strategy \
  --reps 1 --host "<host label>" --commit <hash> \
  --output-dir <benchmark-output-dir>
```

The pinned comparison model is `Qwen/Qwen3.5-4B-Base` at revision
`1001bb4d826a52d1f399e183466143f4da7b741b`. The MLX backend is built with the
vendored MLX 0.32.2 toolchain. Keep BF16 benchmark comparisons on
`--strategies repeated_full` until Gate B passes; nested continuation now
completes its state checks but exceeds the frozen probability tolerance. The
community `mlx-community/Qwen3.5-4B-MLX-bf16` artifact is a
separate model/conversion and is throughput evidence only. See
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for recorded timings,
speedups, and parity boundaries.
