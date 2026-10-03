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

Before implementing a model profile or family, follow the
[family integration guide](../../docs/families/NEW_FAMILY.md). A backend enum,
checkpoint, or architecture page alone is not a Rust loader. Mark a profile
Rust-loadable in the [family registry](../../docs/families/README.md) only
after its artifacts are pinned and load locally, it has offline parity
fixtures and a `DecisionEngine` adapter, and the daemon registers it under an
alias. Keep task quality and release promotion as separate gates.

### Invariants & Non-Autoregressive Execution Model

1. **No Autoregressive Text Generation**:
   - `openkind` is a **decision engine**, not a text generator.
   - Forward execution uses the selected profile's readout, such as a score-summary head, bounded label logits, or encoder marker scores.
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

- [`src/branch/mod.rs`](./src/branch/mod.rs): Re-exports the branch-state contract owned by [`openkind-runtime`](../openkind-runtime/src/branch/mod.rs).
- [`src/qwen35/`](./src/qwen35/): Native Qwen engine, tokenizer, continuation state, schedulers, and score readout.
- [`src/qwen35/engine/`](./src/qwen35/engine/): Wire adapter, backend selection, request evaluation, identity, and answer mapping.
- [`src/qwen35/backbone/`](./src/qwen35/backbone/): Hybrid model execution, branch state, persistence, and execution strategies.
- [`src/qwen35/head/`](./src/qwen35/head/): Deterministic score-summary projection, rejection, and calibration.
- [`src/qwen35/experimental.rs`](./src/qwen35/experimental.rs): Offline scoring probes. The daemon does not register them. See benchmark docs for methodology.
- [`src/qwen35/mlx/`](./src/qwen35/mlx/): Optional MLX backend. Checkpoint layouts, arithmetic paths, and kernel notes are in [MLX backend internals](../../.claude/docs/mlx-backend-internals.md).
- [`src/families/`](./src/families/): Surveyed-family adapters and shared readouts. The family registry owns profile names and status.
- [`src/device.rs`](./src/device.rs): `FamilyExecution` device resolution enum (`Cpu`, `Cuda`, `Onnx`, `OnnxRocm`) and adapter dispatch.
- [`src/onnx/`](./src/onnx/): ONNX Runtime execution provider integration (`OnnxModel`, `OnnxAcceleration`), optional feature gates (`onnx`, `onnx-cuda`, `onnx-rocm`), and per-family adapters.
- [`src/proxy_cache/`](./src/proxy_cache/): Distilling cache and training lifecycle. See [`docs/PROXY_CACHE.md`](../../docs/PROXY_CACHE.md).
- [`examples/`](./examples/): Offline parity and MLX qualification programs. See [`docs/MLX.md`](../../docs/MLX.md).

## Critical Gotchas & Rules

1. **DeltaNet + Conv + KV State Isolation**:
   Causal attention does not prevent Qwen 3.5 branch crosstalk. Deep-copy attention KV, DeltaNet recurrent state, and convolution state.
2. **Explicit Semantic None**:
   Native Choice requires a non-empty `__none__` criterion. Remove it from candidate text, then restore calibrated rejection mass in the probability map.
3. **Entropy-Based Confidence**:
   Confidence uses normalized entropy, `1 - H / ln(N)`, clamped to `[0,1]`. It is not the top probability.
4. **Offline Parity Tests**:
   Keep tests offline. Checkpoint-gated replays return early without model roots. Report those replays as skipped. Changes to backbone math, layer logic, or readout require replay against the pinned checkpoint. See the family pages for commands.
5. **Batching Claims**:
   `fork_batch` proves state isolation and lane topology, not compute batching. Claim vectorization only when `BackendCapabilities` advertises it.
6. **Memory Names**:
   `tensor_storage_bytes()` excludes metadata, allocator overhead, mapped weights, Candle objects, and scratch. Admission must also use process-memory limits.
7. **Cancellation Holds Admission**:
   Queue and execution permits stay with the blocking model task until it exits. CPU checks cancellation between layers; non-interruptible backends check around each call. Metrics include only fixed labels and durations.
8. **Persisted State Is Pinned**:
   Restore only the selected execution identity and exact Qwen layout. Restored states need fresh lineage and must reproduce `ContentFingerprint`; never serialize `SchedulingFingerprint` as content evidence.
9. **MLX Execution Discipline**:
   MLX requires macOS arm64 and the `mlx` feature. Set `SDKROOT=$(xcrun --show-sdk-path)`. `MlxRuntime::execute` holds a non-reentrant process lock. Never call it around code that takes the lock. Requalify after toolchain changes.
10. **MLX Identity and Gates**:
    MLX identity includes precision and kernel family. Never mix MLX and Candle states. BF16 is a separate candidate, not an FP32 equivalent. Use the [benchmark guide](../openkind-bench/AGENTS.md) for comparison rules.
11. **Gemma 4 Attention Numerics**: Gemma 4 pins the attention softmax
    scale to `1.0` (no `1/sqrt(head_dim)`), so candle's quantized kernels
    (which dequantize q8_0 blocks to f16) reshuffle the softmax argmax.
    The gemma4 family's attention projections must run as dequantized F32
    linears; only residual-stream paths tolerate `QMatMul`. K/V
    projections consume the input-layernorm output, exactly like Q.
12. **Confine Execution Digests to Offline Evidence**:
    Emit role-typed token digests only in explicitly requested offline evidence. Never log request-derived digests because low-entropy inputs can be guessed.
13. **Complete Candidate Retention in Memory Estimates**:
    Admission estimates must include every state retained by the selected strategy. For `repeated_full`, sum candidates across questions with saturating arithmetic.
14. **Canonical State Keys**:
    Structured state object keys serialize in byte-lexicographic order at every nesting level. Construction order must not change model input. This is `state_first` semantics, not a renderer change.
15. **Explicit Family Profiles**:
    Shared family loaders need the selected profile in both CPU and MLX configs. Keep artifact digests, renderer, calibration, limits, and backend identity bound to that profile. JevK5 and Plumb share a backbone but have separate contracts.
16. **Gemma 4 Attention Stays Banded**:
    gemma4 attention must never materialize dense `seq x seq` score or mask tensors. Sliding layers attend a `window + block` key band and full-attention layers the causal prefix, per `ATTENTION_QUERY_CHUNK` query block; the scratch estimate must stay linear and caps the effective context length through the load-time scratch budget (`OPENKIND_GEMMA4_SCRATCH_BUDGET_MB`). Forwards re-check `FamilyControl` between layers, and only KV donors a later layer actually consumes retain their K/V.

## Verification Commands

```bash
# Check compilation across all targets
cargo check -p openkind-backends

# Run backends unit and parity test suites (Qwen 3.5 reference and surveyed families)
cargo test -p openkind-backends
```

### CUDA feature (CUDA toolchain required at build time)

```bash
cargo check -p openkind-backends --features cuda
cargo clippy -p openkind-backends --features cuda --all-targets -- -D warnings
```

CUDA execution reuses the CPU candle model code with a resolved
`candle_core::Device`; the only feature-gated surface is device resolution
(`src/device.rs`) plus per-engine selection arms, so feature-off builds type
check the device threading. Device selection is `FamilyExecution`
(`Cpu`, `Cuda { device_id }`, `Onnx { device_id }`, `OnnxRocm { device_id }`);
every family engine
exposes `load_with_execution`, and `load` stays the CPU reference path.
Continuation-state arithmetic identity follows the device: the native
backbone emits `candle-cuda-fp32` on CUDA. No CUDA host has run parity yet;
see [`docs/CUDA.md`](../../docs/CUDA.md).

### ONNX features

```bash
cargo check -p openkind-backends --features onnx
cargo clippy -p openkind-backends --features onnx --all-targets -- -D warnings
cargo test  -p openkind-backends --features onnx
```

`ort` runs with `load-dynamic`: no binary downloads at build time, and the
ONNX Runtime library resolves through an explicit setting, `ORT_DYLIB_PATH`,
the executable-relative bundle, then standard paths at
load time (fail closed). Family ONNX adapters live in each family's
`onnx.rs` and must reproduce the candle readout contract exactly; the
artifact contract, required export digest manifest, and per-family applicability are in
[`docs/ONNX.md`](../../docs/ONNX.md).

### ROCm feature (Linux runtime; offline build)

```bash
cargo check -p openkind-backends --features onnx-rocm
cargo clippy -p openkind-backends --features onnx-rocm --all-targets -- -D warnings
cargo test  -p openkind-backends --features onnx-rocm
```

`onnx-rocm` implies `onnx` and only gates the ROCm execution-provider
registration path in `src/onnx/mod.rs` plus `OnnxRocm` selection arms; the
build stays offline and needs no HIP toolchain. Runtime selection follows
the ONNX contract with `FamilyExecution::OnnxRocm { device_id }` and
`backend_id` fragments like `encoder-nli/onnx-rocm:0`; no ROCm host has run
parity yet. Build, placement, detection, and evidence gates are in
[`docs/ROCM.md`](../../docs/ROCM.md).

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

Use the canonical warm-throughput command and comparison rules in
[`openkind-bench/AGENTS.md`](../openkind-bench/AGENTS.md) and
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md). Parity commands remain above.
