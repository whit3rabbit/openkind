# openkind-backends

> Native model backends and parity-checked decision readouts for OpenKind.

`openkind-backends` holds the model-facing half of the [openkind](../../README.md)
decision engine: the native Qwen 3.5 backbone, the surveyed-family adapters
behind the `DecisionEngine` trait from `openkind-engine`, and the seam that
selects where each family executes. It produces typed `Noul`, `Choice`, and
`Score` answers through deterministic readouts. It never generates text.

Candle FP32 CPU execution is the frozen parity oracle. CUDA, ONNX Runtime,
and MLX execution are opt-in candidates; none of them inherits the CPU parity
claim until its own gates run.

## Execution backends

Every family engine exposes `load`, the CPU reference path, and
`load_with_execution`, which takes a [`FamilyExecution`](src/device.rs)
selection. `device::execution_support()` reports what the running binary
actually compiled in, next to the NVML hardware detection from
`openkind_runtime::detect_accelerators`.

| Selection | Feature | Status |
|---|---|---|
| `Cpu` | default | FP32 candle. Checkpoint shards are size- and digest-checked in their read-only source directory, then mmap'd in place. Never copied to temporary storage. |
| `Cuda { device_id }` | `cuda` | Native candle execution on NVIDIA GPUs. Requires the CUDA toolchain (nvcc) at build time; loads fail closed without a driver or a valid ordinal. Kernel numerics can differ, and no CUDA host has run the parity gates yet. See [docs/CUDA.md](../../docs/CUDA.md). |
| `Onnx { device_id }` | `onnx`, `onnx-cuda` | ONNX Runtime through `ort` with `load-dynamic`: builds stay offline, and the shared library is an operator placement through `ORT_DYLIB_PATH` or standard paths. Eight families ship adapters with load-time signature, manifest, and digest checks that fail closed. No export has run the parity gates. See [docs/ONNX.md](../../docs/ONNX.md). |
| `OnnxRocm { device_id }` | `onnx-rocm` (Linux) | ONNX Runtime ROCm execution provider on AMD GPUs. Same offline build and operator-placed library as `onnx`; requires a ROCm-enabled ONNX Runtime build. Loads fail closed when the runtime lacks the provider, and there is no fallback to CPU. No ROCm host has run the parity gates. See [docs/ROCM.md](../../docs/ROCM.md). |
| MLX | `mlx` (macOS arm64) | The Qwen 3.5 backbone plus the laya, encoder-instruct-label, and decoder-logit-qwen35/Plumb family backends. FP32 `ReferenceOps` passes the frozen parity gates; native BF16 is unpromoted. See [docs/MLX.md](../../docs/MLX.md). |

Continuation-state arithmetic identity follows the device: a CUDA load emits
`candle-cuda-fp32` instead of the pinned `candle-cpu-fp32`, so CPU and CUDA
states can never be mixed. Surveyed families carry the same distinction in
their `backend_id` strings.

## Branchable state and scheduling

The Qwen 3.5 continuation state isolates attention KV, DeltaNet recurrent
state, and convolution state. A KV-only clone is incomplete.

Execution strategies are `repeated_full`, `nested_sequential`, and
`nested_batched`. `choose_strategy` picks by the lowest measured crossover
ratio for the pinned profile (2.52), declared `BackendCapabilities`, lane
limits, and a process-memory envelope. The CPU backend declares no vectorized
suffix capability, so shared work plans `nested_sequential`;
`nested_batched` is a lane topology, not a vectorized-compute claim.

## Layout

- `src/qwen35/`: native backbone, digest-locked tokenizer, continuation state,
  execution strategies, score-summary head, and the `Qwen35DecisionEngine`
  wire adapter.
- `src/families/`: surveyed-family loaders and bounded engine adapters, one
  `onnx.rs` per ONNX-capable family. Profile status lives in the
  [family registry](../../docs/families/README.md).
- `src/branch/`: re-export of the backend-neutral `BranchableState` contract
  owned by `openkind-runtime`.
- `src/device.rs`: `FamilyExecution` selection and `execution_support`.
- `src/onnx/`: ONNX Runtime session contract (feature `onnx`).
- `src/proxy_cache/`: distilling cache and training lifecycle; see
  [docs/PROXY_CACHE.md](../../docs/PROXY_CACHE.md).

## Getting started

Default offline tests verify the tracked Phase 3B identities and golden-vector
digests, exact embedding widening, frozen-head probability/policy replay,
renderer semantics, cardinality and input limits. The renderer unit tests use
a deterministic synthetic byte tokenizer. The reference subset's provenance
and file digests are in [its source record](tests/fixtures/qwen35_backbone_phase3b_a047d6802c3f06f085b8/SOURCE.md).

The two pretrained-tokenizer tests and the catalog tokenizer-byte check require
an explicitly supplied artifact. They remain separate qualifications rather
than downloads in the default suite. With the already acquired Qwen3.5-4B-Base
tokenizer (19,989,325 bytes, SHA-256
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`), run:

```bash
export OPENKIND_QWEN35_TOKENIZER=path/to/pinned/tokenizer.json
cargo test --locked -p openkind-backends --test qwen35_tokenizer_parity -- --ignored
cargo test --locked -p openkind-model-store --test catalog -- --ignored
```

These commands verify the artifact's digest, exact exported segment IDs,
renderer limits and catalog size/hash. They do not run full checkpoint or
accelerated inference. Missing or changed artifacts fail qualification.

```bash
cargo check -p openkind-backends
cargo test -p openkind-backends
```

Tests stay offline and never download model assets. Family checkpoint replays
skip unless an `OPENKIND_<FAMILY>_MODEL_ROOT` variable points at a locally
downloaded model. The Phase 3B reference gates load the exported reference
bundle from `research/14_phase3b_backbone_parity_results`, produced by
`research/14_phase3b_backbone_parity.ipynb`.

Optional features:

```bash
cargo clippy -p openkind-backends --features cuda --all-targets -- -D warnings
cargo test  -p openkind-backends --features onnx
export SDKROOT=$(xcrun --show-sdk-path)
cargo test  -p openkind-backends --features mlx
```

The ONNX runtime test skips when no runtime library is discoverable. The MLX
feature builds only on macOS arm64.

## Checkpoint-gated parity examples

The examples below need an already-downloaded pinned checkpoint and never
download model assets.


```bash
# No checkpoint required: scheduler and admission stress at K = 32/64/128/255.
cargo run -p openkind-backends --example qwen35_scheduler_stress

# Full, nested, and batched parity take the same three paths.
cargo run --release -p openkind-backends --example qwen35_full_parity -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results \
  crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8

# Two invocations form the checkpoint-gated fresh-process replay.
cargo run --release -p openkind-backends --example qwen35_parity_probe -- \
  path/to/checkpoint research/14_phase3b_backbone_parity_results \
  persist-save path/to/root-state.bin
cargo run --release -p openkind-backends --example qwen35_parity_probe -- \
  path/to/checkpoint research/14_phase3b_backbone_parity_results \
  persist-replay path/to/root-state.bin \
  crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8

# Representative high-K stress. The second argument is the candidate count.
cargo run --release -p openkind-backends --example qwen35_model_stress -- \
  path/to/checkpoint 64

# Phase 3.8 scheduler measurement: roughly 35-45 minutes at default settings.
cargo run --release -p openkind-backends --example qwen35_scheduler_bench -- \
  path/to/checkpoint research/14_phase3b_backbone_parity_results
```

Every checkpoint-backed command verifies both immutable shards, about 9.3 GB
total, before execution, and each run records sanitized native-run evidence
under `target/verification/native-runs`.

## Status and evidence boundaries

- CPU native parity is established against the pinned profile. Daemon
  load/soak and cancellation evidence is in the
  [native service gate report](../../docs/verification/native-service-gate/2026-09-22-rerun2/README.md).
  Practical high-K latency, model-quality review, and release promotion
  remain open. A cancelled caller keeps its queue and execution permits until
  the blocking native work finishes.
- The MLX path has a forced FP32 vectorized daemon request and bounded
  unified-memory admission evidence in the
  [Phase 3M follow-up](../../docs/verification/phase3m-2026-09-22/README.md).
  Automatic vectorized scheduling and the packed Metal kernel remain opt-in,
  and MLX service load/soak and promotion remain open.
- CUDA, ONNX, and ROCm answers are unpromoted candidates. Their promotion
  gates are listed in [docs/CUDA.md](../../docs/CUDA.md),
  [docs/ONNX.md](../../docs/ONNX.md), and [docs/ROCM.md](../../docs/ROCM.md).
  Throughput numbers never establish task quality; only the labeled-dataset
  and parity evidence in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md) does.

## License

Released under the workspace license, MIT OR Apache-2.0. See
[LICENSE](../../LICENSE).
