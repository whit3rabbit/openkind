# MLX backend

> Implementation guide for the optional Qwen 3.5 MLX/Metal parity backend.
> Benchmark methodology and recorded numbers remain canonical in
> [`BENCHMARKS.md`](BENCHMARKS.md). Milestone status remains canonical in
> [`ROADMAP.md`](ROADMAP.md).

## Status

The backend is available on macOS arm64 behind `--features mlx`. It executes
the pinned `Qwen/Qwen3.5-4B-Base` profile through MLX 0.32.2 while preserving
the same renderer, continuation-state, readout, calibration, and policy
contracts as the Candle CPU oracle.

The current production configuration is deliberately conservative:

- FP32 with `ReferenceOps` is the default and passes the frozen full and
  sequential-nested parity gates on the named M4 Max development host.
- GPU work is serialized process-wide on one explicit cross-thread stream.
- A forced `nested_batched` FP32 `ReferenceOps` run can use a real vectorized
  forward for 2–8 equal-start-position lanes, with right padding and each
  lane's true suffix length. The pinned-base variable-length model gate now
  passes for unequal question and candidate lengths. Automatic scheduling
  remains per-lane until a matched performance comparison supports a useful
  lane range.
- The packed FP32 Metal reduction-tree kernel passes parity but remains opt-in
  because it is slower than `ReferenceOps` on the current smoke workload.
- Native BF16 and the downloaded MLX-community checkpoint are unpromoted
  compatibility or research paths. Neither inherits the FP32 parity claim.
- The daemon exposes `--qwen35-backend mlx-fp32` behind its `mlx` feature.
  A forced vectorized two-question request returned HTTP 200 and logged the
  expected physical forwarding mode. Bounded unified-memory admission and
  recovery runs are recorded. Service load/soak and a clean-commit promotion
  record remain open.

## Pinned runtime and model identity

| Component | Identity |
|---|---|
| Rust binding | `mlx-rs 0.32.0` |
| Vendored mlx-c | `v0.6.0-7-gc74db53` |
| MLX core | `0.32.2` |
| Backbone | `Qwen/Qwen3.5-4B-Base` |
| Backbone revision | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| Profile | `a047d6802c3f06f085b8` |
| Production arithmetic family | `mlx-core-0.32.2/fp32/reference-ops` |
| FP32 kernel candidate | `mlx-core-0.32.2/fp32/metal-tree-packed-dk128-v1` |
| Default inactive cache limit | 256 MiB |

The complete arithmetic identity also includes checkpoint format and the
Xcode/Metal toolchain identity. Rebuilding mlx-c under another toolchain is a
new runtime identity and requires runtime qualification plus the model gates.

## Execution flow

```text
Qwen35DecisionEngine
  -> Qwen35Backend::MlxFp32 or MlxBf16
  -> MlxRuntime and verified checkpoint load
  -> host-resident token embedding rows
  -> 24 DeltaNet layers + 8 grouped-query attention layers
  -> final RMSNorm, final token only
  -> score-summary head, calibration, rejection, and typed answer
```

The backend is an execution replacement, not a different decision model.
[`engine/backbone.rs`](../crates/openkind-backends/src/qwen35/engine/backbone.rs)
selects MLX behind the same backend-neutral executor contract used by Candle.
[`model.rs`](../crates/openkind-backends/src/qwen35/mlx/model.rs) owns load,
prefill, continuation, final normalization, and executor-boundary
materialization.

Production forward returns only the final token's FP32 feature vector to the
host. The 34-stage host trace is isolated in `prefill_trace` so ordinary
serving and benchmarks do not pay for 32 layer readbacks.

## Stream and threading contract

MLX streams and lazy graphs are normally thread-sensitive. The engine can
load on one thread and execute on a blocking-pool worker, so relying on each
caller's default stream is unsafe.

[`MlxRuntime`](../crates/openkind-backends/src/qwen35/mlx/runtime.rs)
therefore enforces one process-wide contract:

1. Construct one GPU stream with `mlx_stream_new_thread_unsafe`.
2. Guard that stream with one process-wide mutex.
3. Enter `mlx_rs::with_stream` for every graph construction and evaluation
   scope.
4. Use the same explicit handle for synchronization, and serialize allocator
   telemetry under the same process-wide lock.
5. Fully materialize weights at load and continuation tensors before they
   leave an executor boundary.
6. Convert a panic inside `execute` into `MlxError` while releasing the lock
   normally.

Backend code must not call MLX operations outside `MlxRuntime::execute`.
Custom kernels fail closed when no runtime-scoped stream is installed. More
than one `MlxRuntime` may exist, but every instance shares the same execution
lock, GPU stream, and process-wide allocator-cache setting.

This intentionally permits only one active MLX model operation per process.
It is the correct first-parity design. Concurrency should be revisited only
with an explicit worker or stream ownership model and evidence that parallel
execution does not break state isolation, memory admission, or numerical
repeatability.

## Checkpoint loading and memory

[`weights/`](../crates/openkind-backends/src/qwen35/mlx/weights/) verifies
checkpoint configuration, tokenizer, index, shard length, and shard digests
before execution. Multi-gigabyte shards remain in their read-only checkpoint
directory and are read in place. They are never copied to temporary storage.

The loader:

- keeps the tied embedding table host-resident and reads only required token
  rows;
- filters vision, MTP, embedding, and output-head payloads before allocating
  decoder arrays;
- streams one required tensor at a time so host staging buffers drop between
  tensors;
- widens checkpoint BF16 exactly for FP32 execution;
- decodes native BF16 directly when the BF16 candidate is requested;
- folds offset layer-normalization weights where required, while leaving the
  DeltaNet normalization weight raw;
- records MLX peak allocation, process RSS high-water mark, and load time.

MLX `Array` cloning copies an immutable reference-counted handle. Branch
isolation still covers all Qwen 3.5 state families: attention KV, DeltaNet
recurrent matrices, convolution windows, logical position, and execution
identity. Tensor-payload accounting is separate from MLX allocator cache,
mapped weights, graph scratch, and process RSS. Admission logic uses
saturating arithmetic and must retain a separately measured process envelope.

## Continuation and batching

[`branch_state.rs`](../crates/openkind-backends/src/qwen35/mlx/branch_state.rs)
implements `BranchableState` and `BranchBatch`. A root state can be forked,
selected, and gathered without mutating the original. Strict fingerprints and
tests cover root immutability and sibling isolation.

`fork_batch` creates isolated lane states. For FP32 `ReferenceOps`, the
explicitly forced `nested_batched` runner advances 2–8 lanes through a real
vectorized suffix forward. It right-pads variable lengths and restores each
lane's true position and state. The pinned fixture pair covers unequal
question and candidate lengths. Automatic scheduling still advertises
per-lane execution while the performance comparison is outstanding.
Consequently:

- `nested_sequential` remains the automatic shared-prefix plan;
- forcing `nested_batched` exercises the vectorized FP32 path for diagnostics;
- parity does not establish that vectorized execution is faster;
- no measured vectorized-versus-per-lane throughput result is available yet.

## Gated DeltaNet implementations

The linear-attention recurrence has two implementations:

### `ReferenceOps`

The production default uses ordinary MLX array operations for the recurrent
update. It processes suffix tokens sequentially and remains the independent
differential comparator for any fused kernel.

### `MetalTree`

[`gated_delta_kernel.rs`](../crates/openkind-backends/src/qwen35/mlx/layers/gated_delta_kernel.rs)
contains independently derived custom Metal kernels:

- generic FP32 and BF16 scalar-gate recurrence;
- optional per-head masks;
- vector-gate variants;
- a fixed binary reduction tree over the key dimension;
- packed FP32 `Dk = Dv = 128` step and whole-suffix sequence kernels.

The reduction tree uses separate threadgroup storage for the completed
reduction value. A barrier prevents lane 0 from reusing shared scratch before
the other lanes consume that value. Fresh-process replay is bit-for-bit stable
after this correction.

The FP32 packed sequence kernel passes both frozen model gates, but launch,
preparation, state-traffic, and synchronization costs outweigh its fused
recurrence on the current workload. FP32 can select it explicitly through
`MlxRuntimeConfig`; native BF16 requests currently fall back to
`ReferenceOps`, even if `MetalTree` is requested, because the model-backed
BF16 candidate failed its probability and policy gate.

## Current benchmark decision

The latest recorded comparison uses the 12-row, four-state smoke fixture on a
named Apple M4 Max with one timed repetition after warmup. Model load and
result writes are excluded from strategy totals.

| Arithmetic path | `repeated_full` | `nested_sequential` | `nested_batched` | `choose_strategy` | Peak RSS |
|---|---:|---:|---:|---:|---:|
| `reference-ops` | 29.998 s | 7.819 s | 7.860 s | 7.755 s | 11,838,046,208 bytes |
| `metal-tree-packed-dk128-v1` | 38.279 s | 9.055 s | 8.957 s | 9.060 s | 11,894,095,872 bytes |

The packed candidate is 1.14 to 1.28 times slower, so it is not promoted.
This is a one-sample rejection result, not a stable performance distribution.
The raw summaries and predictions are checked in under
[`benchmarks/2026-09-21-qwen35-mlx-gdn-review/`](benchmarks/2026-09-21-qwen35-mlx-gdn-review/).
See [`BENCHMARKS.md`](BENCHMARKS.md) for timing semantics, the CPU comparison,
older dispatch records, and the load-inclusive parity timings.

## Correctness gates

Performance does not promote an arithmetic path. A new precision, kernel,
toolchain, checkpoint conversion, state layout, or batching graph must pass:

1. linked runtime and primitive qualification;
2. checkpoint and tokenizer verification;
3. embedding and 34-stage localization diagnostics;
4. full probability, argmax, and policy parity;
5. cached-versus-full continuation parity;
6. root immutability and sibling isolation;
7. fresh-process repeatability;
8. same-host benchmark comparison and memory evidence.

Acceptance remains maximum probability error `0.005`, ordering tolerance
`1e-5`, and unchanged argmax and policy behavior. Hidden-vector differences
are localization diagnostics, not replacement acceptance tolerances.

## Known issues and evidence boundaries

| Area | Current issue | Consequence |
|---|---|---|
| GPU concurrency | One process-wide lock and stream | Correct and race-resistant, but model requests do not overlap on the GPU |
| Physical batching | Forced FP32 `ReferenceOps` lanes use one vectorized forward for 2–8 lanes; model parity is pending | Keep automatic capability disabled and make no throughput claim until the variable-length gate and comparison pass |
| Packed Metal kernel | Correct but 14 to 28 percent slower in the smoke sweep | Remains opt-in |
| Native BF16 reference path | 22 September full probability error `0.0060996`; nested probability error `0.0265808`; both have zero argmax/policy changes | Gate B fails at the frozen `0.005` tolerance; nested state and isolation checks pass, but BF16 remains unpromoted |
| Native BF16 fused path | Model-backed probability error was about `0.0287` with one policy change | Runtime falls back to BF16 `ReferenceOps` |
| MLX-community checkpoint | Loads and executes, but differs from the pinned base model and conversion | Compatibility only, no frozen-model parity or quality claim |
| Exact cross-strategy JSON | FP32 smoke outputs differ by bounded floating-point noise | Use frozen probability, argmax, and policy gates, not byte equality, for numerical acceptance |
| Unified memory | 36-GiB host; 8-GiB retained-state cap; Q/K sweep fallback K=92/45/22 for Q=1/2/4; peak process physical footprint 23.25 GB; two recovery cycles complete | Bounded allocator/admission evidence only; no production capacity or concurrency claim |
| Service integration | `mlx-fp32` daemon alias and forced `nested-batched` request path return HTTP 200 and log `vectorized`; clean SIGINT verified | One request smoke only; no service load/soak or production promotion |
| Evidence status | Latest runtime/kernel record is from a dirty tree | Clean-commit formal rerun is still required |

## Enhancement priorities

1. **Profile before changing the recurrence.** Separate kernel launch count,
   preparation ops, recurrent-state bandwidth, normalization, and output
   projection. The current negative result says recurrence fusion alone is
   insufficient.
2. **Measure the vectorized continuation path.** The FP32 path now batches
   question and candidate lanes in MLX tensors, carries true suffix lengths,
   and restores lane positions. Compare it with the per-lane path over matched
   Q/K/length workloads before widening capability or enabling automatic use.
3. **Fuse around the actual bottleneck.** Evaluate combining input
   preparation, causal-convolution update, gate construction, recurrence,
   normalization, and output projection only where profiling justifies the
   added numerical surface.
4. **Qualify lane limits before advertising automatic use.** The first
   variable-length model gate covers unequal question and candidate lengths
   with 2–8 lanes. Extend differential and matched timing coverage across Q, K,
   masks, and suffix buckets, then enable automatic capability only within
   measured limits.
5. **Extend unified-memory coverage.** The bounded Q/K sweep and two-cycle
   recovery record active MLX bytes, cache bytes, RSS, process footprint, and
   fallback behavior. Add concurrent request/load-soak evidence before setting
   a production capacity limit. Tune the 256 MiB cache only from broader runs.
6. **Isolate BF16 drift.** The 22 September pinned-base trace first observes
   shape-dependent rounding at layer 0 `linear_out_projection`: identical
   inputs produce a maximum output difference of `0.000122070312` across 36
   question rows, with the final row still exact. Any arithmetic correction
   receives a distinct identity and reruns the complete gate ladder.
7. **Extend daemon validation.** `--qwen35-backend mlx-fp32` and its feature
   gate are implemented, and a forced vectorized request passes the bounded
   smoke. Add queue-inclusive load, cancellation/recovery, and memory checks
   before production service claims.

## Build, test, and benchmark

```bash
export SDKROOT=$(xcrun --show-sdk-path)

cargo check -p openkind-backends --features mlx --all-targets
cargo clippy -p openkind-backends --features mlx --all-targets -- -D warnings
cargo test -p openkind-backends --features mlx

cargo clippy -p openkind-bench --features mlx --all-targets -- -D warnings
cargo test -p openkind-bench --features mlx
```

Runtime and model gates:

```bash
cargo run --release -p openkind-backends --features mlx \
  --example qwen35_mlx_qualify -- --formal

cargo run --release -p openkind-backends --features mlx \
  --example qwen35_mlx_full_parity -- --stage full --formal <paths...>

cargo run --release -p openkind-backends --features mlx \
  --example qwen35_mlx_nested_parity -- --formal <paths...>
```

Warm benchmark command and artifact schema are documented in
[`BENCHMARKS.md`](BENCHMARKS.md). Formal examples intentionally refuse a dirty
working tree.

## Evidence

- Current runtime and custom-kernel review:
  [`verification/phase3m5-2026-09-21-working-tree.md`](verification/phase3m5-2026-09-21-working-tree.md)
- Current raw benchmark summaries and predictions:
  [`benchmarks/2026-09-21-qwen35-mlx-gdn-review/`](benchmarks/2026-09-21-qwen35-mlx-gdn-review/)
- Initial benchmark dispatch comparison:
  [`verification/phase3m-2026-09-21-dispatch-recheck.md`](verification/phase3m-2026-09-21-dispatch-recheck.md)
- Runtime qualification provenance:
  [`verification/phase3m-2026-09-20/`](verification/phase3m-2026-09-20/)
