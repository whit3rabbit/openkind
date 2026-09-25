# Phase 3M.5 MLX implementation review: 21 September 2026

This record covers a dirty-working-tree review of the existing MLX backend and
the first custom Gated-DeltaNet Metal candidate. The subject baseline is
`e492bab3e4428c0413365ed9d0699ebcb114c320`; every result below also includes
uncommitted changes. It is not a formal clean-commit promotion record.

## Review outcome

The production configuration keeps `ReferenceOps`. The review improved its
runtime, loading, memory accounting, and benchmark fail-closed behavior without
regressing the same-host smoke result. The custom FP32 kernel passes the frozen
model gates but is slower, so it remains an opt-in tuning candidate.

Implemented changes:

- one process-wide GPU stream created through
  `mlx_stream_new_thread_unsafe`, selected explicitly inside every serialized
  `MlxRuntime::execute` scope, and synchronized explicitly;
- a process-wide execution mutex with panic-to-error handling and concurrency
  tests that prove no two MLX closures execute at once;
- decoder tensor filtering before shard allocation/read, plus direct native
  BF16 decoding without an intermediate FP32 vector;
- final-token-only production output, with the 34-stage host trace restricted
  to the opt-in parity path;
- one shared weight-array visitor for materialization and accounting;
- saturating retained-state byte arithmetic in the runtime, Candle, and MLX
  branch implementations;
- a benchmark guard that rejects BF16 nested strategies before loading the
  model;
- generic FP32/BF16 Gated-Delta kernels with scalar or vector gates, optional
  head masks, and a fixed binary reduction tree;
- a packed FP32 `Dk = Dv = 128` step kernel and sequence kernel;
- direct tests for masks, vector gates, BF16 finiteness, packed-versus-generic
  results, exact replay, sequence-versus-step equality, shared-input
  immutability, and segmented continuation equality.

Replay stress found a real race in the first reduction implementation. Lane 0
could overwrite `partial[0]` for the second reduction before every lane had
consumed the first reduction result. The corrected kernels copy the result to a
separate threadgroup value and cross a barrier before scratch reuse. Ten fresh
test processes then replayed the packed sequence result bit-for-bit.

## Pinned-base FP32 gates

The custom candidate used arithmetic identity
`mlx-core-0.32.2/fp32/metal-tree-packed-dk128-v1`; the checkpoint was the
digest-locked `Qwen/Qwen3.5-4B-Base` revision
`1001bb4d826a52d1f399e183466143f4da7b741b`.

`qwen35_mlx_full_parity` passed:

- maximum feature delta: `0.00011444091796875`;
- maximum probability delta: `3.9154827997239794e-7`;
- argmax changes: `0`;
- policy changes: `0`;
- layer 0 maximum absolute diagnostic delta: `3.337860107421875e-6`;
- final-normalization maximum absolute diagnostic delta:
  `0.0000286102294921875`.

`qwen35_mlx_nested_parity` passed:

- maximum cached-versus-full feature delta: `0.0000762939453125`, under the
  `0.0001` guard;
- maximum probability delta: `6.613295700175215e-6`;
- argmax changes: `0`;
- policy changes: `0`;
- root storage, position, root immutability, and sibling isolation gates: pass.

The full and nested examples refuse `--formal` on a dirty tree, so these are
working-tree gates only.

## Same-host throughput decision

Command shape for both arithmetic paths:

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --release \
  -p openkind-bench --features mlx -- score \
  crates/openkind-bench/fixtures/decisions_smoke.jsonl \
  --engine qwen35-mlx-fp32 \
  --bundle-root crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8 \
  --checkpoint-root <pinned-checkpoint-root> \
  --tokenizer research/14_phase3b_backbone_parity_results/backbone_runtime/tokenizer/tokenizer.json \
  --strategies repeated_full,nested_sequential,nested_batched,choose_strategy \
  --reps 1 --host "Mac16,5 Apple M4 Max 36 GiB (named Mac)" \
  --commit e492bab3e4428c0413365ed9d0699ebcb114c320 \
  --output-dir <output-dir>
```

Both runs used one timed repetition per strategy after the harness warmup.
Model load and output writes are excluded from each strategy total.
The raw `openkind-bench/v1` summaries and per-strategy predictions are
retained in
[`docs/benchmarks/2026-09-21-qwen35-mlx-gdn-review/`](../benchmarks/2026-09-21-qwen35-mlx-gdn-review/).

| Arithmetic path | `repeated_full` | `nested_sequential` | `nested_batched` | `choose_strategy` | Peak RSS |
|---|---:|---:|---:|---:|---:|
| `reference-ops` (final default) | 29.998 s | 7.819 s | 7.860 s | 7.755 s | 11,838,046,208 bytes |
| `metal-tree-packed-dk128-v1` (candidate) | 38.279 s | 9.055 s | 8.957 s | 9.060 s | 11,894,095,872 bytes |

The candidate is 1.14 to 1.28 times slower across these cells. This
single-sample smoke result is sufficient to reject promotion, not to establish
a stable performance distribution. The default rerun remains consistent with
the earlier reference-ops record (`29.42/7.91/7.92/7.94 s`) and preserves the
working implementation's speed.

## BF16 boundary

The generic BF16 kernel is implemented and directly tested. A model-backed
candidate run exceeded the frozen gates (maximum probability delta about
`0.0287` with one policy change), so `effective_gated_delta_kernel` falls back
to `ReferenceOps` for native BF16. This does not close BF16 Gate B: the existing
reference-ops BF16 profile also remains outside its frozen acceptance boundary.
No BF16 nested or production-promotion claim is made.

## Verification

Final checks completed during the review:

```text
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
env -u RUST_LOG cargo test --workspace
SDKROOT=$(xcrun --show-sdk-path) cargo check -p openkind-backends --features mlx --all-targets
SDKROOT=$(xcrun --show-sdk-path) cargo clippy -p openkind-backends --features mlx --all-targets -- -D warnings
SDKROOT=$(xcrun --show-sdk-path) cargo test -p openkind-backends --features mlx
SDKROOT=$(xcrun --show-sdk-path) cargo clippy -p openkind-bench --features mlx --all-targets -- -D warnings
SDKROOT=$(xcrun --show-sdk-path) cargo test -p openkind-bench --features mlx
cargo run -p openkind-gen-schemas -- --write
git diff --check
```

The final MLX backend run passed 95 unit tests, 19 integration tests, and doc
tests. The MLX benchmark harness passed 7 tests. The default workspace test
battery, strict Clippy in both configurations, formatting, and diff checks
passed. Schema regeneration produced no schema diff.

## Remaining work

- A fused kernel must beat `ReferenceOps` on representative workloads before
  it can become the default.
- Native cache merge and vectorized multi-lane forward remain open. Current
  `nested_batched` execution is still physically `per_lane`.
- BF16 needs a distinct model-backed correction and a full rerun of Gate B.
- Clean-commit formal parity, high-cardinality memory stress, daemon aliases,
  and queue-inclusive load/soak remain separate gates.

The implementation contract, current limitations, and prioritized enhancement
path are consolidated in [`docs/MLX.md`](../MLX.md).
