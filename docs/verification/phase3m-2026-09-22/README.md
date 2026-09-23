# Phase 3M follow-up evidence, 22 September 2026

This record groups the pinned-base BF16 gate rerun, variable-length FP32
batch parity, unified-memory stress, and MLX daemon smoke. The checked-out subject was
`a5a752ab50efccba2eff0345fc5435c01248d41e`; the worktree was dirty. The
checkpoint was `Qwen/Qwen3.5-4B-Base` at revision
`1001bb4d826a52d1f399e183466143f4da7b741b`, profile
`a047d6802c3f06f085b8`, on the named Apple M4 Max host using Xcode 27.0,
Metal toolchain 32023.921, and MLX 0.32.2.

## BF16 Gate B

Both runs used the frozen probability tolerance `0.005` and the pinned Phase
3B token fixtures and head bundle. Full-sequence BF16 reached maximum
probability delta `0.006099619710620674`, with zero argmax changes and zero
policy changes across 10 candidates. The gate fails on probability parity.
Nested BF16 reached maximum probability delta `0.026580797832947478`, also
with zero argmax and policy changes. It preserved positions, root immutability,
root storage accounting, and sibling isolation. Its cached-versus-full feature
delta (`0.5`) is diagnostic only for BF16 and is not an asserted gate.

The bounded `q4897_a26093` question trace found the first captured mismatch at
the layer 0 `linear_out_projection` boundary. Its gated-Delta input matches
bit-for-bit; the BF16 projection output over 36 suffix rows, shape
`[36, 2560]`, differs by maximum absolute `0.000122070312` (RMS
`9.53987e-7`). The comparison's full and cached runs use different enclosing
row counts. Layer 0's final question row still matches exactly; the small
non-final-row difference is carried into later state and grows through the
following layers. This localizes the first observed split to shape-dependent
BF16 projection behavior, not a demonstrated cache-state corruption. No
arithmetic change or tolerance relaxation was made. BF16 Gate B remains open.

Raw outputs:

- [`gate-b-full.txt`](gate-b-full.txt)
- [`gate-b-nested.txt`](gate-b-nested.txt)
- [`q4897-layer-trace.txt`](q4897-layer-trace.txt)
- [`q4897-layer0-operations.txt`](q4897-layer0-operations.txt) and
  [`q4897-layer0-operations.command.txt`](q4897-layer0-operations.command.txt)
- [`q4897-layer1-operations.txt`](q4897-layer1-operations.txt) and
  [`q4897-layer1-operations.command.txt`](q4897-layer1-operations.command.txt)

The layer 1 operation trace is a follow-on localization after the layer 0
non-final-row mismatch had already been observed. It confirms that the
difference grows through later operations; the layer 0 trace identifies the
first captured boundary.

## FP32 variable-length batch gate

The release-mode `qwen35_mlx_nested_parity` run passed both the existing
sequential-nested frozen gate and a variable-length vectorized batch gate for
the pinned base checkpoint. The added gate uses two questions forked from one
root, with question suffix lengths `[11, 16]` and candidate suffix lengths
`[[12, 13, 13], [20, 20]]`. Question fan-out and both candidate fan-outs
reported `vectorized`; the aggregate physical mode was `vectorized`. Positions,
root storage accounting, root immutability, and sibling isolation passed.

Across the added batch fixture, the maximum probability delta against its
frozen head reference was `1.4643666819136314e-6`, with zero argmax and policy
changes. The largest per-case frozen-head probability delta was
`1.263484928126779e-6`. The feature delta against the corresponding sequential
nested baseline was `3.62396240234375e-5`; this is a diagnostic, not a new
acceptance tolerance. The full example also reports `gate_passed=true` for its
10-candidate frozen full/nested fixture gate. Raw output:
[`batch-fp32-parity.txt`](batch-fp32-parity.txt). The earlier invocation with
incorrect fixture roots is retained separately in
[`batch-fp32-parity-path-error.txt`](batch-fp32-parity-path-error.txt).

The vectorized capability is currently diagnostic-only: the daemon can force
it with `--qwen35-execution nested-batched`, but automatic scheduling remains
per-lane until a matched vectorized-versus-per-lane performance comparison
establishes a useful lane range. No speedup or production promotion is claimed.

## Unified-memory stress

The release probe used the same base revision and profile, FP32
`ReferenceOps`, synthetic repeated token ID `1`, two recovery cycles, a
36-GiB host, and an 8-GiB retained-state cap (22.22% of physical memory). The
Q/K sweep covered Q = 1, 2, 4 and K = 32, 64, 128, 255. Q1/K32 and Q1/K64
were admitted at 3.08 GB and 5.98 GB estimated retained state. Requests above
the cap were rejected before allocation and reduced to the largest admitted
candidate count: K = 92 for Q1, K = 45 for Q2, and K = 22 for Q4. The largest
fallback retained payload was 8.52 GB. The maximum observed process physical
footprint was 23.25 GB during the near-cap Q1/K92 request. The loaded model
baseline itself was about 14.48 GB process physical footprint.

The recovery run alternated a small Q1/K2 request (224.9 MB retained state)
with Q1/K64 (5.98 GB). Across two cycles, requests were admitted, released
state returned to baseline, the MLX inactive cache stayed near its configured
256-MiB limit, and no allocation failed or was rejected. The largest process
physical footprint in recovery was 20.75 GB. These measurements cover the
probe's retained continuation payload and process footprint only; they do not
establish production capacity, concurrent service load, latency, or decision
quality. Raw records are
[`memory-qk-sweep.jsonl`](memory-qk-sweep.jsonl) and
[`memory-recovery.jsonl`](memory-recovery.jsonl).

The two probe invocations were:

```text
cargo run --release --offline -p openkind-backends --features mlx --example qwen35_mlx_memory_stress -- <checkpoint-root> --cycles 2 --skip-recovery
cargo run --release --offline -p openkind-backends --features mlx --example qwen35_mlx_memory_stress -- <checkpoint-root> --cycles 2 --skip-qk-sweep
```

## Daemon request-path smoke

The release MLX daemon build passed with the `mlx` feature. The smoke process
registered `qwen35-native` with `--qwen35-backend mlx-fp32`, forced
`--qwen35-execution nested-batched`, and served the synthetic two-question
request in [`daemon-batch-smoke-request.json`](daemon-batch-smoke-request.json).
The response was HTTP 200 in 2.00 seconds and is retained at
[`daemon-batch-smoke-response.json`](daemon-batch-smoke-response.json). The
daemon logged `strategy="nested_batched"`,
`batch_forward_mode="vectorized"`, `admitted=true`, and `forced=true`; SIGINT
was followed by `openkindd exited cleanly`. The captured run is in
[`daemon-batch-smoke.log`](daemon-batch-smoke.log). This is a bounded request
path smoke, not service load/soak evidence.

Build output is retained in
[`daemon-mlx-build-retry.txt`](daemon-mlx-build-retry.txt); runtime parity and
BF16 traces are recorded above. All runs used the dirty worktree anchored at
`a5a752ab50efccba2eff0345fc5435c01248d41e`, Xcode 27.0, Metal toolchain
32023.921, and MLX 0.32.2. The packed MetalTree candidate remains opt-in: the
prior same-host smoke sweep measured it 14–28% slower than `ReferenceOps`.
