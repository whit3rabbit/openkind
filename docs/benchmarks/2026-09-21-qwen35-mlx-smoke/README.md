# 2026-09-21 — qwen35-MLX smoke sweep (first `opendecision-bench` MLX dispatch)

First recorded MLX run through the full `opendecision-bench` request path:
12 decisions (4 states × 3 primitives) through `Qwen35DecisionEngine` with
`Qwen35Backend::MlxFp32` — render, state-first tokenization, scheduling,
MLX backbone execution, readout, answer mapping — under each execution
strategy. Same fixture, host, and methodology as the CPU smoke record in
[`2026-09-20-qwen35-smoke/`](../2026-09-20-qwen35-smoke/), so the two runs
are directly comparable. **Smoke-scale evidence**: single sample per
strategy on a short-state fixture; quote ratios, not absolutes.

- Schema: `opendecision-bench/v1` (`summary-qwen35-mlx-fp32.json`)
- Engine: `qwen35-mlx-fp32`, profile `a047d6802c3f06f085b8`,
  model revision `1001bb4d826a52d1f399e183466143f4da7b741b`,
  bundle version `2ij.2.0`, `mlx-core-0.32.2/fp32/reference-ops`
  (mlx-rs 0.32.0, vendored mlx-c `v0.6.0-7-gc74db53`)
- Build: `--features mlx` behind
  `SDKROOT=$(xcrun --show-sdk-path)`, release profile
- Fixture: `crates/opendecision-bench/fixtures/decisions_smoke.jsonl`,
  SHA-256 `3a673e843690b942658b4c9de6cc594770185756098d356e78dcd3efd5cffeeb`,
  12 rows grouped per state (4 groups)
- Host: Mac16,5 Apple M4 Max 36 GiB (named Mac), warm process, untimed warmup
- Commit: `ddbc5dd7e46bb1318bb98af8b0d924d78d4452ee` with the (then) uncommitted
  working-tree changes that added MLX engine/bench dispatch — a dirty-tree
  record, like the same-day Phase 3M working-tree verification note
- Timing scope: request construction + dispatch + answer extraction; model
  load (~22–23 s per strategy instance, including weight materialization)
  excluded and reported separately
- Digest-locked tokenizer: vendored Phase 3B copy
  (`research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z/backbone_runtime/tokenizer/tokenizer.json`,
  SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`)
  — the HF checkpoint download's `tokenizer.json` is a different export and
  fails the engine's digest lock

## Results (single sample per strategy)

| Strategy | Total (s) | Decisions/s | Speedup vs repeated_full | CPU smoke same cell |
|---|---:|---:|---:|---:|
| `repeated_full` | 31.64 | 0.379 | 1.00× | 124.88 s |
| `nested_sequential` | 8.32 | 1.442 | 3.80× | 73.64 s |
| `nested_batched` | 8.31 | 1.444 | 3.81× | 69.99 s |
| `choose_strategy` | 8.32 | 1.442 | 3.80× | 73.70 s |

Against the CPU smoke record on the same fixture and host: the shared-state
strategies run in ~8.3 s versus ~70–74 s on CPU (≈ 8.4–8.9×), and
`repeated_full` in 31.6 s versus 124.9 s (≈ 3.9×). This is the per-lane
reference-ops backend — no fused kernel, no vectorized batch forward — so
the gap is ordinary Metal array ops versus Candle CPU, and `nested_batched`
gains nothing over `nested_sequential` beyond noise, as expected for a
per-lane backend. The measured scheduler selected the shared path, matching
`nested_sequential` latency. Peak process RSS at summary time: 12.02 GB.

## Cross-strategy answer parity flag

`cross_strategy_answer_parity_clean: false` for this run. The assertion
compares exact wire answers across strategies; the CPU engine produces
bit-identical answers under all four strategies, while MLX execution shapes
round differently (full-sequence graphs versus suffix continuations over
forked state). Quantified from the prediction files: maximum probability
delta `1.68e-05`, maximum confidence delta `1.46e-05`, **zero** selected
option / argmax changes — the same cached-versus-full divergence the Phase
3M.4 parity gate measured on candidate features (`7.25e-05`, within the
Phase 3B `1e-4` guard) and far inside the frozen `0.005` probability
tolerance. CPU runs keep asserting exact equality; MLX runs should be read
with this bounded numerical divergence in mind until a backend-aware parity
bound is defined.

## Interpretation boundaries

- Warm-process throughput on one fixture, single sample per cell: quote
  ratios, not absolute latencies.
- This record is throughput evidence only. It does not change the Phase 3M
  gate status: FP32 parity stays qualified through the parity examples, and
  BF16 Gate B remains open.
- Sharing gains remain bounded by the short states (same caveat as the CPU
  record).
- CPU parity and this run do not constitute release promotion.

## BF16 probe (`repeated_full` only)

`qwen35-mlx-bf16` ran the same fixture restricted to `repeated_full`:
the known BF16 nested-continuation defect (layer-4 cache-dtype failure after
the first attention block, Phase 3M.6) fail-closes the nested strategies, so
no BF16 nested throughput claim exists. Result: `45.07 s` total
(`0.266` decisions/s), model load `21.4 s`, peak RSS `8.59 GB`
(`summary-qwen35-mlx-bf16.json`). Candidate-profile timing only: BF16's
probability parity gate is open (`5.4572e-03` > `0.005`). The warm BF16
`repeated_full` cell ran ~1.4× *slower* than FP32 (`45.07 s` versus
`31.64 s`) despite half-size weights; the reference-ops path spends its time
in per-token recurrence over small ops, where BF16 gains nothing and
precision conversions cost. This is an implementation observation for the
3M.5 fused-kernel planning, not a tuned profile.
