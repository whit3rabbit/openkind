# Phase 3M MLX dispatch benchmark recheck: 21 September 2026

This is a fresh dirty-working-tree benchmark recheck after adding MLX backend
selection to `Qwen35DecisionEngine` and `openkind-bench`. It compares the
same 12-row smoke fixture and the same four execution strategies on the named
M4 Max. The CPU and pinned MLX rows load the same frozen
`Qwen/Qwen3.5-4B-Base` checkpoint, so they are the backend comparison. The
community row uses a different source model and is throughput evidence only.

## Inputs and scope

- Subject commit: `ddbc5dd7e46bb1318bb98af8b0d924d78d4452ee`, with uncommitted
  MLX dispatch changes in the working tree.
- Host: Mac16,5 Apple M4 Max, 36 GiB.
- Fixture: `crates/openkind-bench/fixtures/decisions_smoke.jsonl`, 12 rows,
  4 state groups, SHA-256
  `3a673e843690b942658b4c9de6cc594770185756098d356e78dcd3efd5cffeeb`.
- Timing: one warm-process timed repetition per strategy. Model load and output
  writes are excluded from the strategy totals and model load is reported
  separately. These are smoke-scale measurements, not production throughput.
- MLX runtime: MLX core `0.32.2`, `mlx-rs 0.32.0`, vendored mlx-c
  `v0.6.0-7-gc74db53`, Xcode 27.0, Metal toolchain 32023.921.

The command shape was:

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --release \
  -p openkind-bench --features mlx -- score \
  crates/openkind-bench/fixtures/decisions_smoke.jsonl \
  --engine qwen35-mlx-fp32 \
  --bundle-root crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8 \
  --checkpoint-root <pinned-or-community-checkpoint> \
  --tokenizer research/OpenKind_Phase3B_BackboneParity_20260920T152206Z/backbone_runtime/tokenizer/tokenizer.json \
  --strategies repeated_full,nested_sequential,nested_batched,choose_strategy \
  --reps 1 --host "Mac16,5 Apple M4 Max 36 GiB (named Mac)" \
  --commit ddbc5dd7e46bb1318bb98af8b0d924d78d4452ee \
  --output-dir <output-dir>
```

## Warm request results

| Backend and checkpoint | `repeated_full` | `nested_sequential` | `nested_batched` | `choose_strategy` | Peak RSS | Exact cross-strategy parity |
|---|---:|---:|---:|---:|---:|---|
| Candle CPU, pinned base | 119.22 s | 70.34 s | 71.41 s | 68.83 s | 10.08 GB | true |
| MLX FP32, pinned base | 29.42 s | 7.91 s | 7.92 s | 7.94 s | 11.66 GB | false |
| MLX FP32, community export | 29.57 s | 7.91 s | 7.89 s | 7.88 s | 11.92 GB | false |

Per-strategy model loads were `17.69–17.92 s` for CPU, `22.08–22.93 s` for
pinned MLX, and `21.71–21.98 s` for the community export. Relative to CPU,
pinned MLX was `4.05x` faster for repeated-full, `8.89x` for
nested-sequential, `9.02x` for nested-batched, and `8.67x` for the measured
scheduler choice. The community export was within roughly `0.4%` of pinned MLX
on these single-sample cells.

## Output comparison

For `choose_strategy`, pinned MLX versus CPU had maximum probability delta
`1.7687e-05`, maximum scalar delta `1.8477e-05`, and zero Choice selection
changes. The summary's `cross_strategy_answer_parity_clean: false` is caused
by exact JSON comparison between full-sequence and continuation execution
shapes. It is a bounded numerical consistency result, not a frozen model-gate
failure.

The community export differed from CPU on three Choice selections:

| Question | CPU | Community MLX |
|---|---|---|
| `t02.route` | `billing` | `account_security` |
| `t03.route` | `account_security` | `billing` |
| `t04.route` | `billing` | `account_security` |

Its maximum probability delta against CPU was `0.91194`. This is consistent
with the previously recorded community model-parity failure and must not be
interpreted as an MLX dispatch or kernel regression. The benchmark summary's
`model_revision` remains the fitted head's pinned-base revision; the actual
community checkpoint is identified by the adapter state identity and the
community checkpoint revision in the model comparison record.

## Boundaries

- The pinned MLX FP32 parity examples remain the acceptance gate; this smoke
  benchmark is throughput evidence.
- The MLX reference-ops path is per-lane, not a fused or vectorized Metal
  implementation. `nested_batched` therefore has no expected compute-batching
  advantage over `nested_sequential`.
- Native BF16 remains separately gated. Its nested continuation defect and
  probability tolerance failure are recorded in the Phase 3M working-tree
  parity note.
