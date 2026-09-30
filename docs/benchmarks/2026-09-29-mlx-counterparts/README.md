# MLX counterpart campaign — encoder-instruct-label and decoder-logit-qwen35 (2026-09-29)

One `openkind-bench score` campaign adding MLX fp32 execution backends for
the two surveyed families whose backbones already had parity-verified MLX
implementations (GLiClass → shared ModernBERT body; JevK5 → shared Qwen3.5
backbone), plus a candle CPU re-run of encoder-instruct-label for a
same-session comparison.

## Scope and provenance

- Workload: standard shape777 (`be397bfc…`, 777 rows, 37 state groups,
  seed 291607), request-path timing, model load reported separately.
- Host: `openkind-mac-arm64-local` — Apple M4 Max, 14 logical cores, 36 GiB
  unified memory, macOS arm64.
- Arithmetic: MLX fp32 (`mlx-rs 0.32.0` / MLX core 0.32.2), ReferenceOps;
  candle CPU fp32 remains the correctness oracle.
- Checkpoints: pinned digests verified in place —
  `knowledgator/gliclass-modern-base-v3.0` @ `ac369222…` (FP32 shard) and
  `alibiserikbay/JevK5` @ `c4f7fdb3…` (BF16 shard widened exactly on load).
- Parity gates (frozen 0.005 probability tolerance, zero selection flips),
  golden-fixture replay under `--features mlx`:
  - encoder-instruct-label: 15 answers, max |Δp| = 4.487e-6, 0 flips.
  - decoder-logit-qwen35: 9 answers, max |Δp| = 1.003e-6, 0 flips.
- Working-tree record; a clean-commit rerun remains open, consistent with
  prior MLX campaign practice.

## Results (single sample, `--reps 1`, warm process)

| Engine | Decisions/s | Input tok/s | Peak RSS | Model load |
|---|---:|---:|---:|---:|
| `encoder-instruct-label` (candle CPU) | 4.73 | 1,013 | 1.27 GB | 1.23 s |
| `encoder-instruct-label-mlx-fp32` | 75.42 | 16,136 | 1.02 GB | 1.46 s |
| `decoder-logit-qwen35-mlx-fp32` | 0.43 | 117 | 6.98 GB | 20.2 s |

The decoder-logit-qwen35 candle CPU row is not re-run here; the family's
recorded CPU numbers live in
[`2026-09-27-decoder-logit-qwen35/`](../2026-09-27-decoder-logit-qwen35/)
(0.16 decisions/s, 8.04 GB peak RSS, 15.2 s load on a smoke-scale record).
The MLX gain for that family is therefore roughly 2.7×, far below the
continuation-aware state-first engine's: each pass is an independent full
forward over a 1–2.5k-token JSON payload with no state reuse, so the GPU
wins less than it does on nested strategies. The two CPU cells are
different-scale records and the comparison is order-of-magnitude only.

No model-quality claim: request-path timing only.
