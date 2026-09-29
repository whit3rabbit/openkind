# Registry model benchmark campaign — MLX-preferred execution (2026-09-28)

One `openkind-bench score` campaign over every model in
[`registry/v1/catalog.json`](../../../registry/v1/catalog.json), choosing an
MLX execution path where one exists on this Mac and a CPU path otherwise. The
campaign also introduces request-path telemetry for CPU usage, host hardware,
and per-engine context limits, and aggregates everything into
[`recommendation-data.json`](./recommendation-data.json) for model-recommendation
work.

## Scope and attribution

- Host: `openkind-mac-arm64-local` — Mac16,5, Apple M4 Max, 14 logical cores,
  38,654,705,664 bytes physical memory (recorded per summary in
  `host_hardware`), macOS 26.6.2.
- **Commit attribution correction:** the first runs of this campaign were
  labeled `--commit d80c685` from a stale session-start snapshot. The actual
  HEAD at run time was `b8c80ae` (landed 13:05, before the first run at
  13:40), so the exercised tree content is `b8c80ae` plus this campaign's
  uncommitted telemetry changes. The `qwen35-mlx-bf16` re-run records
  `b8c80ae` directly; the earlier summaries keep their original
  `d80c685` label and must be read with this correction.
- The MLX runtime identity is `mlx-core-0.32.2/fp32/reference-ops` per
  [`docs/MLX.md`](../../MLX.md).
- Concurrent campaign: the [laya MLX backend campaign](../2026-09-28-laya-mlx-campaign/README.md)
  ran in this repository during this campaign (same binary lineage, same
  telemetry schema). Its laya MLX rows are cited below; the two campaigns
  avoided overlapping timed regions after 17:51 except where noted.
- Workload: the standard seeded shape777 fixture
  (`gen-workload --states 37 --criteria 21 --seed 291607`, sha256
  `be397bfc48209c8f7379d0d76ccbe3d3e7dca3269724928f0868cf872f4c8b01`, 777 rows,
  37 state groups, `--reps 1`, grouped by state, warm process), identical to
  the 2026-09-26 surveyed-family and 2026-09-27 laya records.
- Timing scope: request construction, validation, dispatch, and answer
  extraction; model load is reported separately per strategy. These are
  request-path timing records only — none of the profiles has model-quality
  evidence from this run.

## MLX-equivalent survey per registry model

| Registry model | Backbone | MLX equivalent used | Why |
|---|---|---|---|
| `qwen35-state-first:a047d6802c3f06f085b8` | Qwen/Qwen3.5-4B-Base @ `1001bb4d…` | Yes — openkind's parity-qualified `qwen35-mlx-fp32` backend over the pinned base checkpoint | The qwen35 family has a parity-qualified MLX execution path; the fp32 `ReferenceOps` arithmetic passes the frozen parity gates on this host |
| `laya-english:c8ea29bf1e33a343c4b7` | convaiinnovations/laya (ModernBERT-large) | Yes — `laya-english-mlx-fp32` (`families/laya/mlx/`), reading the identical pinned checkpoint shard | The MLX encoder backend landed during this campaign (see the [laya MLX campaign](../2026-09-28-laya-mlx-campaign/README.md)); golden-fixture parity drift 6.5e-6, zero flips. MLX Hub conversions of the bare backbone exist, but openkind reads its own pinned shard directly |
| `laya-multilingual:f4064eb56fb7f7d325e1` | convaiinnovations/laya-multilingual (mmBERT-base) | Yes — `laya-multilingual-mlx-fp32` over the pinned checkpoint | Same landed backend; parity drift 7.2e-6, zero flips. No mmBERT MLX port exists on the Hub |
| `laya-typed-decisions:9d28cfa9567902801ed1` | convaiinnovations/laya-typed-decisions (ModernBERT-large) | Yes — `laya-typed-decisions-mlx-fp32` over the pinned checkpoint | Same landed backend; parity drift 2.5e-6, zero flips |

The `qwen35-mlx-bf16` engine is an additional unqualified candidate probe:
BF16 fails the frozen probability tolerance (see
[`docs/MLX.md`](../../MLX.md)), so its row is throughput evidence only and is
never decision-safe.

## Results

All rows are single-sample `--reps 1` cells on the shape777 workload, warm
process, grouped by state. CPU time and average CPU percent cover the timed
region only (warmup and model load excluded). Peak RSS is the process
high-water mark for the whole run, so it is shared by all strategies of one
process.

### `qwen35-state-first` — MLX FP32 (`qwen35-mlx-fp32`, qualified)

One process ran the four-strategy sweep; peak RSS 12.09 GB;
`cross_strategy_answer_parity_clean: false` — investigated (below).

| Strategy | p50 request | Decisions/s | CPU time | Avg CPU % | Model load |
|---|---:|---:|---:|---:|---:|
| `repeated_full` | 2411.66 s | 0.322 | 3142.8 s | 130.3 | 22.48 s |
| `nested_sequential` | 316.78 s | 2.453 | 258.5 s | 81.6 | 20.93 s |
| `nested_batched` | 314.20 s | 2.473 | 310.0 s | 98.7 | 20.69 s |
| `choose_strategy` | 316.10 s | 2.458 | 259.2 s | 82.0 | 20.74 s |

The scheduler selects a nested plan (2.45 decisions/s); fresh repeated-full
scoring on the same workload costs 7.6× more wall time (0.322 decisions/s).
GPU-bound MLX execution keeps process CPU near or below ~130% of one core,
while the strategy sweep's cross-strategy wire comparison is not byte-exact:
maximum scalar/probability delta 3.12e-05 across all 777 answers, zero
Choice argmax changes, zero selected-option changes. This is the bounded
numerical divergence documented in the 2026-09-21 smoke record (full-wire
JSON comparison, last-bit float scheduling differences), not a parity gate
failure; the frozen model-parity gates live in the MLX parity examples and
remain passed for this arithmetic identity.

### `qwen35-state-first` — MLX BF16 candidate (`qwen35-mlx-bf16`, unqualified)

Single-strategy probe (`repeated_full`, the only strategy the harness
accepts for BF16), committed as `b8c80ae`:

| Strategy | p50 request | Decisions/s | CPU time | Avg CPU % | Model load | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `repeated_full` | 2184.99 s | 0.356 | 2643.5 s | 121.0 | 19.26 s | 4.86 GB |

Against the FP32 `repeated_full` row above, native BF16 is ~1.10× faster on
fresh scoring but holds 2.5× less resident memory (4.86 GB vs 12.09 GB): the
fp32 path widens checkpoint BF16 exactly for execution, while the BF16
candidate decodes native weights. Throughput-only evidence: BF16 fails the
frozen probability tolerance (see
[`docs/MLX.md`](../../MLX.md)) and is never decision-safe.

### `qwen35-state-first` — Candle CPU anchor (`qwen35-native-cpu`)

Single-strategy anchor on the same workload (`choose_strategy`, which selects
a nested plan):

| Strategy | p50 request | Decisions/s | CPU time | Avg CPU % | Model load | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `choose_strategy` | 3531.83 s | 0.220 | 3670.5 s | 103.9 | 16.51 s | 18.62 GB |

Same workload, same strategy decision, same pinned profile: the qualified
MLX FP32 backend is ~11.2× faster than the Candle CPU oracle (316.10 s vs
3531.83 s p50) and peaks at 12.09 GB against 18.62 GB of resident memory.
The CPU run also shows why accelerated execution matters for this profile:
at 0.220 decisions/s the standard 777-decision workload costs about an hour
per pass on 14 CPU cores, while nested branch state retained across each
21-question group pushes process RSS to 18.62 GB (a real constraint on
smaller hosts). `repeated_full` was not run on CPU — two passes would cost
roughly 3 hours for the same fresh-scoring ratio already recorded above.

### laya profiles — Candle CPU fp32 (this campaign) and MLX fp32 (concurrent campaign)

This campaign's CPU rows ran 15:18-15:39, before the concurrent campaign
began. The MLX rows come from the
[laya MLX backend campaign](../2026-09-28-laya-mlx-campaign/README.md)
(`summary-laya-*-mlx-fp32.json`), which ran the same workload, host, and
telemetry schema against the identical pinned checkpoints.

| Engine | p50 request | Decisions/s | CPU time | Avg CPU % | Model load | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `laya-english` (CPU) | 313.67 s | 2.48 | 385.2 s | 122.8 | 1.76 s | 2.75 GB |
| `laya-english-mlx-fp32` | 31.93 s | 24.34 | 21.4 s | 66.9 | 2.77 s | 2.12 GB |
| `laya-multilingual` (CPU) | 133.45 s | 5.82 | 156.4 s | 117.2 | 1.78 s | 2.80 GB |
| `laya-multilingual-mlx-fp32` | 13.78 s | 56.37 | 12.5 s | 90.4 | 3.54 s | 2.94 GB |
| `laya-typed-decisions` (CPU) | 313.35 s | 2.48 | 385.1 s | 122.9 | 1.78 s | 2.76 GB |
| `laya-typed-decisions-mlx-fp32` | 32.68 s | 23.77 | 26.5 s | 81.2 | 2.87 s | 2.13 GB |

The landed MLX backend is ~10.6-11× faster than the candle CPU path on every
laya profile while holding peak RSS at or below the CPU runs, and its
sub-one-core average CPU utilization leaves host cores free. My CPU rows sit
within run-to-run variance of the concurrent campaign's CPU re-runs (2.23 /
5.30 / 2.16 decisions/s) and of the 2026-09-27 records. Request-path timing
only; the upstream model-card accuracies belong to the reference's own
evaluation suites and are not `openkind` measurements.

## Recommendation dataset

[`recommendation-data.json`](./recommendation-data.json) (schema
`openkind-recommendation-data/v1`) aggregates the summaries above: per
registry model, per executable backend — context limits, peak resident bytes,
CPU time and average CPU percent over the measured region, model load time,
p50/p95, and decisions per second — plus host hardware. Regenerate it with:

```bash
python3 scripts/build-recommendation-data.py \
  --campaign 2026-09-28-registry-mlx-campaign \
  --output docs/benchmarks/2026-09-28-registry-mlx-campaign/recommendation-data.json \
  docs/benchmarks/2026-09-28-registry-mlx-campaign/summary-*.json
```

Interpretation notes for recommendation work:

- `peak_resident_bytes` is the process high-water mark including mmap'd
  checkpoint shards and forward scratch; the qwen35 profile also carries a
  scheduler retained-state budget separate from this observation.
- `cpu_time_seconds` / `avg_cpu_percent` cover the timed region only
  (warmup and model load excluded). GPU-bound MLX execution shows low CPU
  percent; CPU-bound candle execution saturates cores. Percentages exceed
  100 with multithreading.
- `context` records each engine's frozen per-sequence token budget. The
  qwen35 profile fails closed above 1,792 tokens (truncation is forbidden);
  laya profiles budget the encoder sequence and
  the choice head separately.
- Single-sample `--reps 1` cells, like all prior records in this series.

## Files

- `summary-qwen35-mlx-fp32.json` + `predictions-qwen35-mlx-fp32-<strategy>.jsonl`
  — the qualified four-strategy MLX FP32 sweep.
- `summary-qwen35-mlx-bf16.json` + `predictions-qwen35-mlx-bf16-repeated_full.jsonl`
  — the unqualified BF16 candidate probe (commit `b8c80ae`).
- `summary-qwen35.json` + `predictions-qwen35-choose_strategy.jsonl` — the
  Candle CPU anchor.
- `summary-laya-<profile>.json` + `predictions-laya-<profile>-laya-<profile>.jsonl`
  — the three laya CPU re-runs.
- MLX rows for the laya profiles live in the
  [laya MLX backend campaign](../2026-09-28-laya-mlx-campaign/README.md) and
  are referenced (not duplicated) by [`recommendation-data.json`](./recommendation-data.json).
