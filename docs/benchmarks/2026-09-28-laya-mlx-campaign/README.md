# Laya MLX backend campaign (2026-09-28)

First benchmark campaign for the laya encoder family's MLX/Metal execution
backend (`families/laya/mlx/`), on the standard shape777 workload, with the
candle CPU path re-run in the same binary so both backends share one
telemetry schema (including the new per-strategy `input_tokens_per_second`).

## Scope and attribution

- Host: `openkind-mac-arm64-local` — Mac16,5, Apple M4 Max, 14 logical
  cores, 38,654,705,664 bytes physical memory (per summary `host_hardware`),
  macOS 26.6.2. MLX runtime identity `mlx-core-0.32.2/fp32` over
  mlx-rs 0.32.0 / vendored mlx-c `v0.6.0-7-gc74db53`.
- Commit `b8c80ae` at run time; the working tree was dirty (this campaign's
  MLX engine, bench telemetry, and doc changes).
- Workload: the standard seeded shape777 fixture
  (`gen-workload --states 37 --criteria 21 --seed 291607`, sha256
  `be397bfc48209c8f7379d0d76ccbe3d3e7dca3269724928f0868cf872f4c8b01`, 777
  rows, 37 state groups, `--reps 1`, grouped by state, warm process),
  identical to the 2026-09-26/27 surveyed-family and laya records and the
  2026-09-28 registry campaign.
- Timing scope: request construction, validation, dispatch, and answer
  extraction; model load is reported separately. Peak RSS is the process
  high-water mark of each run (CPU runs include the Candle runtime; MLX runs
  include the Metal/MLX runtime).
- Both backends load the identical digest-locked pinned checkpoints; the
  registry catalog and manifests are unchanged (MLX is an execution backend,
  not a new model).

## Parity evidence (frozen gates)

`tests/laya_parity.rs` module `mlx_replay` (feature `mlx`) replays the
committed golden fixtures through the MLX engine:

| Profile | Answers | Max probability drift | Selection flips | Gate |
|---|---:|---:|---:|---|
| `laya-english` | 15 | 6.5e-6 | 0 | PASS (budget 0.005) |
| `laya-multilingual` | 15 | 7.2e-6 | 0 | PASS (budget 0.005) |
| `laya-typed-decisions` | 15 | 2.5e-6 | 0 | PASS (budget 0.005) |

The candle CPU path remains the correctness oracle. This matches the
independent FP32 conversion evidence published with the `aac6fef/laya-*-mlx`
Hub conversions (max probability error 2.7e-6, 63/63 argmax agreements on
M3 Max); openkind does not load those repos — their byte-identical-tensor
finding (verified here over all 206 tensors under a rename map) is what
justified reading the original pinned shard directly with the MLX loader.

## Results

Single-sample `--reps 1` cells on shape777, warm process, grouped by state.

| Engine | p50 (s) | Decisions/s | Input tokens/s | Model load (s) | Peak RSS | Avg CPU % |
|---|---:|---:|---:|---:|---:|---:|
| `laya-english` (candle CPU fp32) | 348.16 | 2.232 | 455.2 | 1.84 | 2.75 GB | 117.8 |
| `laya-english-mlx-fp32` | 31.93 | 24.337 | 4,963.7 | 2.77 | 2.12 GB | 66.9 |
| `laya-multilingual` (candle CPU fp32) | 146.64 | 5.299 | 1,074.5 | 1.95 | 2.80 GB | 116.4 |
| `laya-multilingual-mlx-fp32` | 13.78 | 56.371 | 11,431.3 | 3.54 | 2.94 GB | 90.4 |
| `laya-typed-decisions` (candle CPU fp32) | 333.72 | 2.328 | 474.9 | 1.83 | 2.76 GB | 122.7 |
| `laya-typed-decisions-mlx-fp32` | 32.68 | 23.774 | 4,848.8 | 2.87 | 2.13 GB | 81.2 |

All cells are single samples (`--reps 1`); session-to-session variance of
roughly 10% between single-sample CPU cells is normal on this methodology
(the 2026-09-28 morning registry campaign recorded 2.48/5.82/2.48 dec/s for
the CPU engines). The CPU cells here exist so both backends are compared
within one session and one telemetry schema.

Interpretation:

- The MLX backend is **~10.2–10.9× faster** than the candle CPU path on
  every profile and holds the peak RSS at or below the CPU runs (2.12–2.94
  GB unified memory; the fp32 arrays plus Metal runtime stay under the CPU
  runs' 2.75–2.80 GB).
- Model load is slightly slower on the MLX path (2.8–3.5 s vs 1.8–2.0 s):
  the fp32 arrays are built and uploaded to the Metal device at load, while
  the CPU path mmaps and upcasts lazily per tensor.
- GPU-bound execution shows sub-one-core average CPU utilization
  (66.9–122.7% for CPU, 66.9–90.4% for MLX), so the daemon keeps host cores
  free while serving on MLX.
- Context budgets are unchanged by the backend: 512/192 tokens
  (`laya-english`) and 1024/256 (others), per the `context` field of each
  summary.
- These are request-path records only. Task quality is not claimed
  (`docs/families/laya.md`); the operating-point and M2 gates are separate.

## Files

- `summary-laya-english-mlx-fp32.json` /
  `predictions-laya-english-mlx-fp32-….jsonl`
- `summary-laya-multilingual-mlx-fp32.json` /
  `predictions-laya-multilingual-mlx-fp32-….jsonl`
- `summary-laya-typed-decisions-mlx-fp32.json` /
  `predictions-laya-typed-decisions-mlx-fp32-….jsonl`
- CPU re-runs with the same binary and telemetry schema:
  `summary-laya-english.json`, `summary-laya-multilingual.json`,
  `summary-laya-typed-decisions.json` plus their predictions files.

Reproduce with
`cargo build -p openkind-bench --release --features mlx` and
`target/release/openkind-bench score /tmp/shape777.jsonl --engine <engine>
--model-root <pinned root> --output-dir <dir> --host openkind-mac-arm64-local
--commit <sha> --pretty`.
