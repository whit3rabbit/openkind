# External Qwen2.5 MLX field-pooling reference

This is an offline benchmark of the field-suffix batching pattern in the
external Python implementation. It is not an OpenKind `DecisionEngine` run,
a Jev request, or evidence that the linked `RLCD` repository contains model
weights. The [family survey](../../families/parallel-constrained-qwen2.md)
records the support boundary.

## Identity and method

- Source: `harshatheg/Qwen-2.5-1B-RLCD` revision
  `2af86848be75847ccb3553b0941cc51d6ef7e4e9` (Python code and presets).
- Checkpoint: `mlx-community/Qwen2.5-1.5B-Instruct-4bit` revision
  `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`, 28-layer Qwen2, MLX
  group-64 4-bit. Local `model.safetensors` SHA-256 is
  `0979f33d1bc58afcf696d13f57977644e7b11a6f0eec3e631d8e9463d18c0717`;
  `tokenizer.json` is
  `a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf`.
- Host: `Mac16,5` (Apple M4 Max, 36 GiB), macOS 26.6.2, Xcode 27.0
  build 27A266a, Python 3.11.13, MLX 0.32.2, mlx-lm 0.31.3.
- Workload: the source's `code_security.json` context and first 4 or all 28
  ordered fields. One field has colliding first candidate tokens in both
  shapes. Unequal suffix lengths require 9 and 60 padded token slots in the
  pooled Q4 and Q28 shapes, respectively.
- Each row below is a fresh process with model load excluded, one untimed
  warmup, then 5 Q4 or 3 Q28 repetitions. Process order was `per_field`,
  `pooled`, `pooled`, `per_field`. Both paths use the same schema-first prompt,
  model, first-token candidate readout, and host-side result extraction.
  `per_field` forks the prefilled cache for each field; `pooled` repeats it
  across fields and submits one right-padded MLX forward. The recorded
  `total_ms` includes prefill, branch setup, suffix forward, and readout.

The unmodified upstream `run_parallel_generation` also loaded the local
checkpoint and returned all 28 fields in a direct smoke run
([`upstream_smoke.json`](upstream_smoke.json)). One output had the source's
heuristic `0.75` probability. That confirms Python MLX execution, not
OpenKind support or calibrated decisions.

## Results

| Fields | Process order | Per-field median (ms) | Pooled median (ms) | Per-field / pooled suffix median (ms) | Physical forwards | Pooled padding |
|---:|---|---:|---:|---:|---|---:|
| 4 | first pair | 161.1 | 141.1 | 45.1 / 26.6 | 5 / 2 | 9 |
| 4 | reverse pair | 161.4 | 141.8 | 45.1 / 26.5 | 5 / 2 | 9 |
| 28 | first pair | 604.0 | 365.0 | 353.0 / 116.1 | 29 / 2 | 60 |
| 28 | reverse pair | 601.0 | 366.6 | 351.1 / 116.4 | 29 / 2 | 60 |

Pooling reduced full timed reference median by about 12% at Q4 and 39% at
Q28. Prefill medians were approximately 112–114 ms at Q4 and 235–239 ms at
Q28, so most of the gain came from suffix execution. Across processes, peak
RSS was 1.160–1.164 GB, including model load and warmup; these small peak
differences do not establish a memory advantage. Individual timings and
scores are in the eight JSON files beside this record.

The first-token option winners matched for all fields in each paired run.
The maximum softmax difference over those first-token option scores was
`0.00334` at Q4 and `0.00370` at Q28. One field has a first-token collision,
so these checks do not prove full-option parity. The source's continuation
path can assign heuristic probabilities and fall back to the first option.
There are no selection, policy, or calibrated-probability gates for Jev here.

## Reproduction and decision

Download both revisions explicitly outside the repository, install the
listed Python packages in an isolated environment, then run the
[`offline reference script`](../../../scripts/bench_qwen25_mlx_field_pool.py):

```bash
python scripts/bench_qwen25_mlx_field_pool.py \
  --checkpoint <local-qwen2.5-checkpoint> \
  --source-root <local-rlcd-source> \
  --preset <local-rlcd-source>/presets/code_security.json \
  --fields 28 --mode pooled --reps 3
```

Repeat with `--mode per_field` and `--fields 4` in the process order above.
The script checks local artifacts before loading and does not download them.

This result supports the potential of field batching for this checkpoint and
prompt shape. It does not justify adding either Hugging Face repository to
OpenKind's installable catalog or claiming Rust MLX support. That requires a
Qwen2.5 quantized loader, exact branch isolation, full-option readout,
offline parity fixtures, a Jev adapter, and fresh full-request benchmarks.
OpenKind's existing Qwen3.5 candidate-pooling diagnostic had a different,
negative result on its pinned FP32 profile.
