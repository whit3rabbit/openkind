# von CPU shape777 record (2026-09-30)

First measured run of the `von` family port (profile
`von:69219703407bd39cca0c`, alias `von:1.1`). Single-sample `--reps 1` run
of the seeded shape777 workload on `openkind-mac-arm64-local` (Apple M4
Max, 14 logical cores), warm process, request-path timing with model load
reported separately. CPU fp32 (candle) over the author's `option_marker.pt`
pickle; no MLX backend exists for this family.

| Backend | Peak RSS | Decisions/s | Input tok/s | Model load |
|---|---:|---:|---:|---:|
| Candle CPU FP32 | 1.83 GB | 2.25 | 493 | 3.3 s |

- Fixture: `/tmp/shape777.jsonl`, 777 rows, 37 groups, sha256
  `be397bfc48209c8f7379d0d76ccbe3d3e7dca3269724928f0868cf872f4c8b01`
  (regenerate with `openkind-bench gen-workload --states 37 --criteria 21
  --seed 291607`).
- Checkpoint: `wfzyx/von` at
  `d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd` (von-1.1, Apache-2.0).
- Commit under measurement: `9dadff9` (dirty tree; von family files are the
  delta).

Reproduce:

```bash
cargo run --release -p openkind-bench -- score /tmp/shape777.jsonl \
  --engine von --model-root "$VON_MODEL_ROOT" --reps 1 \
  --host openkind-mac-arm64-local --commit "$(git rev-parse --short HEAD)" \
  --output-dir bench-output/von --pretty
```

This is request-path timing only, not semantic quality evidence; see
[`../../BENCHMARKS.md`](../../BENCHMARKS.md) for methodology and
[`../../families/von.md`](../../families/von.md) for the parity record.
