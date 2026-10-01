# 2026-10-01 — Raw `decoder-logit-qwen3` controls (0.6B / 1.7B / 4B)

Request-path benchmark records for the three raw direct-logit control
profiles landed for the JevBench board's raw-logit rows (board #21 / #79 /
#81): `decoder-logit-qwen3-06b` (`d900f4af57509fe02e62`),
`decoder-logit-qwen3-17b` (`8119b9271f8d011e7d03`), and
`decoder-logit-qwen3-4b` (`9dfaf11792a8d061b6b8`). Request-path timing
only — **no model-quality claim**; these are untrained controls by
construction (temperature 1.0, declared family renderer, one forward per
question).

Host: Apple M4 Max (14 logical cores, 36 GB), macOS arm64, release build,
commit `9dadff941d4514b1d0ee08b474c484c043d2858c`, host label
`openkind-campaign`, single rep, warm process, model load excluded from
timing.

| Record | Engine | Workload | Rows / groups | p50 total | Throughput | Peak RSS | Model load |
|---|---|---|---|---|---|---|---|
| [summary-decoder-logit-qwen3-06b.json](summary-decoder-logit-qwen3-06b.json) | `decoder-logit-qwen3-06b/cpu-fp32` | [decisions_smoke](../../../crates/openkind-bench/fixtures/decisions_smoke.jsonl) `3a673e…` | 12 / 4 | 7.44 s | 1.613 dec/s | 4.21 GB | 4.9 s |
| [summary-decoder-logit-qwen3-17b.json](summary-decoder-logit-qwen3-17b.json) | `decoder-logit-qwen3-17b/cpu-fp32` | [decisions_smoke](../../../crates/openkind-bench/fixtures/decisions_smoke.jsonl) `3a673e…` | 12 / 4 | 5.22 s | 2.301 dec/s | 11.46 GB | 9.7 s |
| [summary-decoder-logit-qwen3-4b.json](summary-decoder-logit-qwen3-4b.json) | `decoder-logit-qwen3-4b/cpu-fp32` | [decisions_smoke](../../../crates/openkind-bench/fixtures/decisions_smoke.jsonl) `3a673e…` | 12 / 4 | 12.65 s | 0.949 dec/s | 15.00 GB | 23.2 s |

Notes:

- All three profiles execute the shared dense-Qwen3 candle forward
  (execution arithmetic `candle-cpu-fp32-qwen3`) with weights widened to
  FP32 at load; peak RSS tracks roughly 2× the BF16 on-disk size plus
  activations.
- Parity: each profile replays its committed golden fixture
  (`tests/decoder_logit_qwen3_parity.rs`) against the pinned checkpoint
  inside the frozen 0.005 probability gate, including usage and selection
  identity.
- `Qwen/Qwen3-8B` (board #29) stays surveyed-only: fp32 CPU needs roughly
  32 GB of weights alone, which does not fit the reference host.
