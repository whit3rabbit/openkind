# 2026-09-30: JevBench v1.5.4 expansion: `plumb-4b` and `decider-4b` profiles

Request-path benchmark records for the two profiles added to close the
official top-10 gap on the JevBench v1.5.4 board
([`benchmarkheaven.com/jev-models`](https://benchmarkheaven.com/jev-models)):
`plumb-4b` (`c1f080794d38e94a0bc2`, official #5) and `decider-4b`
(`0529bf6f2bed84641701`, official #7). These are the board's official ranks;
its capability ranking places the systems at #8 and #11. Request-path timing
only, **no model-quality claim**; the accuracy line below is a vendored synthetic
diagnostic, not a leaderboard measurement.

Host: Apple M4 Max (14 logical cores, 36 GiB), macOS arm64, release build,
commit `9dadff941d4514b1d0ee08b474c484c043d2858c`, host label
`openkind-campaign`, single rep, warm process, model load excluded from
timing.

The recorded commit predates the Plumb and Decider adapters. These records
therefore include later source changes; the exact measured source diff is
not archived here, so the commit alone cannot reproduce these runs.

| Record | Engine | Workload | Rows / groups | p50 total | Throughput | Peak RSS | Model load |
|---|---|---|---|---|---|---|---|
| [summary-plumb-4b-smoke.json](summary-plumb-4b-smoke.json) | `plumb-4b/cpu-fp32` | [decisions_smoke](../../../crates/openkind-bench/fixtures/decisions_smoke.jsonl) `3a673e…` | 12 / 4 | 309.1 s | 0.039 dec/s | 7.84 GB | 16.8 s |
| [summary-plumb-4b-choice-diagnostic.json](summary-plumb-4b-choice-diagnostic.json) | `plumb-4b/cpu-fp32` | [joint_choice_diagnostic](../../../crates/openkind-bench/fixtures/joint_choice_diagnostic.jsonl) `279dc6…` | 96 / 78 | 617.4 s | 0.155 dec/s | 7.83 GB | 22.1 s |
| [summary-decider-4b-smoke.json](summary-decider-4b-smoke.json) | `decider-4b/cpu-fp32` | [decisions_smoke](../../../crates/openkind-bench/fixtures/decisions_smoke.jsonl) `3a673e…` | 12 / 4 | 211.7 s | 0.057 dec/s | 7.91 GB | 16.9 s |

Notes:

- Both profiles execute the shared native FP32 CPU Qwen3.5 backbone
  (execution arithmetic `candle-cpu-fp32-qwen35-text`); full-forward-per-pass
  compute dominates, and the smoke workload's long JSON states make its
  per-decision cost several times the diagnostic's shorter rows.
- `plumb-4b` serves 2–16 options in one calibrated read (jevk5 v0.2.0
  single-read contract, temperatures 2.07 choice/noul and 1.2 score).
  `decider-4b` serves up to 255 options with isolated Score level rows
  (per-type temperatures 1.11 / 1.56 / 1.287 from `decider_config.json`).
- Reported parity: both profiles replayed their committed golden fixtures
  (`crates/openkind-backends/tests/decoder_logit_qwen35_parity.rs`,
  `crates/openkind-backends/tests/decider_parity.rs`) against the pinned
  checkpoints, and Plumb also replayed through MLX inside the frozen 0.005
  probability gate. Replay logs are not archived in this record. These tests
  are checkpoint-gated; a normal offline test pass skips the replay.
- Diagnostic accuracy (vendored 96-case `joint_choice_diagnostic`,
  gold stated in the state document): reported `plumb-4b` 96/96. The
  diagnostic predictions are not archived here; its summary binds their
  digest but contains no accuracy metric. This result cannot be independently
  checked from this record. It is a fixture diagnostic, not JevBench scoring.
