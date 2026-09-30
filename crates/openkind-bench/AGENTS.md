# AGENTS.md — openkind-bench

> LLM developer guide for `openkind-bench`. Read this before modifying the benchmark harness.

## Crate Purpose & Boundaries

`openkind-bench` provides the `openkind-bench` binary: an offline scoring and
timing harness over JSONL decision workloads. It exists to:

- Score decision workloads end to end through a real engine path (`score`), recording
  per-row predictions and a provenance summary (`openkind-bench/v1`).
- Allow an explicit `--no-warmup` pass and `--history-aba` exact-request sequence for history probes.
- Generate seeded, deterministic shape-matched workloads (`gen-workload`).
- Keep service-level queue/HTTP measurements in [`scripts/native-service-gate.py`](../../scripts/native-service-gate.py), not in this in-process harness.

Methodology, timing scope, and recorded results are owned by
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) — do not duplicate numbers here.

### Critical Invariants

1. **Dependency Direction**: Depends on `openkind-core`, `openkind-engine`,
   `openkind-backends`, and `openkind-runtime` (all sit at or below the CLI/serve
   layer). It must never be depended on by any other crate.
2. **Fully Offline**: Tests use only the mock engine and checked-in fixtures. The native
   engine loads locally pinned checkpoint/bundle/tokenizer paths and must never download
   anything (workspace invariant 7).
3. **Timing Scope Discipline**: The measured region covers request construction,
   validation, dispatch, and answer extraction. Model load and result-file writes are
   excluded; model-load wall time is reported separately per strategy.
4. **f64 on Reports**: All probabilities and timings serialized into summaries and
   predictions are `f64` (workspace wire-precision rule applies to benchmark evidence
   the same way).
5. **MLX Boundary**: Behind the optional `mlx` cargo feature (macOS arm64, built
   with `SDKROOT=$(xcrun --show-sdk-path)`), `--engine qwen35-mlx-fp32` and
   `--engine qwen35-mlx-bf16` dispatch the MLX backends through the same
   `Qwen35DecisionEngine` request path as the CPU engine
   (`Qwen35Backend` selection in
   [`openkind-backends`](../openkind-backends/AGENTS.md)), and
   `--engine laya-english-mlx-fp32` (and sibling laya profiles),
   `--engine encoder-instruct-label-mlx-fp32`, and
   `--engine decoder-logit-qwen35-mlx-fp32` dispatch the surveyed-family
   MLX backends through those families' engines (`--model-root`, no
   strategy sweep). Default builds (no feature) expose the CPU engines —
   `mock`, `qwen35`, and the thirteen surveyed-family engines — and never
   link MLX.
   MLX runs use the same fixtures, strategy sweep, warmup, and parity
   assertions; their numbers are throughput evidence only — the frozen parity
   gates live in the parity examples. The pinned-base BF16 reference path fails
   the frozen probability gate in both full and nested execution; use
   `--strategies repeated_full` for BF16 throughput comparisons unless a run is
   explicitly labeled as a nested diagnostic. Do not describe the mismatch as
   cache corruption: the current trace first localizes shape-dependent
   projection rounding.
   FP32 benchmarks use the production `ReferenceOps` default. Do not publish
   an opt-in Metal-kernel run without naming its arithmetic identity and
   comparing it against that default on the same fixture and host.

## Key Files & Types

- [`src/main.rs`](./src/main.rs): Entrypoint; `gen-workload` prints one JSON result line,
  `score` and the paired `compare-choice` / `calibrate-choice` diagnostics print
  the summary JSON to stdout (progress goes to stderr).
- [`src/args.rs`](./src/args.rs): Clap parser; `EngineArg` mirrors
  `EngineKind` one-for-one: `mock`, `qwen35`, the surveyed-family engines
  (`decoder-letter`, `encoder-nli`, `encoder-instruct-label`, `decoder-llm`,
  `schema-scorer`, `router-script`, `qwen3-guard`, `kev`,
  `decoder-logit-qwen35`, `laya-english`, `laya-multilingual`,
  `laya-typed-decisions`, `winnow`), and behind the `mlx` feature the MLX
  engines (`qwen35-mlx-fp32`, `qwen35-mlx-bf16`, `laya-*-mlx-fp32`,
  `encoder-instruct-label-mlx-fp32`, `decoder-logit-qwen35-mlx-fp32`).
  `--model-root` is required for surveyed families, `--adapter` for the
  winnow router; plus `--no-warmup`, `--history-aba`, and `parse_strategies`.
- [`src/quality/`](./src/quality/): Offline paired Choice diagnostics over the
  pinned Qwen3.5 probe, separate from timing workloads. [`mod.rs`](./src/quality/mod.rs)
  runs `compare-choice` (nine methods: fitted independent and catalogue
  renderers, four joint renders including the position/code separation arms,
  and three fixed ensembles); [`metrics.rs`](./src/quality/metrics.rs) owns
  proper scores, none recall/false-none, ECE and fixed risk/coverage points;
  [`report.rs`](./src/quality/report.rs) owns source-group bootstrap deltas and
  per-factor order sensitivity; [`calibrate.rs`](./src/quality/calibrate.rs)
  runs `calibrate-choice` (deterministic grid + golden-section NLL fit of a
  temperature and none-logit offset on a calibration partition, locked before
  a disjoint gate is scored; refuses partitions that share source groups or
  row ids). Fixture contracts live in
  [`fixtures/`](./fixtures/) (`joint_choice_diagnostic.jsonl`,
  `joint_reference_card_diagnostic.jsonl`, and the disjoint
  `joint_calibration_diagnostic.jsonl` / `joint_gate_diagnostic.jsonl`
  pair). See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for the commands
  and [`docs/benchmarks/`](../../docs/benchmarks/) for recorded runs.
- [`src/workload.rs`](./src/workload.rs): `WorkloadRow` (flattened `primitive` tag),
  `load_workload`/`parse_workload` (SHA-256 recorded), `state_groups`,
  `build_request`. Choice rows always carry a non-empty `__none__` criterion — one is
  injected when absent so native Choice evaluation keeps semantic-none mass on wire.
- [`src/gen.rs`](./src/gen.rs): Seeded ticket-grid generator (`states × criteria` binary
  noul rows), byte-identical for identical seeds.
- [`src/score/mod.rs`](./src/score/mod.rs): `run_score`/`ScoreArgs`. Native engine sweeps run the
  scheduler's `forced_strategy` diagnostic override per strategy (admission still
  enforced) and assert cross-strategy answer equality per workload. Surveyed-family engines
  execute their single pinned plan without a strategy sweep.
- [`src/score/types.rs`](./src/score/types.rs): `EngineKind`, `native_backend`, `family_identity`,
  and `is_family_engine` helpers.
- [`src/score/summary.rs`](./src/score/summary.rs): Provenance summary builder recording engine slug,
  profile ID, and backbone revision in `openkind-bench/v1` records.
- [`src/tests.rs`](./src/tests.rs): Offline tests (fixture parsing, grouping, `__none__`
  injection, generator determinism, mock end-to-end run).

## Critical Gotchas & Rules

1. **Grouped Requests Share One Root**: The state-first renderer requires every question
   in one request to tokenize to the same root prefix. Grouping is by exact state
   serialization; `--no-group` emits one request per row (the fresh-scoring baseline).
2. **Request Latency Is Not Row Latency**: In grouped mode, every row in a group reports
   its request's latency. Never sum those into per-decision latencies; use
   `decisions_per_second` from the strategy report instead.
3. **Strategy Sweep Is Native-Only**: The mock and surveyed-family engines do not accept
   a strategy sweep (`--strategies` is rejected for family engines, and ignored on mock).
   On Qwen 3.5, `choose_strategy` lets the measured scheduler decide; concrete
   strategy names force the plan through `SchedulerConfig::with_forced_strategy`.
4. **Attribution Is Mandatory for Published Numbers**: `--host` and `--commit` exist so
   recorded results can name their hardware and commit; summaries carry a default
   "unattributed" host label that must be replaced before results are quoted in docs.

## Verification Commands

```bash
cargo check -p openkind-bench
cargo test -p openkind-bench
```

Native timing runs (checkpoint-gated, not part of `cargo test`):

```bash
cargo run --release -p openkind-bench -- score <workload.jsonl> \
  --engine qwen35 --bundle-root <dir> --checkpoint-root <dir> --tokenizer <json> \
  --host "<host label>" --commit <hash>
```

MLX timing runs use the same command with the optional feature and an explicit
MLX engine. Run this only on macOS arm64 with the pinned local checkpoint,
profile bundle, and digest-locked tokenizer:

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --release \
  -p openkind-bench --features mlx -- score <workload.jsonl> \
  --engine qwen35-mlx-fp32 \
  --bundle-root <profile-bundle-dir> \
  --checkpoint-root <pinned-checkpoint-dir> \
  --tokenizer <digest-locked-tokenizer.json> \
  --strategies repeated_full,nested_sequential,nested_batched,choose_strategy \
  --reps 1 --host "<host label>" --commit <hash> \
  --output-dir <benchmark-output-dir>
```

Use `--engine qwen35-mlx-bf16` with `--strategies repeated_full` for the
default throughput comparison. Both full and nested BF16 paths fail frozen
probability tolerance, so nested measurements are diagnostics and must not be
reported as parity-qualified. The current evidence does not establish cache
corruption. The pinned base model for the recorded comparison is
`Qwen/Qwen3.5-4B-Base` at revision
`1001bb4d826a52d1f399e183466143f4da7b741b`. Use the community MLX checkpoint
only for throughput or format-compatibility comparison, not parity claims.
See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for the current recorded
results and attribution requirements.

Use the opt-in download commands in the repository
[`README.md`](../../README.md) to obtain the pinned local checkpoint. Do not add
model downloads to this harness or to its tests.
