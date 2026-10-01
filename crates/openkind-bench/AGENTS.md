# AGENTS.md — openkind-bench

> LLM developer guide for `openkind-bench`. Read this before modifying the benchmark harness.

## Crate Purpose & Boundaries

`openkind-bench` provides the `openkind-bench` binary: an offline scoring and
timing harness over JSONL decision workloads. It exists to:

- Score decision workloads end to end through a real engine path (`score`), recording
  per-row predictions and a provenance summary (`openkind-bench/v1`).
- Allow an explicit `--no-warmup` pass and `--history-aba` exact-request sequence for history probes.
- Generate seeded, deterministic shape-matched workloads (`gen-workload`).
- Acquire pinned public evaluation datasets and report labeled accuracy
  (`dataset pull/pin/build/eval`); acquisition is explicit and networked,
  evaluation reads only the verified local install.
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
## MLX Qualification Rules

- MLX is optional and runs on macOS arm64. Build with `--features mlx` and `SDKROOT=$(xcrun --show-sdk-path)`.
- Qwen MLX uses the normal request path. Surveyed-family MLX runs use `--model-root` and one pinned plan, without a strategy sweep.
- MLX throughput does not establish parity. Native Qwen gates live in backend examples. Surveyed-family gates are checkpoint-dependent integration tests.
- Use `repeated_full` for BF16 throughput comparisons. Label nested runs as diagnostics until the frozen probability gate passes.
- Do not describe the current BF16 mismatch as cache corruption. The trace localizes shape-dependent projection rounding.
- FP32 uses production `ReferenceOps` by default. Name and compare opt-in Metal arithmetic against that default on the same fixture and host.

## Key Files & Types

- [`src/main.rs`](./src/main.rs): Entrypoint; `gen-workload` prints one JSON result line,
  `score` and the paired `compare-choice` / `calibrate-choice` diagnostics print
  the summary JSON to stdout (progress goes to stderr).
- [`src/args.rs`](./src/args.rs): `EngineArg` mirrors available engines. Surveyed families require `--model-root`; winnow also requires `--adapter`. `--no-warmup` and `--history-aba` control diagnostic runs.
- [`src/quality/`](./src/quality/): Offline paired Choice diagnostics, separate from timing workloads.
- [`src/quality/calibrate.rs`](./src/quality/calibrate.rs): Fits calibration values and rejects calibration/gate partitions that share source groups or row IDs.
- Quality methodology and fixture evidence live in [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) and [`docs/benchmarks/`](../../docs/benchmarks/).
- [`src/workload.rs`](./src/workload.rs): `WorkloadRow` (flattened `primitive` tag),
  `load_workload`/`parse_workload` (SHA-256 recorded), `state_groups`,
  `build_request`. Choice rows always carry a non-empty `__none__` criterion — one is
  injected when absent so native Choice evaluation keeps semantic-none mass on wire.
- [`src/gen.rs`](./src/gen.rs): Seeded ticket-grid generator (`states × criteria` binary
  noul rows), byte-identical for identical seeds.
- [`src/score/mod.rs`](./src/score/mod.rs): Native sweeps force each strategy while preserving admission and check answer equality. Surveyed-family engines use one pinned plan.
- [`src/score/types.rs`](./src/score/types.rs): `EngineKind`, `native_backend`, `family_identity`,
  and `is_family_engine` helpers.
- [`src/score/summary.rs`](./src/score/summary.rs): Records family `engine`, profile/backend `engine_variant`,
  profile ID, and checkpoint revision. Its offline regression tests need no model assets.
- [`src/dataset/`](./src/dataset/): Materialization, evaluation, and metrics. [`templates.rs`](./src/dataset/templates.rs)
  freezes Bonn MIT wording. Changes need new versions.
  [`eval.rs`](./src/dataset/eval.rs) binds predictions to gold by digest.
  `openkind-dataset-eval/v1` reports provide quality evidence.
- [`src/tests.rs`](./src/tests.rs): Offline tests (fixture parsing, grouping, `__none__`
  injection, generator determinism, mock end-to-end run).

## Critical Gotchas & Rules

1. **Grouped Requests Share One Root**: Every question in a grouped request must tokenize to the same root prefix. Grouping uses exact state serialization. `--no-group` emits one request per row.
2. **Request Latency Is Not Row Latency**: Grouped rows report their shared request latency. Do not sum them. Use `decisions_per_second` for per-decision throughput.
3. **Strategy Sweep Is Native-Only**: Mock and family engines reject `--strategies`. On Qwen 3.5, `choose_strategy` uses the scheduler. Named strategies force a plan through `SchedulerConfig::with_forced_strategy`.
4. **Published Attribution**: `--commit` is caller-supplied metadata. It does not verify the measured source. Records need attributable hardware, the exact fixture, and source state. Follow the [evidence-retention rules](../../docs/BENCHMARKS.md#evidence-retention).
5. **Dataset Bytes Never Enter the Repo**: Acquisition uses
   [`openkind-datasets`](../openkind-datasets/AGENTS.md) and the external cache.
   Plain `score` ignores gold labels. Choice requests inject `__none__`, so
   external comparisons use dataset metric `answerable_ranking_accuracy`.
6. **Family Provenance**: Every profile needs a `family_identity` entry. Missing provenance reaches `native_backend` and panics after scoring. Keep matching exhaustive. Shared families retain their family `engine` slug while `engine_variant` identifies the profile and backend.
7. **Conditional Engine Mappings**: Update CPU and MLX mappings together. Family variants belong in the rejected arms of `native_backend`. Follow the [integration checklist](../../docs/families/NEW_FAMILY.md#benchmark-integration) and compile MLX on macOS arm64.

## Verification Commands

```bash
cargo check -p openkind-bench
cargo test -p openkind-bench
```

For MLX mappings, also compile all targets and run the offline feature tests
on macOS arm64. Default workspace checks omit these branches:

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo clippy -p openkind-bench \
  --features mlx --all-targets -- -D warnings
SDKROOT=$(xcrun --show-sdk-path) env -u RUST_LOG cargo test \
  -p openkind-bench --features mlx
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
