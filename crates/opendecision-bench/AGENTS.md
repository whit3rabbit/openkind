# AGENTS.md — opendecision-bench

> LLM developer guide for `opendecision-bench`. Read this before modifying the benchmark harness.

## Crate Purpose & Boundaries

`opendecision-bench` provides the `opendecision-bench` binary: an offline scoring and
timing harness over JSONL decision workloads. It exists to:

- Score decision workloads end to end through a real engine path (`score`), recording
  per-row predictions and a provenance summary (`opendecision-bench/v1`).
- Generate seeded, deterministic shape-matched workloads (`gen-workload`).

Methodology, timing scope, and recorded results are owned by
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) — do not duplicate numbers here.

### Critical Invariants

1. **Dependency Direction**: Depends on `opendecision-core`, `opendecision-engine`,
   `opendecision-backends`, and `opendecision-runtime` (all sit at or below the CLI/serve
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

## Key Files & Types

- [`src/main.rs`](./src/main.rs): Entrypoint; `gen-workload` prints one JSON result line,
  `score` prints the summary JSON to stdout (progress goes to stderr).
- [`src/args.rs`](./src/args.rs): Clap parser; `EngineArg`, `parse_strategies`.
- [`src/workload.rs`](./src/workload.rs): `WorkloadRow` (flattened `primitive` tag),
  `load_workload`/`parse_workload` (SHA-256 recorded), `state_groups`,
  `build_request`. Choice rows always carry a non-empty `__none__` criterion — one is
  injected when absent so native Choice evaluation keeps semantic-none mass on wire.
- [`src/gen.rs`](./src/gen.rs): Seeded ticket-grid generator (`states × criteria` binary
  noul rows), byte-identical for identical seeds.
- [`src/score.rs`](./src/score.rs): `run_score`/`ScoreArgs`. Native engine sweeps run the
  scheduler's `forced_strategy` diagnostic override per strategy (admission still
  enforced) and assert cross-strategy answer equality per workload.
- [`src/tests.rs`](./src/tests.rs): Offline tests (fixture parsing, grouping, `__none__`
  injection, generator determinism, mock end-to-end run).

## Critical Gotchas & Rules

1. **Grouped Requests Share One Root**: The state-first renderer requires every question
   in one request to tokenize to the same root prefix. Grouping is by exact state
   serialization; `--no-group` emits one request per row (the fresh-scoring baseline).
2. **Request Latency Is Not Row Latency**: In grouped mode, every row in a group reports
   its request's latency. Never sum those into per-decision latencies; use
   `decisions_per_second` from the strategy report instead.
3. **Strategy Sweep Is Native-Only**: The mock engine ignores `--strategies` and reports
   a single `mock` pass. `choose_strategy` lets the measured scheduler decide; concrete
   strategy names force the plan through `SchedulerConfig::with_forced_strategy`.
4. **Attribution Is Mandatory for Published Numbers**: `--host` and `--commit` exist so
   recorded results can name their hardware and commit; summaries carry a default
   "unattributed" host label that must be replaced before results are quoted in docs.

## Verification Commands

```bash
cargo check -p opendecision-bench
cargo test -p opendecision-bench
```

Native timing runs (checkpoint-gated, not part of `cargo test`):

```bash
cargo run --release -p opendecision-bench -- score <workload.jsonl> \
  --engine qwen35 --bundle-root <dir> --checkpoint-root <dir> --tokenizer <json> \
  --host "<host label>" --commit <hash>
```
