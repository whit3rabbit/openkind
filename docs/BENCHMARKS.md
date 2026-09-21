# Benchmarks

> How `opendecision` is benchmarked for scoring and timing, how to reproduce
> runs, and where recorded evidence lives.

## Ownership

This document owns benchmark methodology (timing scope, warm policy,
repetitions, percentiles, execution-strategy sweep, parity assertions), the
`opendecision-bench` harness guide, workload fixture inventory, and checked-in
harness-run records under [`benchmarks/`](./benchmarks/). Related material is
owned elsewhere and linked, not duplicated:

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — measurement-plan metric definitions
  (complete-request p50/p95, `T(Q)/T(1)`, state-prefill fraction, memory) and
  the landed execution-strategy contract.
- [`whitepaper/OpenDecision_Whitepaper_v0.8.0.md`](./whitepaper/OpenDecision_Whitepaper_v0.8.0.md)
  — canonical measured-results register for the native engine.
- [`ROADMAP.md`](./ROADMAP.md) — open benchmark work: practical high-K latency,
  full restored head/decision replay, and queue-inclusive load/soak.
- [`verification/`](./verification/) — verification records for gate runs.
- The `qwen35_scheduler_bench` example
  ([`crates/opendecision-backends/examples/`](../crates/opendecision-backends/examples/))
  — the Phase 3.8 scheduler deep-dive harness (forward-call accounting,
  synthetic mechanics grid, crossover measurement) that produced the measured
  `2.52` savings ratio. It remains the tool for backbone-level cost-model
  measurements; `opendecision-bench` measures the full request path.

## Prior-art comparability (SemIf)

The methodology mirrors the published SemIf systems benchmark
(github.com/TheoLeeCJ/SemIf, MIT) so runs are **methodology-comparable**:

| SemIf scoring path | opendecision equivalent |
|---|---|
| Fresh direct scoring (batch one per decision) | `--no-group` requests; scheduler-forced `repeated_full` |
| Serial prefix reuse (one state prefill) | State-grouped requests, `nested_sequential` |
| Parallel shared-state suffixes | State-grouped requests, `nested_batched` |
| — (no equivalent) | `choose_strategy` (measured scheduler) |

Their timing scope matches ours: prompt construction, tokenization, forward
passes, and readout are included; model loading and result writes are
excluded; warm process.

SemIf's published numbers are **reference points, not head-to-head results** —
their own METHOD.md says the same about cross-system comparisons. Their
records are RTX 3090 CUDA BF16 (direct readout 1.023 s median vs 5.332 s
autoregressive on 21 binary criteria; 777-decision throughput 2.33 fresh /
10.75 serial-reuse / 20.03 parallel decisions-per-second) and an MLX demo on
Apple M5 Max (6.36 s wall time including load). Ours run candle-cpu-fp32 on a
named M4 Max with different fixtures and different pinned model revisions.
Only ratio-shaped quantities (strategy ratios, `T(Q)/T(1)`, prefill fraction)
are meaningfully comparable across these environments; absolute
decisions-per-second are not.

## Workload model

JSONL, one decision per row, flattened primitive tag:

```json
{"id": "t01.route", "state": {"ticket_id": "..."}, "primitive": "choice",
 "text": "Which queue should own this ticket?",
 "options": [{"id": "billing", "description": "Charges, invoices, taxes, refunds"}]}
```

- `state` is any Jev state (string, object, or array); `primitive` is
  `noul`, `choice`, or `score`.
- `noul` rows take optional `criteria: {"true": ..., "false": ...}`;
  `score` rows take ordered `levels` (at least two); `choice` rows take
  `options`.
- Choice rows always carry the reserved `__none__` criterion on the wire; the
  harness injects a default description when a row omits it, preserving
  semantic-none mass per the workspace invariant.

### Fixtures

| Fixture | Contents | SHA-256 |
|---|---|---|
| [`crates/opendecision-bench/fixtures/decisions_smoke.jsonl`](../crates/opendecision-bench/fixtures/decisions_smoke.jsonl) | 4 support tickets × 3 primitives (noul, choice, score) = 12 rows; mock-testable | `3a673e843690b942658b4c9de6cc594770185756098d356e78dcd3efd5cffeeb` |

Larger shape-matched workloads are **generated, not vendored**:
`opendecision-bench gen-workload` produces a seeded ticket × binary-criterion
grid (default 37 × 21 = 777 decisions, matching the prior-art shape). Identical
seeds emit byte-identical files; every run summary records the fixture
SHA-256, which pins the workload without committing megabytes.

## Harness

### Timing scope

Included: request construction, wire validation, dispatch, and answer
extraction (for the native engine this spans render/tokenize, backbone
execution, and readout). Excluded: model load, result-file writes, and the
untimed warmup pass over every group that precedes timing; model load is
reported separately per strategy (`model_load_seconds`). Warm process.
`p50` is the median repetition total; `p95` is the `ceil(0.95·n)−1` sample;
sample counts ship alongside as `samples_seconds`.

### Commands

```bash
# CI-safe smoke (mock engine, fully offline)
cargo run -p opendecision-bench -- score \
  crates/opendecision-bench/fixtures/decisions_smoke.jsonl \
  --engine mock --output-dir bench-output --reps 3

# Generate a shape-matched workload
cargo run -p opendecision-bench -- gen-workload \
  --states 37 --criteria 21 --seed 291607 --output bench-output/shape777.jsonl

# Native engine (checkpoint-gated; loads only local pinned artifacts)
cargo run --release -p opendecision-bench -- score <workload.jsonl> \
  --engine qwen35 \
  --bundle-root <profile-bundle-dir> \
  --checkpoint-root <pinned-checkpoint-dir> \
  --tokenizer <digest-locked tokenizer.json> \
  --strategies repeated_full,nested_sequential,nested_batched,choose_strategy \
  --reps 1 --host "<host label>" --commit <hash> --output-dir bench-output
```

Every published number must carry `--host` and `--commit` attribution;
summaries default to an "unattributed" host label that must be replaced before
results are quoted anywhere.

### Outputs

- `summary-<engine>.json` — schema `opendecision-bench/v1`: provenance
  (profile id, model revision, bundle version, fixture digest, host, commit),
  per-strategy `samples_seconds` / `p50_seconds` / `p95_seconds` /
  `decisions_per_second` / `input_tokens_total`, peak resident bytes, and
  `cross_strategy_answer_parity_clean`.
- `predictions-<engine>-<strategy>.jsonl` — one row per decision: id,
  strategy, group size, request latency, and the full typed answer
  (probabilities are wire-precision `f64`).

Grouped mode shares one request per distinct state, so `request_latency_ms`
per row is that row's request latency, not an independent per-decision timing;
use `decisions_per_second` for throughput claims. `--no-group` produces one
request per row (the fresh-scoring baseline).

Native sweeps force each strategy through the scheduler's diagnostic override
(`SchedulerConfig::with_forced_strategy`); admission ceilings still apply.
Answer equality across strategies is asserted per workload — a violation
flags `cross_strategy_answer_parity_clean: false` in the summary and must be
investigated before any numbers from that run are quoted.

## Recorded runs

| Record | Engine | Status |
|---|---|---|
| [`benchmarks/2026-09-20-mock-smoke/`](./benchmarks/2026-09-20-mock-smoke/) | mock | Complete — harness validation only; not performance evidence |
| [`benchmarks/2026-09-20-qwen35-smoke/`](./benchmarks/2026-09-20-qwen35-smoke/) | qwen35-native-cpu | Complete — smoke-scale, single-sample cells; CPU parity does not imply Metal or accelerated parity |
| [`verification/phase3m-2026-09-21-working-tree.md`](./verification/phase3m-2026-09-21-working-tree.md) | qwen35-mlx-fp32 | Complete parity probe — dirty-tree, load-inclusive timing, not an `opendecision-bench` throughput record |

### MLX parity timing boundary

The 21 September 2026 MLX probe used the pinned original Qwen3.5-4B-Base
checkpoint on the named macOS arm64 development Mac, with MLX 0.32.2,
Xcode 27.0, and Metal toolchain 32023.921. These timings include process
startup, model loading, and the parity fixture work, so they are useful for
bring-up and memory sizing only. They exclude neither load nor fixture
comparison and must not be compared directly with the warm-process
`opendecision-bench` numbers above.

| Run | Result | Load | Wall | Peak MLX allocation | Peak RSS |
|---|---|---:|---:|---:|---:|
| Full FP32 parity, 10 candidates | pass | 4.7 s | 44.7 s | 16.10 GB | 6.40 GB |
| Nested FP32 parity, 10 candidates | pass | included | 59.3 s | not emitted | 9.85 GB |
| Full BF16 parity, 10 candidates | gate fail | 3.7 s | 37.9 s | 8.05 GB | 8.35 GB |
| MLX-community full FP32, 10 candidates | model-parity fail | 4.4 s | 38.2 s | 15.61 GB | 8.70 GB |
| MLX-community nested FP32, 10 candidates | model-parity fail | included | 69.7 s | not emitted | 6.62 GB |
| MLX-community full BF16, 10 candidates | model-parity fail | 3.7 s | 55.3 s | 7.81 GB | 7.66 GB |

The BF16 full run retained all argmax and policy decisions but exceeded the
probability tolerance (`0.005457` versus `0.005`). The BF16 nested run failed
state validation at layer 4, so no BF16 nested throughput claim is valid.
The community adapter completed loading and full execution, but its output was
not a frozen-reference parity result: embedding max error was `8.5449e-04`,
feature max error `45.64`, probability max error `0.9999983`, with four argmax
changes and three policy changes. The complete commands, model revisions, and
shard digests are recorded in the linked verification note. These are
load-inclusive bring-up timings, not warm-process throughput measurements. The
community native-BF16 run also completed, but its probability delta was
`0.99055`, with one argmax change and two policy changes.

The community nested FP32 run preserved position checks, root storage and
immutability, and sibling isolation, but failed model parity with maximum
probability delta `0.9999982301` and maximum cached-versus-full feature delta
`0.2339146631`. Its `/usr/bin/time -l` wall time was `69.69 s`, with peak RSS
`6.62 GB` and peak memory footprint `16.09 GB`. The nested result confirms that
the adapter can exercise the complete branch topology, but it does not turn the
community export into a parity-qualified model.

Against the pinned FP32 runs, the community full run was roughly 15% faster
wall-clock, while its nested run was roughly 18% slower. Because the community
checkpoint has a different source model and conversion, these load-inclusive
timings are implementation observations, not model-speed claims.

The named-machine native follow-up verification records bounded model-backed
K=32/64/128/255 completion and RSS behavior, structural fresh-process
persistence replay, and native service lifecycle smoke. It is linked from
[`verification/2026-09-20-v0.8.0-native-follow-up-working-tree.md`](./verification/2026-09-20-v0.8.0-native-follow-up-working-tree.md).
Those numbers are correctness/memory and lifecycle evidence, not practical
high-K latency or production throughput benchmarks.

Queue-inclusive HTTP service latency and long-duration soak remain deferred
roadmap work; the harness measures the in-process request path only.
