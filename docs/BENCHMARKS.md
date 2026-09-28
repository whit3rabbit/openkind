# Benchmarks

> How `openkind` is benchmarked for scoring and timing, how to reproduce
> runs, and where recorded evidence lives.

## Ownership

This document owns benchmark methodology (timing scope, warm policy,
repetitions, percentiles, execution-strategy sweep, parity assertions), the
`openkind-bench` harness guide, workload fixture inventory, and checked-in
harness-run records under [`benchmarks/`](./benchmarks/). Related material is
owned elsewhere and linked, not duplicated:

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — measurement-plan metric definitions
  (complete-request p50/p95, `T(Q)/T(1)`, state-prefill fraction, memory) and
  the landed execution-strategy contract.
- [`whitepaper/WHITEPAPER.md`](./whitepaper/WHITEPAPER.md)
  — canonical measured-results register for the native engine.
- [`ROADMAP.md`](./ROADMAP.md) — open benchmark work: practical high-K latency,
  full restored head/decision replay, and queue-inclusive load/soak.
- [`verification/`](./verification/) — verification records for gate runs.
- The `qwen35_scheduler_bench` example
  ([`crates/openkind-backends/examples/`](../crates/openkind-backends/examples/))
  — the Phase 3.8 scheduler deep-dive harness (forward-call accounting,
  synthetic mechanics grid, crossover measurement) that produced the measured
  `2.52` savings ratio. It remains the tool for backbone-level cost-model
  measurements; `openkind-bench` measures the full request path.

## Prior-art comparability (SemIf)

The methodology mirrors the published SemIf systems benchmark
(github.com/TheoLeeCJ/SemIf, MIT) so runs are **methodology-comparable**:

| SemIf scoring path | openkind equivalent |
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
| [`crates/openkind-bench/fixtures/decisions_smoke.jsonl`](../crates/openkind-bench/fixtures/decisions_smoke.jsonl) | 4 support tickets × 3 primitives (noul, choice, score) = 12 rows; mock-testable | `3a673e843690b942658b4c9de6cc594770185756098d356e78dcd3efd5cffeeb` |

Larger shape-matched workloads are **generated, not vendored**:
`openkind-bench gen-workload` produces a seeded ticket × binary-criterion
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
`--no-warmup` skips that pass for cold and execution-history probes; its
summary records `warmup: false`. Do not compare those first-request times to
the warm-process benchmark table.
`p50` is the median repetition total; `p95` is the `ceil(0.95·n)−1` sample;
sample counts ship alongside as `samples_seconds`.

The web playground's benchmark tab is **not** a harness. It measures
browser-side wall-clock round trips (including HTTP) against a daemon, which
is useful for interactive comparison but not comparable to the numbers above
and never recorded as evidence. Pinned numbers come from `openkind-bench
score` alone.

### Commands

```bash
# CI-safe smoke (mock engine, fully offline)
cargo run -p openkind-bench -- score \
  crates/openkind-bench/fixtures/decisions_smoke.jsonl \
  --engine mock --output-dir bench-output --reps 3

# Generate a shape-matched workload
cargo run -p openkind-bench -- gen-workload \
  --states 37 --criteria 21 --seed 291607 --output bench-output/shape777.jsonl

# Native engine (checkpoint-gated; loads only local pinned artifacts)
cargo run --release -p openkind-bench -- score <workload.jsonl> \
  --engine qwen35 \
  --bundle-root <profile-bundle-dir> \
  --checkpoint-root <pinned-checkpoint-dir> \
  --tokenizer <digest-locked tokenizer.json> \
  --strategies repeated_full,nested_sequential,nested_batched,choose_strategy \
  --reps 1 --host "<host label>" --commit <hash> --output-dir bench-output

# MLX engines (macOS arm64; requires the vendored mlx-c toolchain)
SDKROOT=$(xcrun --show-sdk-path) cargo run --release -p openkind-bench \
  --features mlx -- score <workload.jsonl> \
  --engine qwen35-mlx-fp32 \
  --bundle-root <profile-bundle-dir> \
  --checkpoint-root <pinned-or-community-checkpoint-dir> \
  --tokenizer <digest-locked tokenizer.json> \
  --reps 1 --host "<host label>" --commit <hash> --output-dir bench-output
```

Every published number must carry `--host` and `--commit` attribution;
summaries default to an "unattributed" host label that must be replaced before
results are quoted anywhere.

### Non-final Choice qualification

`scripts/qwen-qualification.py choice` joins a `score` prediction file to a
locked Choice workload and gold JSONL. Each gold row declares `id`,
`source_id`, `split`, `family`, `gold`, and the complete `options` list including
`__none__`. The tool checks exact row/option coverage, normalized probability
vectors, the workload digest in the benchmark summary, and source isolation
between declared splits. Group each source document and its generated or
paraphrased relatives under one `source_id`; the tool cannot detect undeclared
near duplicates. New benchmark summaries also bind each prediction file by
SHA-256. It refuses rows marked `final`. It reports accuracy, per-class
recall, false-none, NLL, Brier, `__none__` Brier, and fixed 10-bin top-label
calibration with bin counts. On small slices, the bins are diagnostic. A
previously fixed `--policy-threshold` adds accepted error and coverage. Zero acceptance
has null accepted error. NLL is null with a zero-probability gold label, and
the report counts those rows rather than clipping the infinite loss. An optional
locked reference summary and predictions
add paired correct-answer retention and supported-answer loss to none. Reports
are non-final and do not select a threshold.

```bash
python3 scripts/qwen-qualification.py choice \
  --workload <locked-choice-workload.jsonl> --gold <locked-gold.jsonl> \
  --summary <summary-engine.json> --predictions <predictions-engine-strategy.jsonl> \
  --output <report.json>
```

Add `--reference-summary` and `--reference-predictions` together for a paired
control, and `--policy-threshold` only when that threshold was fixed before
evaluating the supplied split.

`scripts/qwen-qualification.py history` runs two separate `openkind-bench`
processes against a two-row Choice workload: exact request A, unrelated request
B, A again, then A after a fresh process. `score --history-aba` reuses the
first workload row and its question ID; prediction rows retain sequence indices.
It compares the full probability
vectors, selected answers, and the pinned Choice policy from the bundle.
Both processes use `--no-warmup`, one repetition, one strategy, and local
artifacts only. Use the checked-in synthetic
[`qwen_history_smoke.jsonl`](../crates/openkind-bench/fixtures/qwen_history_smoke.jsonl)
to exercise mechanics. A pass is a bounded stability observation on that
workload and backend, not a quality or release result.

```bash
cargo build --release -p openkind-bench
python3 scripts/qwen-qualification.py history \
  --workload crates/openkind-bench/fixtures/qwen_history_smoke.jsonl \
  --bench-bin target/release/openkind-bench --engine qwen35 \
  --bundle-root <profile-bundle-dir> --checkpoint-root <pinned-checkpoint-dir> \
  --tokenizer <digest-locked-tokenizer.json> --host "<host label>" \
  --commit <hash> --output <history-report.json>
```

On macOS arm64, build the harness with `SDKROOT=$(xcrun --show-sdk-path)` and
`--features mlx`, then use `qwen35-mlx-fp32` to test that backend separately.
The script accepts `qwen35-mlx-bf16` only as a distinct candidate profile;
its frozen equivalence gate still fails. New serving profiles need their own
adapter and identity before the history result can be treated as theirs.

The `qwen35-mlx-fp32` and `qwen35-mlx-bf16` engines (behind the harness's
`mlx` feature) run the identical request path as `qwen35` — render,
state-first tokenization, scheduling, backbone execution, readout — with the
backbone swapped through `Qwen35Backend`; summaries carry the distinct engine
ids `qwen35-mlx-fp32` / `qwen35-mlx-bf16` and per-engine output slugs. MLX
bench numbers are throughput evidence only: the frozen parity gates live in
the parity examples. Pinned-base BF16 currently fails the frozen probability
tolerance in both full and nested execution, so BF16 remains unqualified for
decision use even though the nested state checks complete. A community
checkpoint can be passed as `--checkpoint-root` for throughput comparison,
but its bench output is not a parity claim.

### Candidate pooling diagnostic

[`qwen35_candidate_pool_bench`](../crates/openkind-backends/examples/qwen35_candidate_pool_bench.rs)
replays the pinned Phase 3B token suffixes through FP32 MLX. It reports
prefill, question, candidate, and readout time, observed physical forward
calls, padded token slots, and process/MLX peak memory. `Q=2` draws the two
equal-position questions with unequal candidate lengths; `Q=3` includes the
shorter question. Larger `Q` or `K` repeat fixture tokens as a load shape and
have no semantic-quality interpretation. `--q 3 --k 0` uses the three
original fixture questions and their natural candidate counts for decision
parity comparison.

Run each strategy in a separate process on a quiet host, alternating the
strategy order across paired repetitions:

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --release -p openkind-backends \
  --features mlx --example qwen35_candidate_pool_bench -- \
  --checkpoint <pinned-checkpoint-dir> --q 2 --k 2 --iterations 5 \
  --max-lanes 8 --strategy nested_sequential
# Repeat with --strategy nested_batched and --strategy pooled.
```

The timed region includes backbone continuation and score-summary readout,
but excludes request rendering, tokenization, wire validation, and model load.
These stage measurements localize a bottleneck. Promotion requires paired
fresh-process **full-request** `openkind-bench` runs, at least 10% median
latency improvement, no material p95 or memory regression, and the frozen
probability/selection/policy parity gates. The pooled runner is diagnostic
until those results exist; the automatic scheduler and service path retain
their current behavior.

The [27 September candidate-pooling record](./benchmarks/2026-09-27-candidate-pooling/)
reports paired FP32 MLX stage replays and a separate full-request baseline.
Pooling missed the median target and was slower than the current batched
runner on both measured shapes, so it was not promoted.

The [external Qwen2.5 MLX field-pooling record](./benchmarks/2026-09-27-qwen25-mlx-field-pool/)
compares pooled and per-field suffix forwards on a separate 4-bit checkpoint
through Python MLX. Its positive timing result is a reference experiment, not
an OpenKind request-path or Rust-backend result.

The [Rust flat-field record](./benchmarks/2026-09-27-python-flat-field/)
ports that shared-root field schedule to the pinned Qwen3.5 FP32 MLX backend.
Paired fresh-process native compute runs find it slower than the current
`nested_batched` traversal at Q2/K2 and Q8/K4. The flat path is diagnostic and
does not change the service or automatic scheduler.

### Outputs

- `summary-<engine>.json` — schema `openkind-bench/v1`: provenance
  (profile id, model revision, bundle version, fixture digest, host, commit),
  per-strategy prediction SHA-256 and warmup flag,
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

## Criterion microbenchmarks

Criterion 0.8.2 provides warm-process statistical timing for three product
surfaces that are intentionally smaller than the model-backed harness:

| Target | Measured region | Excluded |
|---|---|---|
| `openkind-cli` / `cli` | Clap parsing plus in-memory request JSON deserialization and validation for 1, 8, and 32 questions | Process startup, file I/O, stdout, and HTTP |
| `openkind-server` / `server` | Authenticated Axum middleware, JSON extraction, validation, MockEngine dispatch, serialization, and full response-body collection | TCP, daemon startup, rate limiting, and model execution |
| `openkind-client` / `client` | Warm localhost HTTP connection, SDK serialization/headers, authenticated Axum/MockEngine response, and SDK decoding | Connection setup, retry sleeps, remote network behavior, and model execution |

Run the complete timing targets with:

```bash
cargo bench -p openkind-cli --bench cli -- --noplot
cargo bench -p openkind-server --bench server -- --noplot
cargo bench -p openkind-client --bench client -- --noplot
```

Use `cargo test --workspace --benches --locked` for a one-iteration smoke
check. This verifies that every benchmark executes, but it does not produce
performance evidence or enforce regression thresholds. Criterion reports are
written below `target/criterion/` and are not committed.

Criterion measures speed and throughput, not scoped heap or resident memory.
For a process-level peak-RSS comparison, run one exact benchmark id in its own
process:

```bash
scripts/bench-rss.sh \
  --package openkind-client \
  --bench client \
  --filter client_http_systemone/questions/32
```

The helper builds the bench target with the lockfile, runs its executable
directly under `/usr/bin/time -l` on macOS or `/usr/bin/time -v` on Linux, and
records the subject commit, dirty-tree state, toolchain, host, command, and
output under `target/bench-rss/`. The result is the benchmark process high-water
mark, including Criterion, the runtime, and allocator retention. It is not
bytes per operation and must only be compared on the same host, toolchain,
benchmark id, and profile duration.

These microbenchmarks use MockEngine and localhost only. They are useful for
finding serialization, validation, middleware, and SDK regressions; they are
not native-model throughput, queue-inclusive service load, soak, Metal, or
production memory evidence. `openkind-bench` remains the authority for the
full engine request path and model-backed peak RSS.

## Native service load and soak

[`scripts/native-service-gate.py`](../scripts/native-service-gate.py) launches
the real `openkindd` process and drives the native HTTP endpoint over TCP. It
measures client-observed queue-inclusive p50/p95/p99 latency and accepted
request/question throughput at client concurrency 1, 2, and 4. It records
startup-to-health, first and resident requests, wire validation, request-ID
echoes, overloads, client-disconnect cancellation and recovery, a steady
single-client soak, current RSS samples, and the daemon's OS peak-RSS metric.
It starts a second daemon with a short queue-inclusive deadline and checks the
HTTP 504 path. The workload has no reviewed labels, so it cannot support
semantic accuracy, accepted-error, or coverage claims. The report saves no
request bodies, IDs, or API key; the daemon log is kept separately for
lifecycle diagnosis.

The runner defaults to a 30-minute soak. Download the pinned checkpoint using
the opt-in command in the repository README and set
`OPENKIND_QWEN35_CHECKPOINT` to that local directory. Build the release daemon
and run:

```bash
mkdir -p docs/verification/native-service-gate/run-YYYYMMDD
python3 scripts/native-service-gate.py --capture-source . \
  > docs/verification/native-service-gate/run-YYYYMMDD/build-source.json
CARGO_TARGET_DIR=target/native-service-gate cargo build --release --locked --offline -p openkind-server --bin openkindd
python3 scripts/native-service-gate.py --capture-source . \
  > docs/verification/native-service-gate/run-YYYYMMDD/post-build-source.json
python3 -c 'import json; a=json.load(open("docs/verification/native-service-gate/run-YYYYMMDD/build-source.json")); b=json.load(open("docs/verification/native-service-gate/run-YYYYMMDD/post-build-source.json")); keys=("commit", "tracked_diff_sha256", "untracked_crate_source_sha256", "harness_sha256"); assert all(a[k] == b[k] for k in keys), "source changed during build"'
python3 scripts/native-service-gate.py \
  --repo . \
  --daemon target/native-service-gate/release/openkindd \
  --bundle-root crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8 \
  --runtime-manifest research/14_phase3b_backbone_parity_results/bundle_probability_runtime/runtime.json \
  --checkpoint-root "$OPENKIND_QWEN35_CHECKPOINT" \
  --tokenizer research/14_phase3b_backbone_parity_results/backbone_runtime/tokenizer/tokenizer.json \
  --fixture crates/openkind-bench/fixtures/decisions_smoke.jsonl \
  --output-dir docs/verification/native-service-gate/run-YYYYMMDD \
  --build-source-record docs/verification/native-service-gate/run-YYYYMMDD/build-source.json \
  --soak-seconds 1800 \
  --build-profile release \
  --build-command 'CARGO_TARGET_DIR=target/native-service-gate cargo build --release --locked --offline -p openkind-server --bin openkindd'
```

The report records hardware, OS, toolchain, commit, tracked-diff, untracked
crate-source, harness, and executable hashes, plus profile/model/fixture
identity, load parameters, per-request status and latency, and shutdown status.
It deliberately distinguishes sampled current RSS from the operating-system
peak. The source snapshot is captured before building and checked again after
the build and at daemon start. The runner also records the final workspace
fingerprint and verifies that the tested executable hash stayed unchanged.
Source edits after daemon startup are reported, but do not invalidate the
already built artifact. Results from this endpoint campaign are service
evidence, separate from the in-process `openkind-bench score` timing scope
above.

### Recorded native CPU service campaign

The release-mode native service gate passed on a Mac16,5 Apple M4 Max (36 GiB,
macOS 26.6.2, Rust 1.98.1). The full report, service logs, source identity,
artifact hash, log-redaction audit, and interpretation are in
[`verification/native-service-gate/2026-09-22-rerun2/`](verification/native-service-gate/2026-09-22-rerun2/README.md).

| Load phase | Accepted latency p50 / p95 / p99 | Accepted throughput | Outcome |
|---|---:|---:|---|
| Concurrency 1 | 15.22 / 15.40 / 15.40 s | 0.06577 requests/s | 3/3 HTTP 200 |
| Concurrency 2 | 15.61 / 30.91 / 30.91 s | 0.06500 requests/s | 6/6 HTTP 200 |
| Concurrency 4 | 31.07 / 47.02 / 47.02 s | 0.06420 requests/s | 9 HTTP 200, 3 HTTP 529 |
| 30-minute soak | 15.78 / 15.88 / 15.94 s | 0.06339 requests/s | 115/115 HTTP 200, 345 answers |

Sampled RSS peaked at 10,280,976,384 bytes; the separate daemon peak-RSS
metric reported 10,283,155,456 bytes. The cancellation counter advanced and
the recovery request returned HTTP 200. The 1 ms deadline probe returned HTTP
504 in 2.12 ms. Build and daemon-start source fingerprints matched, and the
release executable hash stayed unchanged through the run. Shared source edits
after startup are recorded in the report and do not change the running
artifact. This evidence closes the native CPU service gate only. The profile
manifest remains `production_ready: false`, so model-quality review and
official release promotion remain separate.

## Recorded runs

| Record | Engine | Status |
|---|---|---|
| [`benchmarks/2026-09-27-python-flat-field/`](./benchmarks/2026-09-27-python-flat-field/) | qwen35-mlx-fp32 | Negative diagnostic: Rust shared-root flat field batching was 6% to 64% slower than nested batching across paired Q2/K2 and Q8/K4 compute runs; no scheduler promotion |
| [`benchmarks/2026-09-27-candidate-pooling/`](./benchmarks/2026-09-27-candidate-pooling/) | qwen35-mlx-fp32 | Negative diagnostic: pooled candidate lanes were slower than current batching at Q2/K2 and Q8/K4; no service or automatic-scheduler promotion |
| [`benchmarks/2026-09-26-surveyed-families/`](./benchmarks/2026-09-26-surveyed-families/) | decoder-logit-letter, encoder-nli, encoder-instruct-label, decoder-logit-llm, kev, schema-scorer, qwen3guard, winnow, router-script | Complete — single-shot surveyed-family records on the standard shape777 workload; request-path timing only, no model-quality claim; see the section below for numbers and provenance |
| [`benchmarks/2026-09-23-criterion-optimization/`](./benchmarks/2026-09-23-criterion-optimization/) | CLI, server, and client with MockEngine | Concluded same-host working-tree comparison; 11 of 15 existing cases meet 1.20x, and four sequential client cases remain below target |
| [`benchmarks/2026-09-22-criterion-microbenchmarks/`](./benchmarks/2026-09-22-criterion-microbenchmarks/) | CLI, server, and client with MockEngine | Complete clean-commit Criterion timing baseline, 100 samples per case; component overhead only |
| [`benchmarks/2026-09-20-mock-smoke/`](./benchmarks/2026-09-20-mock-smoke/) | mock | Complete — harness validation only; not performance evidence |
| [`benchmarks/2026-09-20-qwen35-smoke/`](./benchmarks/2026-09-20-qwen35-smoke/) | qwen35-native-cpu | Complete — smoke-scale, single-sample cells; CPU parity does not imply Metal or accelerated parity |
| [`benchmarks/2026-09-21-qwen35-mlx-smoke/`](./benchmarks/2026-09-21-qwen35-mlx-smoke/) | qwen35-mlx-fp32 (+ bf16 repeated_full probe) | Complete — first MLX bench dispatch; dirty-tree, single-sample cells; cross-strategy exact-answer flag false with bounded `1.68e-05` probability divergence, zero selection changes |
| [`benchmarks/2026-09-21-qwen35-mlx-community-smoke/`](./benchmarks/2026-09-21-qwen35-mlx-community-smoke/) | qwen35-mlx-fp32 (community checkpoint) | Complete — throughput only; community export fails the frozen parity gates and its numbers carry no model-quality claim |
| [`benchmarks/2026-09-21-qwen35-mlx-gdn-review/`](./benchmarks/2026-09-21-qwen35-mlx-gdn-review/) | qwen35-mlx-fp32 reference ops vs packed Metal tree | Complete working-tree record: raw summaries and predictions for the current production-default decision; candidate parity passed but throughput regressed |
| [`verification/phase3m-2026-09-21-dispatch-recheck.md`](./verification/phase3m-2026-09-21-dispatch-recheck.md) | qwen35-native-cpu vs qwen35-mlx-fp32 | Complete — same fixture and four strategies; fresh CPU, pinned-base MLX, and community MLX recheck |
| [`verification/phase3m-2026-09-21-working-tree.md`](./verification/phase3m-2026-09-21-working-tree.md) | qwen35-mlx-fp32 | Complete parity probe — dirty-tree, load-inclusive timing, not an `openkind-bench` throughput record |
| [`verification/phase3m5-2026-09-21-working-tree.md`](./verification/phase3m5-2026-09-21-working-tree.md) | qwen35-mlx-fp32 kernel review | Complete working-tree comparison: serialized explicit stream, fused-kernel parity, and same-host default-versus-candidate smoke sweep; candidate not promoted |


### Surveyed-family single-shot records (2026-09-26)

One `openkind-bench score` run per family over the standard seeded workload
(`gen-workload --states 37 --criteria 21 --seed 291607`, sha256
`be397bfc48209c8f7379d0d76ccbe3d3e7dca3269724928f0868cf872f4c8b01`, 777 rows,
37 state groups, `--reps 1` except router-script `--reps 3`). Host:
Apple Silicon Mac, 14 cores, 38 GB RAM, macOS arm64; commit `67d8283` at run
time; each engine loaded from its pinned, digest-verified checkpoint. These
are **request-path timing records only** — none of the profiles has M2
model-quality evidence, and the summaries carry no classification-accuracy
claim.

| Engine (profile) | Backbone | p50 request | Decisions/s | Peak RSS | Model load |
|---|---|---|---|---|---|
| `decoder-logit-letter` (`5492c97dfcdaf3fe9439`) | Qwen2.5-0.5B-Instruct fp32 | 107.55 s | 7.22 | 3.33 GB | 1.96 s |
| `encoder-nli` (`1041a4c362338a61b820`) | typeform/distilbert-base-uncased-mnli fp32 | 22.11 s | 35.14 | 0.56 GB | 0.53 s |
| `encoder-instruct-label` (`9fd68313a5606eca42f2`) | knowledgator/gliclass-modern-base-v3.0 fp32 (ModernBERT-base, hand-implemented) | 194.20 s | 4.00 | 1.27 GB | 1.23 s |
| `kev` (`39d88c11faeb4ac165fa`) | jaredpalmer/kev-0.6b (LoRA on Qwen3-0.6B-Base, pointer head) fp32 | 160.39 s | 4.84 | 4.22 GB | 5.61 s |
| `decoder-logit-llm` (`465963d705b6f35d6208`) | Qwen2.5-0.5B-Instruct-GGUF q8_0 (candle quantized runner) | 1836.70 s | 0.42 | 1.55 GB | 1.52 s |
| `schema-scorer` (`5a7350af556f0ee66566`) | cross-encoder/ms-marco-MiniLM-L-6-v2 fp32 | 75.83 s | 10.25 | 0.26 GB | 0.22 s |
| `qwen3guard` (`0fcf416cab16d94f933d`) | Qwen3Guard-Stream-0.6B fp32 | 156.26 s | 4.97 | 4.20 GB | 2.44 s |
| `winnow` (`4dff8c5b03cfbf680db6`) | Qwen2.5-0.5B-Instruct fp32 + LoRA (routing pass only; mock siblings) | 3.99 s | 194.58 | 3.32 GB | 2.40 s |
| `router-script` (rule table) | none — Unicode detector (mock siblings) | 1.10 ms | 704,336 | 0.01 GB | — |

Notes:

- The p50 request groups 21 binary questions per state document; per-question
  cost is roughly p50 / 21 for the per-candidate families, and the letter
  families run one forward pass per question.
- `decoder-logit-llm` is the candle CPU quantized (q8_0) runner: the
  dequantized matmul path is an order of magnitude slower per token than the
  fp32 safetensors letter profile on the same host. The GGUF binding keeps
  the checkpoint format; it is not an acceleration claim, and llama.cpp /
  Metal execution remains unexplored.
- `winnow` and `router-script` runs delegate answering to mock siblings, so
  their numbers measure the routing pass (model forward or rule table) only.
- Calibration temperatures for the scored profiles are fitted on the pinned
  synthetic calibration workloads documented in each family module
  (`families/calibration.rs` cases, stated-fact construction); the fits are
  recorded in `docs/families/*.md` and in the frozen profile constants.
- `encoder-instruct-label` was appended to this record set after its
  unblocking push (still commit `67d8283`, dirty tree). Its temperature fit
  minimized mean NLL over 15 stated-fact calibration cases with the exact
  wire readout per primitive: optimum `T = 0.44530092168688412`, mean NLL
  `0.151` (vs `0.194` at `T = 1`); an interior optimum, so unlike
  `qwen3guard` the fit is not the degenerate sharpening case. The Rust
  forward was validated against a PyTorch reference of the same checkpoint
  to `<= 4.3e-6` maximum answer delta over the 15 golden cases.
- `kev` was appended in the same push after the family's blocker lapsed
  (the reference author published open-weight checkpoints, code, and
  protocol under Apache-2.0). Its temperature fit is degenerate in the
  `qwen3guard` sense — the merged model already assigns >= 0.95 to the
  correct side of 14 of 15 stated-fact cases, so NLL sharpens to a hard
  one-hot (`T -> 0.048`) without improving decisions — and `T = 1.0` is
  pinned with that rationale. The Rust forward was validated against a
  PyTorch reference with the same merged weights: probabilities agree to
  `<= 3e-4` on 14 of 15 golden cases and `<= 1.9e-2` on the single
  near-degenerate case (top-two logits within 1.4), with the argmax
  preserved everywhere; the residual is fp32 GEMM accumulation-order
  difference between Apple Accelerate (candle) and MKL (PyTorch), shown by
  bit-identical merged weights and per-layer divergence that starts at the
  first GEMM.

### Initial MLX dispatch recheck (historical)

This fresh working-tree comparison used the same 12-row, four-state smoke
fixture, one timed repetition per strategy, warm-process timing, and the same
named M4 Max host. CPU and pinned MLX use the same frozen base checkpoint, so
that pair is the meaningful backend comparison. The community row uses the
digest-locked `mlx-community/Qwen3.5-4B-MLX-bf16` adapter and is a separate
model comparison. Model load is excluded from the totals and reported in the
run summaries; load ranged from `17.69–17.92 s` for CPU, `22.08–22.93 s` for
pinned MLX, and `21.71–21.98 s` for community MLX.

| Backend / checkpoint | `repeated_full` | `nested_sequential` | `nested_batched` | `choose_strategy` | Peak RSS | Cross-strategy exact parity |
|---|---:|---:|---:|---:|---:|---|
| Candle CPU / pinned base | 119.22 s | 70.34 s | 71.41 s | 68.83 s | 10.08 GB | true |
| MLX FP32 / pinned base | 29.42 s | 7.91 s | 7.92 s | 7.94 s | 11.66 GB | false |
| MLX FP32 / community export | 29.57 s | 7.91 s | 7.89 s | 7.88 s | 11.92 GB | false |

Relative to the CPU run, pinned MLX was `4.05x` faster for repeated-full,
`8.89x` faster for nested-sequential, `9.02x` faster for nested-batched, and
`8.67x` faster for the measured scheduler choice. The community export was
within roughly `0.4%` of pinned MLX on these single-sample cells, so this does
not show a meaningful speed difference between the two MLX weight sets.
Pinned MLX versus CPU `choose_strategy` answers had maximum probability delta
`1.7687e-05`, maximum scalar delta `1.8477e-05`, and zero Choice selection
changes. The exact-answer flag is false because the benchmark compares full
wire JSON values, not only selected options; this bounded numerical difference
is separate from the frozen model-parity gate.

The community `choose_strategy` answers differed from CPU on three Choice
selections, with maximum probability delta `0.91194`. That is expected from
the community artifact's different source model/conversion and is not evidence
of an MLX kernel or dispatch regression. Its summary still reports the fitted
head's pinned-base revision in `model_revision`; use the checkpoint identity in
the community record and adapter state identity when interpreting that run.

### Current Gated-DeltaNet kernel review

A later same-day working-tree review compared the production
`mlx-core-0.32.2/fp32/reference-ops` path with an opt-in packed FP32 Metal
sequence kernel on the same host, pinned checkpoint, fixture, and one-sample
methodology. The production rerun also includes the new serialized explicit
GPU stream and loader/memory-accounting changes.

| Arithmetic path | `repeated_full` | `nested_sequential` | `nested_batched` | `choose_strategy` | Peak RSS |
|---|---:|---:|---:|---:|---:|
| `reference-ops` (production default) | 29.998 s | 7.819 s | 7.860 s | 7.755 s | 11,838,046,208 bytes |
| `metal-tree-packed-dk128-v1` (opt-in candidate) | 38.279 s | 9.055 s | 8.957 s | 9.060 s | 11,894,095,872 bytes |

The packed candidate was 1.14–1.28 times slower, so it is not promoted. This
is a useful negative optimization result, not a parity failure: the candidate
passed the pinned-base full gate (maximum probability delta `3.9155e-07`, no
argmax or policy changes) and nested gate (maximum probability delta
`6.6133e-06`, cached-versus-full `7.6294e-05`). The generic masked and
vector-gate kernels remain direct-test coverage and future batching building
blocks. The comparison is a dirty-working-tree, single-sample result. Raw
summaries and prediction files are preserved in the linked GDN review record;
exact commands and boundaries are in the linked 3M.5 verification note.

### MLX parity timing boundary

The 21 September 2026 MLX probe used the pinned original Qwen3.5-4B-Base
checkpoint on the named macOS arm64 development Mac, with MLX 0.32.2,
Xcode 27.0, and Metal toolchain 32023.921. These timings include process
startup, model loading, and the parity fixture work, so they are useful for
bring-up and memory sizing only. They exclude neither load nor fixture
comparison and must not be compared directly with the warm-process
`openkind-bench` numbers above.

| Run | Result | Load | Wall | Peak MLX allocation | Peak RSS |
|---|---|---:|---:|---:|---:|
| Full FP32 parity, 10 candidates | pass | 4.7 s | 44.7 s | 16.10 GB | 6.40 GB |
| Nested FP32 parity, 10 candidates | pass | included | 59.3 s | not emitted | 9.85 GB |
| Full BF16 parity, 10 candidates | gate fail | 3.7 s | 37.9 s | 8.05 GB | 8.35 GB |
| MLX-community full FP32, 10 candidates | model-parity fail | 4.4 s | 38.2 s | 15.61 GB | 8.70 GB |
| MLX-community nested FP32, 10 candidates | model-parity fail | included | 69.7 s | not emitted | 6.62 GB |
| MLX-community full BF16, 10 candidates | model-parity fail | 3.7 s | 55.3 s | 7.81 GB | 7.66 GB |

These BF16 timings are from the 21 September bring-up run. Its full run
retained all argmax and policy decisions but exceeded the probability
tolerance (`0.005457` versus `0.005`), and its nested run stopped at layer 4
state validation. The 22 September pinned-base rerun supersedes those Gate B
results: full maximum probability delta was `0.006099619710620674`, and nested
maximum probability delta was `0.026580797832947478`; both retained all
argmax and policy decisions, and nested state, position, root, and sibling
checks passed. The frozen tolerance remains `0.005`, so Gate B still fails.
The bounded first-mismatch trace localizes the first captured BF16 split to
shape-dependent layer 0 output projection rounding. Raw commands and outputs
are in [`verification/phase3m-2026-09-22/README.md`](verification/phase3m-2026-09-22/README.md).
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

## External evaluation and submission

When implementation, native parity qualification, and release readiness are
finished, submit the model for external evaluation:

- **BenchmarkHeaven Custom Evaluation**:
  <https://benchmarkheaven.com/jev-models/custom-evaluation> — submission portal
  for independent Jev-compatible model evaluation.
- **`jevbench` Harness**:
  [`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench) —
  evaluation suite and benchmark harness for Jev models.
