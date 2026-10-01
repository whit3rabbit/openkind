# Benchmarks

> How `openkind` is benchmarked for scoring and timing, how to reproduce
> runs, and where recorded evidence lives.

## Ownership

This document owns benchmark methodology (timing scope, warm policy,
repetitions, percentiles, execution-strategy sweep, parity assertions), the
`openkind-bench` harness guide, workload fixture inventory, and checked-in
harness-run records under [`benchmarks/`](./benchmarks/). Related material is
owned elsewhere and linked, not duplicated:

- [`MODELS.md`](./MODELS.md) owns the per-profile backend and performance
  index. This page owns measurement methodology and individual run records;
  it is not a consolidated model ranking table.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — measurement-plan metric definitions
  (complete-request p50/p95, `T(Q)/T(1)`, state-prefill fraction, memory) and
  the landed execution-strategy contract.
- [`whitepaper/WHITEPAPER.md`](./whitepaper/WHITEPAPER.md)
  — canonical measured-results register for the native engine.
- [`verification/`](./verification/) — verification records for gate runs.
- The `qwen35_scheduler_bench` example
  ([`crates/openkind-backends/examples/`](../crates/openkind-backends/examples/))
  — the Phase 3.8 scheduler deep-dive harness (forward-call accounting,
  synthetic mechanics grid, crossover measurement) that produced the measured
  `2.52` savings ratio. It remains the tool for backbone-level cost-model
  measurements; `openkind-bench` measures the full request path.

## Prior-art comparability (SemIf)

The methodology mirrors the published SemIf systems benchmark
(github.com/TheoLeeCJ/SemIf-OpenJev, MIT — formerly `TheoLeeCJ/openjev`) so runs are **methodology-comparable**:

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

## Prior-art comparability (jev-benchmarking)

The University of Bonn harness
([`AppliedMachineLearning-Lab/jev-benchmarking`](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking),
MIT) published with [Deußer et al., arXiv:2609.37647](https://arxiv.org/abs/2609.37647)
is the closest published methodology to our labeled diagnostics: frozen
zero-shot request templates, a dev split for prompt work with the wording
frozen before the reported eval split, exact option probabilities from
open-weight reference models scored on identical requests, ECE/Brier,
bootstrap intervals, development-tuned Noul thresholds, and memorization
probes. Its target is the commercial Jev API; the findings are reviewed in
[`RESEARCH.md`](./RESEARCH.md#independent-37-dataset-benchmark-of-jev-reviewed-2026-09-30).
This section records what transfers to `openkind-bench` and what does not.

| Their practice | `openkind-bench` today | Assessment |
|---|---|---|
| One frozen template per dataset; question wording frozen before the eval split | Pinned workload fixtures; renderer identities frozen in registered profiles | Aligned |
| `--split dev` for prompt work; `--split eval` is the reported split | Choice diagnostics reject shared source groups and row IDs across partitions; qualification refuses `final` rows | Aligned |
| 95% percentile bootstrap intervals, 500 resamples | 2,000 source-group paired bootstraps (seed 29160717) | Aligned; resample counts differ, so interval widths are not cross-suite comparable |
| ECE with 15 equal-width bins plus Brier | Fixed 10-bin ECE, multiclass Brier, NLL (probability floor `1e-15`) | Aligned; bin counts differ, so ECE values are not cross-suite comparable |
| Selective prediction: accuracy at 80%/50% coverage plus AURC | Fixed risk/coverage points in the Choice diagnostics | Coverage points present; AURC is a candidate addition |
| Noul thresholds tuned on ~1,000 development items, then locked | `calibrate-choice` fit/lock/gate partitions | Same protocol at smaller scale |
| Content-addressed response cache keyed by the exact request; evaluation never re-runs inference | Every `score` run recomputes all forwards | Candidate adaptation |
| Deterministic hash-ordered `--limit N`; smaller samples are subsets of larger ones so pilot answers are reused | Seeded `gen-workload`; no pilot-subset-of-full guarantee | Candidate adaptation |
| Memorization probes: option rotation and question withholding | `compare-choice` position and code rotation factors; no question-withholding probe | Withholding probe is a candidate addition |
| API spend caps (`--max-cost`, `JEV_TOTAL_BUDGET_USD`) | Fully local, offline inference | Not applicable — no metered cost to cap |
| 37 public datasets pulled from the Hugging Face Hub on first use | Pinned, digest-verified downloads under explicit `openkind-bench dataset pull`/`pin` commands; tests and builds never download | Adopted with pinning — see [Dataset accuracy evaluation](#dataset-accuracy-evaluation); nothing is redistributed from this repository |

### Candidate adaptations (proposed)

None of these are landed; each would follow the existing summary schema,
attribution rules, and offline constraints.

1. **Request-digest response cache.** Key finished typed answers by a
   digest of engine identity, profile revision, checkpoint digest, and the
   exact request bytes, so metric or report changes can be recomputed
   without re-running the backbone. This is the local-compute analog of
   their "nothing is ever paid for twice" rule and would make
   qualification re-scoring cheap. Cache entries must record the same
   provenance the summaries do, or a stale cache becomes silently wrong
   evidence.
2. **Nested deterministic sampling.** Landed for dataset evaluation:
   materialized rows are hash-ordered, so `--limit N` selects a subset of
   any larger limit and pilot forwards stay valid for the full run.
3. **Question-withholding probe.** Reuse the `compare-choice` panel
   machinery with the question text removed. Accuracy should fall toward
   chance while option rotation stays flat; together the two probes bound
   state-only shortcuts and option-set memorization.
4. **AURC alongside coverage points.** Landed for dataset evaluation:
   `dataset eval` reports AURC plus accuracy at 50%/80% coverage for Choice
   and Noul rows; the authored-panel diagnostics keep their fixed points.

Their comparison protocol — identical requests, options rendered as single
tokens, exact probabilities from one forward pass, thinking disabled — is
also the standard an external evaluator will apply to OpenKind profiles.
Complete per-row predictions with provenance, already the harness rule, are
what make that comparison possible.

Licensing note: their Zenodo release ships Jev responses under a
research/evaluation-only license (no distillation, no competing products);
the open-model responses are Apache 2.0. Their response corpora cannot
become OpenKind training data. The MIT harness code may be consulted or
ported with attribution.

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
| [`joint_choice_diagnostic.jsonl`](../crates/openkind-bench/fixtures/joint_choice_diagnostic.jsonl) | 96 labeled Choice cases, 4 tasks, 24 source groups; balanced real-option and none labels; [panel contract](../crates/openkind-bench/fixtures/joint_choice_diagnostic.md) | `279dc6285bf65f61e8c3a81d136c9aeb4fed4ceea1943202a62d1c8d7593a45a` |
| [`joint_reference_card_diagnostic.jsonl`](../crates/openkind-bench/fixtures/joint_reference_card_diagnostic.jsonl) | 24 labeled card interventions, 6 source groups; only the reference option changes within each group; [panel contract](../crates/openkind-bench/fixtures/joint_reference_card_diagnostic.md) | `c1edcb1161c3e8eb6ec833ae3b3071d14b8f5ea8747c69d8bc97b5cb910224af` |

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

For the existing runs linked below, `request_latency_ms` times dispatch and
validation, excluding request construction and answer extraction. The current
timer includes both. Quoted token rates use one repetition and are unaffected
by the correction to token accounting across repetitions.

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

### Evidence retention

`--commit` records the supplied string. The scoring harness does not verify
that revision or capture dirty source changes. A commit label alone cannot
reproduce a run built from later edits.

Keep these with each published record:

- The exact command, hardware, OS, toolchain, checkpoint revision, and
  measured source state. For a dirty tree, archive the tracked diff and
  untracked source files used by the build. Record the executable digest.
- The workload name and SHA-256, grouping, repetitions, and warmup settings.
  Label workload exceptions in model tables before comparing throughput.
- Raw summaries and the prediction files bound by their SHA-256 fields.
  A timing summary does not contain accuracy metrics. Accuracy claims need
  predictions joined to gold and the evaluation output.
- Parity commands, configured checkpoint roots, and replay logs. Tests that
  return early without a local checkpoint provide no checkpoint parity
  evidence. Preserve an initial failure when reporting a successful rerun.

Identify missing evidence in the record. Keep historical attribution intact
when the measured source state cannot be recovered, and label unsupported
claims as reported rather than verified.

### Experimental joint-option comparison

`compare-choice` compares the current fitted state-first scorer against one
joint prompt that contains all real options and an explicit none option. It
reads only the tied output rows for single-token letters A..P and Z, with raw
temperature 1.0 and no generated text. The same verified base checkpoint and
FP32 arithmetic serve every method. This is an offline probe; it does not
register a model or change daemon defaults.

The comparison records nine methods. `independent_fitted` replays the frozen
renderer and head. `catalogue_fitted` keeps the fitted head and frozen
temperature but inserts a canonical all-option catalogue into every candidate
prompt's question branch under the separate `catalogue_state_first/v1`
renderer identity. `joint_forward`, `joint_reverse`, `joint_text_rotate`, and
`joint_code_rotate` are single joint renders: forward, the recorded reversal
(order and codes changed together, none last), a rotation of the full
displayed list with codes kept bound to their options, and a letter-code
permutation at fixed positions that moves the none option off `Z`. The
ensembles `joint_average` (forward plus reversal), `joint_pair_text` (forward
plus text rotation), and `joint_ensemble_four` (all four joint renders)
average the remapped distributions and cost the sum of their member passes.

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --release -p openkind-bench \
  --features mlx -- compare-choice \
  crates/openkind-bench/fixtures/joint_choice_diagnostic.jsonl \
  --bundle-root crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8 \
  --checkpoint-root <pinned-base-checkpoint-dir> \
  --tokenizer research/14_phase3b_backbone_parity_results/backbone_runtime/tokenizer/tokenizer.json \
  --backend mlx-fp32 --host "<host label>" --commit <hash> \
  --output-dir bench-output/joint-choice
```

For CPU, omit `--features mlx` and use `--backend cpu`. Custom workloads use
the usual Choice row schema plus `gold`, `task`, and `source_group`. They must
offer at least two real options and supply a non-empty `__none__` description.
An optional `split` identifying a final partition is rejected. Group related
source documents, generated cases, and paraphrases under one `source_group`.
Every prompt must fit 1,792 tokens in every renderer; no truncation is allowed.

The report includes full accuracy, answerable ranking accuracy, macro recall
over task/label pairs, NLL (probability floor `1e-15`), multiclass Brier,
fixed 10-bin ECE, none recall, false-none rate, and fixed risk/coverage points.
Paired accuracy/NLL/Brier deltas use 2,000 source-group bootstrap samples with
seed 29160717. Order sensitivity is reported for the coupled forward/reversal
change and separately for the position factor (fixed codes) and the code
factor (fixed positions). Per-row predictions, the workload digest, executable
digest, backend, arithmetic identity, checkpoint revision, host, and commit
accompany the summary. Execution order rotates across rows after one warmup.

These times cover full-forward scoring and readout, excluding prompt preparation,
model load, admission, wire conversion, and file writes. Ensemble methods cost
the sum of their member passes. The independent baseline uses `repeated_full`,
so these times do not compare against the production scheduler's prefix reuse.
The joint arms change prompt and readout together; an improvement cannot be
attributed to joint context alone. The catalogue arm isolates that context but
shifts the head's input distribution. The authored panels test mechanisms and
do not establish broad task quality or production promotion.

### Experimental joint-distribution calibration

`calibrate-choice` fits post-hoc calibration for the forward joint letter
distribution on one labeled partition and evaluates the locked parameters on a
disjoint gate partition. Three arms are fitted by deterministic NLL
minimization (grid search plus golden-section refinement, no randomness): a
positive temperature, a semantic-none logit offset, and a coordinate-descent
combination of both. A temperature alone preserves the winning class; the none
offset can change rejection but cannot supply missing evidence. The raw
distribution is scored alongside as the unlocked control. The tool refuses
partitions that share source groups or row ids, and records the fitted
parameters, in-sample metrics, gate metrics (NLL, Brier, ECE, none recall,
false-none rate, fixed risk/coverage points), paired deltas, and the same
locked parameters applied to the gate's reversed render as an order-transfer
check. This is an offline diagnostic; it does not register a model or change
daemon defaults.

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --release -p openkind-bench \
  --features mlx -- calibrate-choice \
  crates/openkind-bench/fixtures/joint_calibration_diagnostic.jsonl \
  crates/openkind-bench/fixtures/joint_gate_diagnostic.jsonl \
  --bundle-root crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8 \
  --checkpoint-root <pinned-base-checkpoint-dir> \
  --tokenizer research/14_phase3b_backbone_parity_results/backbone_runtime/tokenizer/tokenizer.json \
  --backend mlx-fp32 --host "<host label>" --commit <hash> \
  --output-dir bench-output/joint-calibration
```

Partition contracts match `compare-choice`. Fitting uses the forward render of
the calibration partition only; all parameters are fixed before any gate row
is scored. Improvement claims must come from the gate, not the calibration
partition, and transfer across order is evidence but not qualification.

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
  `decisions_per_second` / `input_tokens_total` / `cpu_time_seconds` /
  `avg_cpu_percent`, peak resident bytes, a `host_hardware` block
  (model identifier, CPU brand, logical cores, total memory), a `context`
  block with the engine's frozen per-sequence token budgets, and
  `cross_strategy_answer_parity_clean`. CPU fields diff process-wide
  user+system time across the timed region only; percentages exceed 100
  when multiple threads run. `scripts/build-recommendation-data.py`
  aggregates summaries into the recommendation dataset used by
  model-recommendation work.
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

## Dataset accuracy evaluation

`openkind-bench dataset` adds labeled public-dataset evaluation, the
model-quality evidence class the timing harness does not produce. The
methodology follows the [University of Bonn suite](#prior-art-comparability-jev-benchmarking)
(Deußer et al., arXiv 2609.37647): frozen zero-shot request templates
ported verbatim from their MIT harness at commit `6bbdeb3`, one request per
example with gold labels, `dev` splits for prompt and threshold work and
`eval` splits for reported numbers, identical-request deduplication, and
deterministic hash-ordered rows so smaller `--limit` values are subsets of
larger ones.

Acquisition lives in `crates/openkind-datasets` and is always explicit and
networked (`dataset pull`, `dataset pin`); tests and builds never download.
Downloads come from the Hugging Face Hub pinned to exact 40-hex revisions of
both the repository `main` branch and the `refs/convert/parquet` conversion
branch, verified against per-file sizes and SHA-256 digests from the
checked-in registry (`crates/openkind-datasets/registry/v1/datasets.json`),
and installed under `OPENKIND_DATASETS_DIR` (platform data dir by default)
as content-addressed blobs hard-linked into locked installs — the
`openkind-model-store` discipline applied to datasets. Gated repositories
authenticate with `HF_TOKEN`, `HF_TOKEN_PATH`, or the token file written by
`hf auth login` ([`ENV.md`](ENV.md#benchmark-dataset-variables)). Dataset
bytes are never committed here and never redistributed: the registry pins
identities, revisions, sizes, and digests only.

The curated core set covers all three primitives with fully open access —
Choice: `sst2`, `ag_news`, `banking77` (77 options), `clinc150` (151
options including out-of-scope), `arc`, `hellaswag`, `winogrande`,
`commonsense_qa`; Noul: `boolq`, `paws`; Score: `stsb`, `sst5`. `dataset
list` shows install state, license, and gating per dataset.

```bash
# List curated datasets and install state
cargo run -p openkind-bench -- dataset list

# Download and verify a pinned dataset (explicit, resumable, networked)
cargo run -p openkind-bench -- dataset pull sst2

# Materialize a labeled workload (dev split for prompt/threshold work)
cargo run -p openkind-bench -- dataset build sst2 --split eval --limit 100

# Score and report accuracy through the normal request path (any engine)
cargo run --release -p openkind-bench -- dataset eval sst2 \
  --engine qwen35 --bundle-root <dir> --checkpoint-root <dir> \
  --tokenizer <json> --host "<label>" --commit <hash> --output-dir bench-output

# Noul datasets: tune the decision threshold on dev, apply it to eval
cargo run --release -p openkind-bench -- dataset eval boolq --tune_threshold \
  --engine ... --host "<label>" --commit <hash>

# Re-check installed digests; resolve new upstream revisions for a curated
# definition and print the registry entry to commit after review
cargo run -p openkind-bench -- dataset verify sst2
cargo run -p openkind-bench -- dataset pin sst2
```

The report (`openkind-dataset-eval/v1`, written to
`dataset-eval-<name>-<split>.json` and printed) carries dataset provenance
(repository, both revisions, license, gating, template identity), engine
provenance, workload/prediction/summary/executable SHA-256 bindings — the
predictions file must match the summary's digest binding — and the metrics.
Choice rows report accuracy with a bootstrap interval over source groups,
answerable ranking accuracy, macro-F1, per-class recall, NLL, multiclass
Brier, 10-bin ECE, AURC, accuracy at 50%/80% coverage, none recall and
false-none rate. Noul rows add AUROC, binary Brier and ECE, class recalls,
and the optional dev-tuned threshold arm. Score rows report Spearman and
Pearson correlation, MAE, level accuracy, and NLL.

Interpretation rules:

- Reports are **model-quality evidence on a pinned public dataset**; they
  never promote a model, profile, or policy, and timing claims stay with
  the `score` harness.
- Choice rows always carry the injected `__none__` option (workspace
  invariant), which the paper's Jev requests did not force. `accuracy`
  includes it; `answerable_ranking_accuracy` is the externally comparable
  figure. This is stated on every report.
- Templates are frozen: a wording change is a new template version, and
  comparison against the paper's published numbers is only meaningful with
  the same request wording.
- The core set intentionally avoids gated and non-commercial-licensed
  datasets. `dataset pin` will not resolve them until a definition exists;
  when added, their license flags travel in the registry entry and their
  bytes remain download-only.
- Contamination still applies to long-established benchmarks (the paper's
  MMLU anomaly); treat absolute numbers on familiar datasets as weak
  evidence and prefer fresh or authored panels for promotion decisions.

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

## Proxy-cache BGE encoder component benchmark

`encoder-embedding` is a sentence encoder used by the proxy-cache, not a
`DecisionEngine`; it cannot be measured by the decision workload harness. The
opt-in `encoder_embedding_bench` example measures one state at a time through
`TextEmbedder::encode`, including tokenization and forward execution while
excluding model load. It cycles three checked-in support-ticket state strings,
records their SHA-256, warms the backend, and reports p50/p95 latency and
embeddings per second. Run CPU and MLX in separate fresh processes:

```bash
cargo run --release -p openkind-backends --example encoder_embedding_bench -- \
  --model-root "$OPENKIND_ENCODER_EMBEDDING_MODEL_ROOT" --backend cpu \
  --host "<hardware label>" --commit <hash> --working-tree-dirty true

SDKROOT=$(xcrun --show-sdk-path) cargo run --release -p openkind-backends \
  --features mlx --example encoder_embedding_bench -- \
  --model-root "$OPENKIND_ENCODER_EMBEDDING_MODEL_ROOT" --backend mlx \
  --host "<hardware label>" --commit <hash> --working-tree-dirty true
```

The example requires host, source commit, and working-tree state attribution.
It uses the registry-pinned BGE revision and never downloads weights. For
peak RSS, build the example first and run its executable under
`/usr/bin/time -l` on macOS, keeping model loading in the measured process.
This component result does not measure proxy routing, student-cache quality,
or end-to-end decision throughput. The paired run and CPU/MLX parity evidence
are recorded in
[`benchmarks/2026-09-30-encoder-embedding/`](benchmarks/2026-09-30-encoder-embedding/README.md).

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

### Laya decision-encoder single-shot records (2026-09-27)

One `openkind-bench score` run per pinned laya profile over the standard
seeded workload (same shape777 fixture, sha256
`be397bfc48209c8f7379d0d76ccbe3d3e7dca3269724928f0868cf872f4c8b01`, 777
rows, 37 state groups, `--reps 1`). Same host and attribution as the
surveyed-family records; commit `d80c685` at run time. The readout was
qualified against the Python reference (`laya` 0.3.21, CPU fp32) to
`<= 5.3e-5` max probability delta with zero selection flips over the shared
15-case fixture set; these runs are still **request-path timing records
only** — no M2 model-quality evidence, and the upstream model card itself
reports the base checkpoints near chance zero-shot on typed decisions.

| Engine (profile) | Backbone | p50 request | Decisions/s | Peak RSS | Model load |
|---|---|---|---|---|---|
| `laya-english` (`c8ea29bf1e33a343c4b7`) | convaiinnovations/laya (ModernBERT-large, typed-decision head) fp32 | 353.80 s | 2.20 | 2.56 GB | 1.83 s |
| `laya-multilingual` (`f4064eb56fb7f7d325e1`) | convaiinnovations/laya-multilingual (mmBERT-base, typed-decision head) fp32 | 151.51 s | 5.13 | 2.61 GB | 1.81 s |
| `laya-typed-decisions` (`9d28cfa9567902801ed1`) | convaiinnovations/laya-typed-decisions (fine-tuned ModernBERT-large) fp32 | 327.84 s | 2.37 | 2.57 GB | 1.80 s |

The large-checkpoint profiles sit between `encoder-instruct-label`
(ModernBERT-base, 4.00 dec/s) and `decoder-logit-llm` (0.42 dec/s) on the
same workload; the mmBERT-base profile is the fastest full-precision encoder
record at 5.13 dec/s. Peak RSS covers the fp16-shard mmap upcast to fp32
weights plus forward scratch.

### Laya MLX encoder-backend records (2026-09-28)

First campaign for the laya family's MLX/Metal backend
(`families/laya/mlx/`, feature `mlx`): the same digest-locked pinned
checkpoints executed as mlx-rs FP32 arrays, with the candle CPU engines
re-run in the same binary so both backends share one telemetry schema
(including per-strategy `input_tokens_per_second`). Standard shape777
workload (same sha256, 777 rows, 37 state groups, `--reps 1`, warm
process); host and attribution match the records above; commit `b8c80ae`
at run time. Parity is frozen in `tests/laya_parity.rs` module
`mlx_replay`: golden-fixture replay per profile with zero selection flips
and maximum probability drift 2.5–7.2e-6 (budget 0.005).

| Engine | p50 request | Decisions/s | Input tokens/s | Peak RSS | Model load |
|---|---:|---:|---:|---:|---:|
| `laya-english` (candle CPU fp32) | 348.16 s | 2.23 | 455 | 2.75 GB | 1.84 s |
| `laya-english-mlx-fp32` | 31.93 s | 24.34 | 4,964 | 2.12 GB | 2.77 s |
| `laya-multilingual` (candle CPU fp32) | 146.64 s | 5.30 | 1,075 | 2.80 GB | 1.95 s |
| `laya-multilingual-mlx-fp32` | 13.78 s | 56.37 | 11,431 | 2.94 GB | 3.54 s |
| `laya-typed-decisions` (candle CPU fp32) | 333.72 s | 2.33 | 475 | 2.76 GB | 1.83 s |
| `laya-typed-decisions-mlx-fp32` | 32.68 s | 23.77 | 4,849 | 2.13 GB | 2.87 s |

The MLX backend is ~10.2–10.9× faster than the candle CPU path on every
profile at equal or lower peak RSS, with sub-one-core average CPU
utilization (66.9–90.4%) while the GPU computes. Context budgets are
backend-independent (512/192 and 1024/256). Request-path timing only —
task quality is not claimed, and the parity gates cover readout agreement
with the CPU oracle, not decision accuracy. Recorded in
[`benchmarks/2026-09-28-laya-mlx-campaign/`](./benchmarks/2026-09-28-laya-mlx-campaign/README.md).

### Registry MLX-preferred campaign (2026-09-28)

One `openkind-bench score` campaign over all four
[`registry/v1`](../registry/v1/catalog.json) catalog models on the standard
shape777 workload, choosing a parity-qualified MLX path where one exists and
CPU otherwise; runs 13:40–19:00 on the same host as the records above. This
campaign introduced the summary telemetry the tables below now carry:
per-strategy `cpu_time_seconds` / `avg_cpu_percent` (timed region only), a
`host_hardware` block, and per-engine `context` token budgets. Full tables,
the cross-strategy parity investigation, and the machine-readable
[`recommendation-data.json`](./benchmarks/2026-09-28-registry-mlx-campaign/recommendation-data.json)
are in the campaign README. Early summaries carry a `d80c685` commit label
from a stale snapshot; the actual HEAD at run time was `b8c80ae` (the
campaign README carries the correction).

`qwen35-state-first` (`a047d6802c3f06f085b8`), pinned
`Qwen/Qwen3.5-4B-Base` @ `1001bb4d…`:

| Backend / strategy | p50 request | Decisions/s | Avg CPU % | Peak RSS | Model load |
|---|---:|---:|---:|---:|---:|
| `qwen35-mlx-fp32` `nested_batched` (qualified) | 314.20 s | 2.473 | 98.7 | 12.09 GB | 20.69 s |
| `qwen35-mlx-fp32` `choose_strategy` (qualified) | 316.10 s | 2.458 | 82.0 | 12.09 GB | 20.74 s |
| `qwen35-mlx-fp32` `repeated_full` (qualified) | 2411.66 s | 0.322 | 130.3 | 12.09 GB | 22.48 s |
| `qwen35-mlx-bf16` `repeated_full` (unqualified candidate) | 2184.99 s | 0.356 | 121.0 | 4.86 GB | 19.26 s |
| `qwen35-native-cpu` `choose_strategy` | 3531.83 s | 0.220 | 103.9 | 18.62 GB | 16.51 s |

The qualified MLX FP32 path is ~11.2× faster than the CPU oracle on the same
workload and strategy decision (316.10 s vs 3531.83 s `choose_strategy`) at
12.09 GB against 18.62 GB peak RSS. The BF16 candidate adds little fresh
scoring speed (~1.10×) but holds 2.5× less memory; it remains gated out of
decision use. The MLX sweep's cross-strategy wire comparison is not
byte-exact: maximum scalar/probability delta 3.12e-05 over all 777 answers
with zero argmax and zero selected-option changes — the bounded numerical
divergence documented in the 21 September smoke record, not a parity-gate
failure. The laya rows (CPU from this campaign, MLX from the concurrent laya
campaign above) and the per-model MLX verdicts are in the campaign README.
Recorded in
[`benchmarks/2026-09-28-registry-mlx-campaign/`](./benchmarks/2026-09-28-registry-mlx-campaign/README.md).

### Decoder-logit-qwen35 smoke record (2026-09-27)

One `openkind-bench score` run for the pinned
[`decoder-logit-qwen35`](./families/decoder-logit-qwen35.md) profile
(`415bcf4a064e6dadcf85`, `alibiserikbay/JevK5` at
`c4f7fdb3aeab5582336406e78d3bef11bf98833d`). This is a **smoke-scale,
single-sample record, not comparable to the shape777 family table**: the
standard 37-state workload costs a 4B-parameter fp32 CPU forward per
question over ~3,200-token prompts (two orders of magnitude more compute
than the 0.5B-0.6B records), so the run uses a generated 1-state × 21-criteria
reduction of the same seeded generator (sha256
`a5539013310ea996c6aee1ba5f9d1f782230cd58572cb69e2aab091b62f4592e`, 21 rows,
`--reps 1`). Host and attribution match the surveyed-family records; commit
`d80c685` at run time.

| Engine (profile) | Backbone | Workload total | Decisions/s | Peak RSS | Model load |
|---|---|---|---|---|---|
| `decoder-logit-qwen35` (`415bcf4a064e6dadcf85`) | alibiserikbay/JevK5 (merged LoRA on Qwen3.5-4B, letter-logit readout) fp32 | 130.20 s | 0.161 | 8.04 GB | 15.19 s |

Readout evidence (recorded with the profile, not a quality claim): rendered
prompt bytes are token-identical to the reference `jevk5` runtime, and an
fp32 PyTorch cross-check (`jevk5` 0.3, CPU) agrees with the native readout
to `<= 1e-6` maximum probability delta over the nine-case golden fixture,
including a 17-option knockout question and a JSON-object evidence payload.

## Recorded runs

| Record | Engine | Status |
|---|---|---|
| [`benchmarks/2026-09-30-jevbench-expansion/`](./benchmarks/2026-09-30-jevbench-expansion/) | plumb-4b, decider-4b | CPU timing recorded: the two JevBench-v1.5.4 profiles (`plumb-4b` single-read letter-logit, `decider-4b` slot-logit with isolated score levels) on the smoke workload plus the Plumb 96-case choice diagnostic; golden-fixture replays reported for both and MLX replay reported for Plumb, with logs and measured source diff unarchived; request-path timing only, no model-quality claim; see the family pages [`decoder-logit-qwen35`](./families/decoder-logit-qwen35.md) and [`decider`](./families/decider.md) |
| [`benchmarks/2026-09-29-mlx-counterparts/`](./benchmarks/2026-09-29-mlx-counterparts/) | encoder-instruct-label-mlx-fp32, decoder-logit-qwen35-mlx-fp32 (+ encoder-instruct-label CPU re-run) | Complete — first MLX backends for the GLiClass and JevK5 surveyed families on the standard shape777 workload with frozen golden-fixture parity gates (max probability drift 4.487e-6 / 1.003e-6, zero selection flips); request-path timing only, no model-quality claim; see the survey-family backends section in [`MLX.md`](MLX.md) |
| [`benchmarks/2026-09-28-laya-mlx-campaign/`](./benchmarks/2026-09-28-laya-mlx-campaign/) | laya-english-mlx-fp32, laya-multilingual-mlx-fp32, laya-typed-decisions-mlx-fp32 (+ candle CPU re-runs) | Complete — first MLX encoder-backend campaign on the standard shape777 workload with frozen golden-fixture parity gates (max probability drift 7.2e-6, zero selection flips); request-path timing only, no model-quality claim; see the laya section below |
| [`benchmarks/2026-09-28-registry-mlx-campaign/`](./benchmarks/2026-09-28-registry-mlx-campaign/) | qwen35-mlx-fp32, qwen35-mlx-bf16 (unqualified candidate), qwen35-native-cpu, laya CPU ×3 | Complete — registry-wide MLX-preferred campaign over all four catalog models on the standard shape777 workload with the new CPU/host/context telemetry and recommendation dataset; see the registry-campaign section above |
| [`benchmarks/2026-09-27-decoder-logit-qwen35/`](./benchmarks/2026-09-27-decoder-logit-qwen35/) | decoder-logit-qwen35 | Complete — smoke-scale single-state record for the JevK5 profile with reference-parity fixture; request-path timing only, no model-quality claim; see the smoke-record section above |
| [`benchmarks/2026-09-27-laya/`](./benchmarks/2026-09-27-laya/) | laya-english, laya-multilingual, laya-typed-decisions | Complete — single-shot decision-encoder records on the standard shape777 workload with reference-parity fixtures; request-path timing only, no model-quality claim; see the laya section below |
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

## Jev-Style 2B MLX survey

The pinned upstream MLX runtime for
[`Jev-Style-2B-Decision-v3-MLX`](./families/jev-style.md) passed manifest
verification and local decision smoke on an Apple M4 Max. Three-call p50
timings were 0.291 / 0.568 s for one / ten questions at 878 state tokens,
1.207 / 1.552 s at 3,950 tokens, and 9.040 / 9.591 s at 24,436 tokens in
BF16. The 8-bit folder was slower in these local cells and used less MLX
memory. Same-host OpenKind Qwen3.5-4B timings and the upstream M1 Max reference
are recorded in the
[benchmark report](./benchmarks/2026-09-27-jev-style-2b-mlx/README.md).

The OpenKind 4B protocol accepts only 1,792 tokens per full candidate
sequence, so it cannot supply a matched long-context comparison. These
measurements establish local execution and request-path timing only; they do
not reproduce model parity or task quality.

## External evaluation and submission

When implementation, native parity qualification, and release readiness are
finished, submit the model for external evaluation:

- **BenchmarkHeaven Custom Evaluation**:
  <https://benchmarkheaven.com/jev-models/custom-evaluation> — submission portal
  for independent Jev-compatible model evaluation.
- **`jevbench` Harness**:
  [`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench) —
  evaluation suite and benchmark harness for Jev models.
- **University of Bonn `jev-benchmarking`**:
  [`AppliedMachineLearning-Lab/jev-benchmarking`](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking)
  — the 37-dataset zero-shot suite and released response corpus from
  Deußer et al. (arXiv:2609.37647); see the
  [comparability section](#prior-art-comparability-jev-benchmarking) above.
