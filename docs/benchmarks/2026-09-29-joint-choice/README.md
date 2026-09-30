# Joint-option scoring diagnostics

Joint scoring improves these authored Choice panels, but remains an offline
experiment. It does not justify replacing the fitted scorer: option order
changes selections, ECE worsens on the simpler panel, and averaged joint
scoring misses every semantic-none case in the reference-card panel.

The [96-case panel](../../../crates/openkind-bench/fixtures/joint_choice_diagnostic.md)
tests facts, negation, numeric thresholds, and measurements supplied in options.
The separate [24-case reference-card intervention](../../../crates/openkind-bench/fixtures/joint_reference_card_diagnostic.md)
changes only a rival option's code mapping. Both use balanced gold labels.
These are templated mechanism diagnostics, not representative task-quality
gates. The card panel was authored after the first panel's run started, without
changing the prompt or readout. Historical calibration gates and final
partitions were not opened.

## Configuration

- Host: Apple M4 Max (`Mac16,5`), 14 logical cores, 36 GiB, macOS arm64.
- Base commit: `ebf2b06d493a17e9f7bd4a50ddbc769ef2356f24`, plus the joint-scoring
  working-tree change. Both summaries identify the measured executable as
  `3484e98187ec5f43a6384ba893d49504df7758432c6a059b18c361d8ecc3cc89`.
- Checkpoint: `Qwen/Qwen3.5-4B-Base`, revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`, verified in place.
- Backend: MLX FP32 ReferenceOps, MLX 0.32.2. The same loaded checkpoint serves
  all methods within each fresh process.
- Baseline: profile `a047d6802c3f06f085b8`, frozen state-first renderer,
  fitted score-summary head, temperature `1.8186799910442777`.
- Joint: `joint_option_letter/v1`, one prompt containing all options, selected
  tied output rows for space-prefixed A..P and Z, temperature 1.0. No generation,
  training, temperature fit, threshold selection, or production registration.
- Real options sort by label. The reversed pass reverses their order and letter
  assignment; Z remains last. Averaging remaps both distributions to option IDs
  before taking their mean, retaining the full none mass.

Prompt, readout, and temperature differ between baseline and joint. This
comparison cannot isolate the causal contribution of joint context.

## Measured quality

The 96-case run has 24 source groups and 24 semantic-none cases:

| Method | Correct | NLL | Brier | ECE, 10 bins | Mean scorer seconds |
|---|---:|---:|---:|---:|---:|
| Independent fitted | 71/96 | 0.636 | 0.317 | 0.157 | 0.970 |
| Joint forward | 96/96 | 0.226 | 0.081 | 0.185 | 0.619 |
| Joint reversed | 87/96 | 0.266 | 0.116 | 0.131 | 0.593 |
| Joint average | 96/96 | 0.241 | 0.093 | 0.192 | 1.212 |

Answerable ranking accuracy is already 71/72 for the fitted scorer, versus
72/72 for every joint method. Most full-accuracy gains come from rejection:
none recall rises from 10/24 to 24/24 with forward or averaged joint scoring,
and false-none falls from 10/72 to 0/72. Reversal changes 9/96 selections;
the largest probability shift is 0.312.

For joint average minus baseline, the source-group bootstrap accuracy delta is
+26.04 percentage points, with a descriptive 95% interval of
[+19.79, +32.29]. NLL delta is -0.395 [-0.549, -0.253], and Brier delta is
-0.224 [-0.276, -0.172]. ECE increases despite better proper scores, so these
results do not establish improved calibration. See [summary.json](summary.json)
and [predictions.jsonl](predictions.jsonl) for per-task metrics, class recall,
and fixed risk/coverage points.

The card intervention has six source groups and six semantic-none cases:

| Method | Correct | NLL | Brier | ECE, 10 bins | Mean scorer seconds |
|---|---:|---:|---:|---:|---:|
| Independent fitted | 0/24 | 4.596 | 1.621 | 0.784 | 2.624 |
| Joint forward | 7/24 | 1.262 | 0.715 | 0.226 | 1.593 |
| Joint reversed | 18/24 | 0.892 | 0.499 | 0.332 | 1.567 |
| Joint average | 13/24 | 1.016 | 0.576 | 0.256 | 3.160 |

The fitted scorer selects the forbidden reference option 23 times. Its
conditional probabilities among the three unchanged action options vary by
at most `2.22e-16` when the reference card changes, confirming its inability
to use the card to change their relative ranking. Joint scoring responds to
the card, but forward, reversed, and averaged scoring still select the
reference option 17, 6, and 11 times respectively. Reversal changes 17/24
selections, with maximum probability shift 0.390. Averaging has 13/18 correct
answerable rankings and 0/6 none recall. Forward scoring recognizes 3/6 none
cases; reversed scoring recognizes none. See the
[card summary](reference-card/summary.json) and
[card predictions](reference-card/predictions.jsonl).

## Timing and validation boundaries

Times measure warmed full-forward scoring and readout, after prompt
preparation, with method order interleaved across rows. Model load, warmup,
admission, wire conversion, and output writes are excluded. The baseline uses
independent `repeated_full` forwards; production prefix reuse is not measured.
Each joint average costs both joint passes, making it slower than the baseline
on both panels. These are exploratory timings from one local run, not a
throughput qualification. Load times are recorded separately in the summaries.

[validation.json](validation.json) binds both prediction files by SHA-256 and
records complete offered-option coverage, finite normalized probabilities,
correct gold joins, and exact remapped averaging. NLL uses a probability floor
of `1e-15`; Brier sums squared errors across all offered labels, including none.
Bootstrap resampling keeps source groups and paired predictions together
(2,000 replicates, seed 29160717). The intervals describe these authored groups.

On the [four-case subset](parity_subset.jsonl),
[CPU/MLX comparison](cpu_mlx_parity.json) has zero selection changes for all
four methods. Maximum probability deltas are `1.07e-5` for the fitted scorer,
`2.83e-6` forward, `6.31e-6` reversed, and `4.57e-6` averaged, within the
unchanged `0.005` tolerance. This subset covers one positive example per rule
task; it does not qualify every prompt or rejection case. Its
[CPU records](cpu-subset/summary.json) are numerical evidence only, collected
alongside workspace verification.

The [existing engine cross-check](engine_crosscheck.json) runs
`Qwen35DecisionEngine` through `score` on those same four cases. Its full
probability maps exactly equal the probe's fitted baseline, with no selection
changes. Thus the baseline replays the current scorer rather than a substitute
head. [Engine records](engine-crosscheck/summary-qwen35-mlx-fp32.json) use a cold
pass and are not part of the timing comparison above.

[Verification records](verification.json) cover formatting, workspace Clippy,
792 workspace tests, the locked bench battery, unchanged generated schemas,
and MLX Clippy. The complete MLX suite passes 328 tests on rerun. Its first run
fails an existing proxy-cache confidence assertion; both a focused rerun and
the complete rerun pass without code changes to that subsystem.

The [runtime diagnostic](runtime-diagnostic.txt) passes FP32 and BF16 primitive
checks. `qwen35_mlx_qualify --formal` refuses this shared dirty checkout, so no
formal runtime provenance receipt is emitted. This limits qualification;
it does not change the recorded numerical comparisons or establish BF16
checkpoint parity. These experiments use FP32 only.

## Reproduce

Use the [harness command](../../BENCHMARKS.md#experimental-joint-option-comparison)
with the 96-case fixture. For the second run, substitute
`crates/openkind-bench/fixtures/joint_reference_card_diagnostic.jsonl` and use
a separate output directory. Both commands load only local pinned artifacts.
The fixture contracts own their gold rules and hashes are in the summaries.

Further qualification needs a locked, admissible natural-task development and
gate panel, explicit order/label controls, rejection and calibration evidence,
and complete request-path timing against prefix reuse. The current scorer and
daemon defaults remain unchanged.
