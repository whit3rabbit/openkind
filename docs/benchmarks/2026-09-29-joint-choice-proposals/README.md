# Joint-choice proposal runs

The three proposed local scoring comparisons from
[`docs/RESEARCH.md`](../../RESEARCH.md) were executed on 2026-09-29 against the
same pinned checkpoint and FP32 arithmetic as the [first joint-choice
diagnostics](../2026-09-29-joint-choice/README.md). Outcomes:

1. **All-options catalogue with the fitted head improves decisively and is
   implemented** as the `catalogue_state_first/v1` experimental renderer and
   the `catalogue_fitted` method. It is not registered and does not change
   daemon defaults.
2. **Position, not letter code, is the dominant output-slot bias.** No
   ensemble was adopted: the fixed four-render ensemble never beat the single
   forward render while costing four forwards. The separation probe methods
   and per-factor reporting are implemented in `compare-choice`.
3. **Locked temperature scaling improves the disjoint gate; the none offset
   does not.** The fitting and locked-gate tooling is implemented as
   `openkind-bench calibrate-choice`. No calibration constant became a runtime
   default, and the fitted temperature failed the reversed-render transfer
   check, so it is recorded as render-specific evidence only.

## Configuration

- Host: Apple M4 Max (`Mac16,5`), 14 logical cores, 36 GiB, macOS arm64.
- Base commit: `c17fec804f4b4e390e0ab6a1eff959fe720477a7`, plus the
  proposal-campaign working-tree change. All three summaries identify the
  measured executable as
  `7b6485d4c0939b8bdc61dfe26e812e1d08ab1bdde817a18b00651053b589d3c1`. The
  repository checkout was dirty and other sessions landed commits during the
  campaign; the executable digest binds the measured binary.
- Checkpoint: `Qwen/Qwen3.5-4B-Base`, revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`, verified in place.
- Backend: MLX FP32 ReferenceOps, MLX 0.32.2. One loaded checkpoint serves
  every method within each fresh process.
- Baseline: profile `a047d6802c3f06f085b8`, frozen state-first renderer,
  fitted score-summary head, temperature `1.8186799910442777`.
- Catalogue arm: `catalogue_state_first/v1`. The shared state root and each
  candidate continuation are byte-identical to the frozen renderer; the
  question branch gains a canonical catalogue of every option description,
  including none. Fitted head and frozen temperature are unchanged.
- Joint arms: `joint_option_letter/v1`, raw temperature 1.0 over the verified
  space-prefixed letter rows A..P and Z. `joint_text_rotate` rotates the full
  displayed list, none included, with codes bound to their options;
  `joint_code_rotate` permutes the letter codes at fixed positions and moves
  none off `Z`, naming the none letter in the instruction. Ensembles average
  remapped distributions: `joint_average` (forward + reversal),
  `joint_pair_text` (forward + text rotation), `joint_ensemble_four` (all four
  joint renders).

## Proposal 1: all-options context with the fitted head

The [96-case panel](../../crates/openkind-bench/fixtures/joint_choice_diagnostic.md)
has 24 source groups and 24 semantic-none cases:

| Method | Correct | NLL | Brier | ECE, 10 bins | Mean scorer seconds/case |
|---|---:|---:|---:|---:|---:|
| Independent fitted | 71/96 | 0.636 | 0.317 | 0.157 | 1.088 |
| **Catalogue fitted** | **93/96** | **0.144** | **0.066** | **0.080** | 2.540 |
| Joint forward | 96/96 | 0.226 | 0.081 | 0.185 | 0.678 |
| Joint reversed | 87/96 | 0.266 | 0.116 | 0.131 | 0.672 |

Catalogue minus baseline, source-group bootstrap: accuracy +22.9 percentage
points [+17.7, +28.1], NLL -0.492 [-0.600, -0.382], Brier -0.252 [-0.294,
-0.209]. None recall rises from 10/24 to 24/24 and false-none falls from
10/72 to 3/72. Unlike the joint arms, the catalogue arm also improves ECE
(0.080 versus 0.157), because the fitted head keeps its calibrated
temperature. The cost is prompt length: 37,620 scored input tokens versus
19,980 for the independent control, and about 2.3x warm scorer seconds per
case on these panels.

On the [24-case reference-card intervention](../../crates/openkind-bench/fixtures/joint_reference_card_diagnostic.md),
the fitted head selects the forbidden reference option 23/24 times under the
frozen renderer but **0/24** under the catalogue renderer, and rises from
0/24 to 18/24 correct with NLL 4.596 -> 1.238. Seeing the rival descriptions
lets the old head use the reference card without replacing the head. See the
[card summary](reference-card/summary.json) and
[card predictions](reference-card/predictions.jsonl).

Decision: implemented as an experimental renderer and comparison method; not
a production replacement. Promotion would require natural-task panels and
request-path timing, and the 2.3x scorer cost sets the terms for that timing.

## Proposal 2: separating text position from output codes

Selection flips and maximum probability shifts against the forward render:

| Factor | 96-case panel | Reference-card panel |
|---|---|---|
| Coupled order + codes (forward vs reversed) | 9 flips, 0.312 | 17 flips, 0.390 |
| Position factor, codes fixed (vs text-rotate) | 7 flips, 0.355 | 7 flips, 0.288 |
| Code factor, positions fixed (vs code-rotate) | 0 flips, 0.236 | 9 flips, 0.294 |

On the rule panel, text position explains most of the coupled effect and the
letter codes alone change no selections. On the card panel both factors flip
selections, so letter identity is not universally negligible. Averaged
ensembles do not buy quality: on the 96-case panel `joint_ensemble_four`
scores NLL 0.248 / accuracy 96/96 at 4 forwards per case, worse proper scores
than the single forward render (NLL 0.226, 1 forward), and `joint_pair_text`
scores NLL 0.255. On the card panel the four-render ensemble collapses to
4/24 correct. No ensemble is adopted; `joint_forward` remains the
representative single joint render. The probe methods and per-factor
reporting stay available in `compare-choice`.

## Proposal 3: calibrating the joint distribution

The [64-case calibration partition](../../crates/openkind-bench/fixtures/joint_calibration_diagnostic.md)
(16 groups, 16 none cases, workload `55107f1a01ac8c74...`) was fitted first;
all parameters were then locked before the disjoint
[64-case gate partition](../../crates/openkind-bench/fixtures/joint_gate_diagnostic.md)
(16 groups, 16 none cases, workload `a7d5d294e8a04dd4...`) was scored. Tasks,
records, and source groups are disjoint between the partitions and from the
historical panels; both panels were authored before the first scoring run.

Fitted values: temperature `0.3984576794747966`; none offset
`-0.7873760036956927`; combined arm (coordinate descent) temperature
`0.4255292134243418`, offset `-0.8053356766602864`.

Gate results, forward render:

| Arm | Correct | None recall | False none | NLL | Brier | ECE | Accepted @0.9 (errors) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 63/64 | 16/16 | 1/48 | 0.178 | 0.060 | 0.143 | 27 (0) |
| **Temperature (locked)** | **63/64** | **16/16** | **1/48** | **0.035** | **0.014** | **0.025** | **47 (0)** |
| None offset (locked) | 63/64 | 15/16 | 0/48 | 0.181 | 0.065 | 0.128 | 32 (0) |
| Temperature + offset | 63/64 | 15/16 | 0/48 | 0.052 | 0.026 | 0.019 | 47 (0) |

The temperature arm improves paired NLL by -0.143 [-0.191, -0.099] and Brier
by -0.045 [-0.069, -0.024] with zero selection changes (argmax is preserved),
and extends zero-error accepted coverage from 27 to 47 of 64 at threshold
0.9. The none offset arm does not improve: paired NLL +0.002 [-0.008, +0.014]
and none recall drops to 15/16, so the offset is not adopted. The raw joint
distribution is underconfident on this render, which is why the fitted
temperature is below one.

The transfer check bounds the claim: applying the same locked parameters to
the gate's reversed render degrades every arm (temperature NLL 0.381 versus
raw-reverse 0.332), so the fitted temperature is a property of the forward
render, not a portable constant. Decision: the temperature arm passes its
locked gate, but because the improvement is render-specific and the panels
are authored diagnostics, no calibration constant becomes a runtime default.
`calibrate-choice` is implemented for repeating the fit/gate protocol on
future partitions, including natural-task ones.

## Timing and validation boundaries

Times measure warmed full-forward scoring and readout, after prompt
preparation, with execution order rotating across rows after one warmup.
Model load, warmup, admission, wire conversion, and writes are excluded.
Ensemble methods report the sum of their member passes. The baseline uses
independent full forwards, so times do not compare against the production
scheduler's prefix reuse. All numbers are exploratory diagnostics from one
local host, not throughput qualifications.

The [CPU/MLX subset comparison](cpu_mlx_parity.json) replays the four-case
parity subset through all nine methods on both backends. Maximum probability
delta is 1.07e-5 (independent fitted) and at most 6.6e-7 for the new catalogue
path, with zero selection changes, within the unchanged 0.005 tolerance.

Each summary binds its prediction file context through the workload SHA-256,
executable SHA-256, arithmetic identity, and host/commit attribution. NLL
uses a probability floor of 1e-15; Brier sums squared errors across all
offered labels including none; bootstrap resampling keeps source groups
together (2,000 replicates, seed 29160717).

## Reproduce

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --release -p openkind-bench \
  --features mlx -- compare-choice \
  crates/openkind-bench/fixtures/joint_choice_diagnostic.jsonl \
  --bundle-root crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8 \
  --checkpoint-root <pinned-base-checkpoint-dir> \
  --tokenizer research/14_phase3b_backbone_parity_results/backbone_runtime/tokenizer/tokenizer.json \
  --backend mlx-fp32 --host "<host label>" --commit <hash> \
  --output-dir <output-dir>
```

For the reference-card run substitute
`crates/openkind-bench/fixtures/joint_reference_card_diagnostic.jsonl`. For
the calibration run use `calibrate-choice` with
`joint_calibration_diagnostic.jsonl` and `joint_gate_diagnostic.jsonl` as
documented in [`docs/BENCHMARKS.md`](../../BENCHMARKS.md#experimental-joint-distribution-calibration).
All commands load only local pinned artifacts.

The production scorer, daemon defaults, and the frozen profile remain
unchanged. Qualifying any surviving candidate still requires natural-task
panels, request-path timing against the current scheduler, and the existing
qualification gates.
