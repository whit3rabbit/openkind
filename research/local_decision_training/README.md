# Local decision training

Start with [notebook 35](../35_local_decision_training.ipynb). It trains a
`Qwen/Qwen3.5-4B` decision LoRA on an A100, with an NF4 QLoRA path for L4.
The deployment target is a 16–32 GB Mac after merge, quantization and native
qualification. This is a v3 experiment, not a promoted model or a completed
4B training result. The [local design](../../docs/whitepaper/LOCAL_DECISION_DESIGN.md)
separates this pilot from native integration.

Upload the notebook at [Google Colab](https://colab.research.google.com/), select
an A100 GPU, and run all cells. Mount Drive when prompted. No prior experiment
folder, teacher API key or repository clone is needed. Training resumes from
the last verified optimizer checkpoint when rerun with the same configuration.

## What I would train on

Use a mixture of human-labeled language tasks, teacher-labeled decisions and
exact rule examples. The initial caps total 8,000 admitted training rows;
400 updates at effective batch 16 present 6,400 examples. Actual admitted counts
and the exposure budget are written to the run's manifest. These are pilot budgets,
not an empirically optimal mixture.

| Dataset | Default cap | Why it is included |
|---|---:|---|
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) | 1,000 | Distinguish supported, contradicted and insufficient-evidence claims |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | 1,000 | Preserve natural-language yes/no decisions grounded in passages |
| [Banking77](https://huggingface.co/datasets/legacy-datasets/banking77) | 1,000 | Dynamic intent candidates, option descriptions and missing-answer cases |
| [MultiRC](https://huggingface.co/datasets/aps/super_glue/viewer/multirc) | 1,000 | Combine evidence across sentences; evaluate each candidate independently |
| [SST-5](https://huggingface.co/datasets/SetFit/sst5) | 1,000 | Five-level ordinal judgments; optional via `include_sst5` |
| [Plumb decisions](https://huggingface.co/datasets/crh225/plumb-decisions/tree/718a9f006f3beaecfa7f66a819553f99ed7b94f6) | 2,000 | Already generated Qwen-teacher decisions covering policies, numerical reasoning, rubrics and exceptions |
| Notebook rule generator | 1,000 | Exact boundary cases, counterfactuals, missing facts and explicit ordinal rules |

Plumb is the practical first test of the proposed reasoning transfer: train a
small model against decisions produced by a larger reasoning teacher. Its labels
receive half the per-example loss weight of human or computed labels. Agreement
between two solutions from the same teacher can still preserve a shared mistake.
The notebook preserves soft distributions and does not feed teacher explanations
to the student. This transfers decision behavior; it does not establish that a
reasoning mechanism has been transplanted.

SST-5's mirror does not specify a license. MultiNLI and MultiRC retain
source-specific terms. The manifest records the available terms and source pins;
the repository's license does not relicense those datasets. SciQ is evaluation-only
and its card specifies CC-BY-NC-3.0.

## What the published recipes actually used

These are inspected recipes from the models discussed in the research dossier,
not a new ranking across incompatible leaderboards. Training-set disclosure and
benchmark scores are separate evidence.

| Model / pinned recipe | Training disclosure | Validation / evaluation disclosure |
|---|---|---|
| [JevK5 v0.2](https://github.com/allebee/jevk5/blob/v0.2.0/README.md) | 3,272 checked Qwen-teacher questions plus 3,272 human-labeled items drawn from MMLU-Pro, WANLI, MultiNLI, BoolQ, Banking77, ARC and CommonsenseQA | Temperature fitted on teacher questions from three domains absent from training; public JevBench is a separate reported evaluation |
| [Jeeves manifest](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/data/manifest.json) | Banking77, BoolQ, AG News, MNLI, SST-5, Yelp, TREC, DBpedia14, Amazon, IMDb, SemIf and WANLI, plus generated composition/contrastive cases | Explicit evaluation-only sources include MMLU, emotion, TweetEval offensive, QNLI, PAWS and SciQ |
| [Imajev-4B](https://huggingface.co/mohit67890/imajev-4b) | A broad human-labeled mixture, teacher decisions, hard-case mining and replay; later stages also contain image tasks | Multiple held-out and public suites, with explicit limitations and some release gate overrides. Its multimodal recipe is not reproduced here |
| [Plumb](https://huggingface.co/datasets/crh225/plumb-decisions/tree/718a9f006f3beaecfa7f66a819553f99ed7b94f6) | Public 5,014-row Qwen-teacher training split, independently solved twice by that teacher | The 131-row split named `test` set Plumb's calibration temperature. It is not an untouched test for that model |
| [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) | Private mixture of synthetic, teacher-supervised and labeled semantic tasks; a refinement stage mixes gold labels, gold-agreeing teacher distributions and replay | The card states training/validation separation and points to evaluation disclosures. The actual private dataset cannot be reconstructed from the card |
| [CLEF / CLEF-flash review](../../docs/RESEARCH.md#cloudflare-clef-and-linked-decision-models-reviewed-2026-10-01) | Frozen Qwen backbone, rank-256 LoRA plus a learned routing head; internal synthetic schemas; disclosed smoothed CE/Brier and a later RLCD objective | Cloudflare reports an internal Decision Index 0.2.1 rerun. Data, loss coefficients, reward implementation and full competitor settings are not released; this notebook does not reproduce that recipe |

The common approach is broad decision supervision plus checked hard examples.
Benchmark test sets are not interchangeable with training datasets. In particular,
this notebook uses neither JevBench items nor Plumb's calibration split for training.
TypeSafe datasets and their reference distributions are also excluded from every
fitting and selection stage. The benchmark reader is separate from the training reader.

## Lessons carried forward from this repository

[Experiments 27–28](../README.md#27-m22-matched-decision-lora-pilot) showed that
improving ContractNLI or short-premise SNLI does not guarantee retention on other
tasks. The new recipe measures each source separately and allows the frozen
parent to win selection. It does not repeat contract-only training or treat SNLI
replay as a proven cure for forgetting.

ContractNLI is a useful later long-document retention panel, but its existing
source-group assignments and protected splits must be respected. QASPER stays
out of this trainer because the repository's source/evidence audit exposed label
and representation problems. Old experiment artifacts are never read or changed.

The 2,048-token cap is for the first bounded pilot. Entire overlength records
are rejected, never truncated or relabeled. Passing this pilot would not establish
long-document retention. A subsequent full-document study needs its own admission
report and source-aligned evaluation.

The [working paper](../../docs/whitepaper/WORKING_PAPER.md) carries evidence
through E42. Its eligibility errors motivate exact partial-information pairs:
a known failing conjunct makes the answer `deny` despite a missing fact; a known
successful disjunct can make it `approve`. The paired unresolved cases still
require semantic none. All nine rule siblings share their split group. This
changes the data identity. Current configuration and source hashes prevent
resuming incompatible v1/v2 runs.

## What CLEF justifies changing

The current readout already does one prefill and projects only frozen answer-code
rows. CLEF's additional mechanism is a learned reader over token evidence plus
cross-field mixing. Neither a larger LoRA rank nor fewer forward calls establishes
a better local model. Keep rank 16 and the selected-vocabulary control; compare
an isolated option reader before testing a full-schema head. CLEF's field softmax
does not supply NJ's explicit joint route/urgency distribution.

[CLEF-flash's pinned settings](https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6)
are a useful architecture reference for the intended task:

| Released setting | CLEF-flash | Current 4B pilot |
|---|---|---|
| Backbone | Qwen3.5-9B, text hidden width 4,096, multimodal tower retained | Qwen3.5-4B text model, width 2,560 |
| Hybrid layers | 32: 24 DeltaNet, 8 full attention | Same layer count and split, different dimensions |
| Embeddings / output rows | Untied in the released text config; custom head reads token states | Tied and frozen; selected answer-code rows |
| Head | Width 1,024; 2 routing layers; 4 decoder layers; 16 heads; feedforward width 4,096; dropout 0 in released code | No added head |
| Question interaction | Full schema in backbone, then unmasked field mixing | One isolated question per prefill |
| Disclosed training adapters | Rank 256 with routing head trained jointly | Rank 16; alpha 32; learning rate `2e-5` |
| Disclosed loss settings | Smoothed CE plus Brier; coefficients and optimization settings absent | Smoothing 0.05, Brier weight 0.1, optional controlled sweep |

The [4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
and [flash head code](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/joint_schema_model.py)
establish these dimensions and boundaries. The released flash weights include a
merged backbone and separate head, not a recoverable optimizer or adapter recipe.
Its head weights cannot be attached directly to the 4B model. Test a newly trained
isolated reader first; full-schema mixing and larger rank need separate comparisons.

The default run enables smoothing 0.05 and Brier weight 0.1. For a bounded
comparison, set `RUN_LOSS_SWEEP=True` in the notebook. It runs these six arms
with identical admitted data, seed, precision, rank, learning rate and update budget:

| Arm | `label_smoothing` | `brier_weight` |
|---|---:|---:|
| CE control | 0.0 | 0.0 |
| Smoothing only | 0.05 | 0.0 |
| Brier only | 0.0 | 0.1 |
| Combined, lower smoothing | 0.02 | 0.1 |
| Combined, default | 0.05 | 0.1 |
| Combined, higher Brier | 0.05 | 0.3 |

The nonzero values are pilot settings, not recovered Cloudflare coefficients.
Smoothing changes CE's target; Brier uses the original distribution and sums over
outcomes. Teacher weight scales the complete example loss. Preserve soft teacher
mass and report ordinary unsmoothed-label NLL/Brier.

The sweep prepares data once, loads one fresh seeded model at a time, and uses a
fresh optimizer for each arm. Every step-zero development report must match;
precision/device drift or a mismatched parent stops the sweep. Each arm resumes
its own committed checkpoints. At defaults, six fits total 2,400 updates and
38,400 presentations, plus evaluation. Leave `RUN_LOSS_SWEEP=False` for one fit.

Select one arm by unsmoothed source-macro development NLL subject to existing
per-source retention guards, including Brier. Ties prefer an earlier checkpoint,
then arm order. Calibration, gate and reserved-test results do not select arms.
Only the winner proceeds to calibration and the gate; a failed gate retains its
parent without trying another arm. `SWEEP_PLAN.json` and `SWEEP_RESULT.json`
record all settings, development metrics and the frozen selection, and accompany
the winner's export.

Sweep only loss settings first. Change learning rate or rank in a separate study
if learning curves show stalled optimization or a capacity limit. Sweeping all
three together would obscure the source of a gain and multiply the GPU budget.

Do not compare raw training losses across objectives or promote an arm from a
leaderboard claim. Separate run identities prevent incompatible resume. A repeated
search using gate results spends that gate and needs fresh acceptance groups.

RLCD, ordinal utility rewards, schema/prompt-template augmentation, and a learned
head remain separate experiments. The public recipe cannot establish their exact
settings or individual effects. Exact-record rewards need multi-field training
records; the current single-question mixture does not provide them.

## Training and validation contract

- Start from the pinned post-trained Qwen3.5-4B, not the Base checkpoint. Train
  rank-16 LoRA on full-attention, DeltaNet and MLP projections. Keep embeddings
  and output rows frozen. Read only the answer-code rows from the final hidden
  state, without generation or a full vocabulary projection.
- Use only upstream training files. Split normalized state groups 75/8/7/5/5
  into training, development, calibration, gate and reserved test before augmentation.
  Exact cross-source state duplicates share a role; rule counterfactuals share a
  group. Exact duplicate requests are removed and conflicting labels stop preparation.
  This is not comprehensive semantic near-duplicate decontamination.
- Randomize option/code positions. Omit the correct option in approximately 20%
  of eligible one-hot Choice cases and supervise `__none__`. Preserve natural
  MNLI neutral labels separately from this intervention. MultiRC is binary
  candidate verification because several candidates can be correct.
- Select on source-macro development NLL with per-source accuracy/NLL/Brier and
  sufficiently populated class-recall constraints. Fit temperature on calibration
  only. A separate gate accepts or rejects calibration and the candidate without
  selecting another checkpoint. The small default panels support screening,
  not tight error-rate guarantees.
- Record paired schema diagnostics for the selected candidate and parent on gate
  groups: reverse option/code positions and rename opaque Choice keys while keeping
  descriptions and semantic none. Report canonical probability vectors, total
  variation, winner changes, counts and overlength skips. They are descriptive,
  do not change selection, and do not establish native parity or prompt invariance.
- Optionally evaluate the frozen export on reserved groups, PAWS, SciQ and an
  unseen rule composition. Public corpora may be in Qwen pretraining; the split
  protects this fine-tune, not an unknown pretraining history.

All reports include sample counts. The default manifest prepares reserved labels
for serialization but computes no final model scores. Historical OpenKind final
splits remain unopened. Turning on `RUN_FINAL_TEST` runs the new final panel only.

## TypeSafe benchmark, evaluation only

Set `RUN_TYPESAFE_BENCHMARK=True` after the export is frozen. The separate
[benchmark.py](benchmark.py) verifies export hashes, evaluates that exact checkpoint
at its accepted temperature, compares the zero-update parent at temperature 1,
and restores the export. It never fits or promotes a model. Training preparation,
training/development selection, calibration and the gate reject benchmark rows;
the training download helper rejects repositories owned by `typesafe`.

The optional notebook panel declares `TYPESAFE_MAX_LENGTH=12288`, separately from
the 2,048-token training/export cap. Set it to `None` to enforce the export cap.
The report records both limits and flags extended-context evaluation as unqualified
for deployment. It keeps the same frozen model, readout and temperature; it does
not change the exported inference contract. Longer inputs need their own quality
and memory evidence, and an OOM stops the benchmark.

The five immutable snapshots contain 20,605 question instances:

| Source | Questions | Pinned revision |
|---|---:|---|
| [O*NET](https://huggingface.co/datasets/typesafe/evalsafe-onet/tree/bda14bdd85be4d93140a842332359f526543b314) | 7,500 | `bda14bdd85be4d93140a842332359f526543b314` |
| [Invoice processing](https://huggingface.co/datasets/typesafe/evalsafe-invoice-processing/tree/6beeb2d2acd65c086c835022f5f4d7434114cafc) | 6,874 | `6beeb2d2acd65c086c835022f5f4d7434114cafc` |
| [Customer service](https://huggingface.co/datasets/typesafe/evalsafe-customer-service/tree/b1342f5a704587dbc465867c38c2694348ff86e4) | 3,287 | `b1342f5a704587dbc465867c38c2694348ff86e4` |
| [Security incidents](https://huggingface.co/datasets/typesafe/evalsafe-security-incidents/tree/fbe1ea5c69cf494157fd23f2002a0d9d9a418443) | 1,820 | `fbe1ea5c69cf494157fd23f2002a0d9d9a418443` |
| [Agent traces](https://huggingface.co/datasets/typesafe/evalsafe-agent-trace-observability/tree/8635540973910a92465fe2bc53e195375aa6e1a8) | 1,124 | `8635540973910a92465fe2bc53e195375aa6e1a8` |

The default predeclares ten complete cases per source, selected by a seeded hash
before token admission. Set `TYPESAFE_CASES_PER_SOURCE=None` for full snapshots.
Reports retain the selected-case denominator: overlength, unsupported and failed
questions count as zero. Exact normalized-state overlap with the prepared mixture
is rejected and counted. No near-duplicate or base-pretraining exclusion is claimed.

O*NET measures `1 - JSD` (base 2) for Noul/Choice and
`1 - abs(predicted_mean - reference_mean) / rubric_range` for Score. Overall
agreement averages type means equally. This is agreement with synthetic Astra/Fable
consensus, not labeled accuracy. The four workflow datasets officially score
complete action sets or primary actions, including arguments. This helper evaluates
their published fixed-input questions only. It does not execute policies or produce
their official workflow scores. Upstream branch inputs are preserved as published;
we do not infer new branches from these diagnostic results.

Choice schemas without `__none__` receive a described none option with zero
reference mass; its predicted mass remains in the metric. This adaptation, sampling
and incomplete coverage prevent direct comparison to CLEF's published scores.
The 16-code cap rejects 145 O*NET Choice rows; three further rows lack option
descriptions. The 2,048-token cap also excludes long states. These are coverage
limits, not evidence of wrong neural answers. Keep the original reference mass,
option descriptions and probability-key association; never truncate or silently
condition away none mass to improve a score.

The [tokenizer-only source audit](TYPESAFE_SOURCE_AUDIT.json) checked all 20,605
schema conversions and admitted the default ten-case samples at 2,048 tokens.
O*NET admitted 334/500, invoices 0/425, customer service 177/177, security
37/63, and agent traces 46/109. Sampled invoice prompts ranged from 4,345 to
9,286 tokens. This audit loaded no neural model and computed no benchmark scores.
At the declared 12,288-token benchmark bound, the same samples admit 496/500,
425/425, 177/177, 63/63 and 109/109 respectively. Four O*NET questions still
exceed the outcome bound. This establishes token/schema admission, not quality,
GPU memory feasibility or full-snapshot coverage.

Pinned files, their hashes, sampled case IDs, prediction vectors, admission failures
and local serial wall time accompany the frozen benchmark report. Wall time includes
Python overhead and is not service or complete-workflow latency. Neither source
data nor reference labels are packaged in the export. O*NET and invoice cards
declare Apache-2.0; the other three cards do not specify a dataset license.
Further tuning after seeing these results spends these references as a final test.

## Outputs and verification

Drive receives the admitted dataset manifest, baseline, training history, committed
checkpoints, selected checkpoint, calibration/gate rows and an export ZIP. The
export includes the adapter, tokenizer, readout contract and hashes. It contains
neither base weights nor source datasets. If selection or the gate rejects training,
the export contains the zero-update adapter and the candidate remains inspectable.

The exported [train.py](train.py) also supplies `decide(model, tokenizer, state,
questions, contract)` for reference inference. The caller supplies the pinned base
with the verified exported adapter and tokenizer. Read `DECISION_CONTRACT.json`
for the prompt identity, code rows, token cap and accepted temperature.

```python
answers = recipe.decide(model, tokenizer, "The application meets every stated requirement.", [
    {"id": "eligibility", "kind": "choice", "instructions": "Does the application qualify?",
     "options": [{"key": "yes", "text": "The application qualifies."},
                 {"key": "__none__", "text": "The evidence does not justify qualification."}]}
], contract)
```

The helper admits at most eight questions before model work, rejects overlength inputs,
requires described Choice none, and evaluates questions serially with their own
options. Question IDs and sibling fields stay out of prompts. Choice/Score confidence
uses normalized entropy; Noul has no confidence. Score returns a finite-level
expectation. These are research distributions, not a wire response or action authority.

The notebook does not merge or quantize the model, register an OpenKind profile,
or claim Mac speed. Those steps require paired native quality and probability
checks after conversion. `Score` supervision here covers finite ordinal scales;
it does not establish arbitrary continuous-score semantics.

The native Rust engine keeps its existing qualified profiles. Experiment 35 needs
a new renderer/profile, merged artifacts and offline probability fixtures before
integration. Its serial reference does not implement hybrid-prefix reuse or CLEF's head.

Readable implementation: [train.py](train.py). Offline checks: [test_train.py](test_train.py).
Regenerate the self-contained notebook with [build_notebook.py](build_notebook.py):

```bash
python research/local_decision_training/build_notebook.py
python -m unittest discover -s research/local_decision_training -p 'test_train.py' -v
```

The tests use tiny randomly initialized models and download no model assets. The
full 4B CUDA training, L4 NF4 path, Google Drive execution and Mac deployment remain
unrun. The authored notebook is an executable experiment, not a completed result.

On 1 October 2026, offline tests passed on macOS arm64 CPU with Torch 2.14.1,
Transformers 5.17.0 and PEFT 0.21.1. A separate tokenizer-only check at the pinned
Qwen revision verified 16 distinct answer-code tokens and append boundaries on
90 generated rows and 170 schema variants. It loaded no backbone and establishes
token/render compatibility only. Sweep checks use tiny hybrid models to verify
fresh matched initialization, resume, development-only selection and rejection
of a mismatched parent. The 4B sweep has not been run.
TypeSafe checks cover source exclusion, reference schemas and soft targets,
published metric math, failed-row denominators, frozen-export verification and
parent comparison without export mutation. Twenty offline tests pass; the
tokenizer-only audit is separate evidence from neural evaluation.
