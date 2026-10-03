# Local decision training

Start with [notebook 35](../35_local_decision_training.ipynb). It trains a
`Qwen/Qwen3.5-4B` decision LoRA on an A100, with an NF4 QLoRA path for L4.
The deployment target is a 16–32 GB Mac after merge, quantization and native
qualification. This is a v4 experiment, not a promoted model or a completed
4B training result. The [local design](../../docs/whitepaper/LOCAL_DECISION_DESIGN.md)
separates this pilot from native integration.

[DECISIONS.md](DECISIONS.md) records the evidence, alternatives and acceptance
conditions for this design.

## Quickstart

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
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/990b2711a36180dd19d9c94b8627844866f8982a) | 0 by default; 1,000 if enabled | Optional adequacy proxy from human helpfulness/correctness ratings of model-written responses |

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
| [Strands Decider / Hobson v19 review](../../docs/RESEARCH.md#strands-decider-2b-hobson-v19-release-and-agent-interventions-reviewed-2026-10-01) | Qwen3.5-2B-Base, rank-16 LoRA and a learned option pointer; label CE with frozen-base and parent-replay KL; public tasks, multi-step reasoning, generated documents and adequacy | Released scripts, corpus hashes and held-out panels expose both gains and failed retention experiments. Historical base revision is inferred; code revision is redacted. This is an external comparator, not a reproduced 4B result |

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

The [working paper](../../docs/whitepaper/WORKING_PAPER.md) and
[whitepaper §24](../../docs/whitepaper/WHITEPAPER.md#24-openkind-t4-recovery--classifier-verification-input-reuse-nli-specialization-and-general-policy-limits)
carry evidence through E43/E44. A targeted eligibility correction also damaged
retry decisions, so field gains must be checked against complete requests.
Numeric capping, chronological order and majority-class collapse are explicit
diagnostics. The recovered encoder comparisons are complete within their bounded
recipe; they establish neither a general encoder replacement nor Mac readiness.

Split groups prevent leakage. Atomic units preserve declared training pairs and
triplets. They serve different purposes: related rule rows share a role, but only
declared units must survive deduplication, token admission and row caps together.
An incomplete unit is rejected; a complete unit that exceeds remaining capacity
is skipped. The admission audit records unused capacity. Ordinary large documents
need not fit a cap as one unit. Configuration, data and implementation hashes
version v4 runs separately from earlier artifacts.

## Controlled experiment sequence

The default is one 400-update, 2,048-token plain-CE run with the existing broad
mixture: `data_intervention="control"`, `presentation_augmentation="none"`,
`label_smoothing=0.0`, and `brier_weight=0.0`.

1. Establish the v4 control with corrected atomic sampling and diagnostics.
2. For a data comparison, set `data_intervention="reasoning"`. Replace half the
   rules allocation with exact numeric-capping, chronology, decisive-known-fact,
   and answer-changing-instruction examples. Preserve other-source exposure,
   total presentation budget, and evaluation manifests. Independently identified
   facts and held-out renderings protect historical evaluation material.
3. For a presentation comparison, hold semantic data and loss fixed. Set
   `presentation_augmentation` to `option_order`, `code_assignment`, `opaque_keys`,
   or `all`. Prefer individual transforms first. Each occurrence has a deterministic
   presentation, with canonical targets remapped exactly and replayed on resume.
4. Only then run the optional six-arm loss sweep on one frozen data/presentation
   configuration. Development selects one candidate; calibration and the gate
   never choose another arm after failure.

`split_seed`, `initialization_seed`, `sampling_seed`, and `augmentation_seed`
are separate run inputs. Each defaults to the legacy `seed=17` when unset.
Keep all four fixed across matched arms. Training schedules record source
exposure; changing a seed, intervention, precision, update budget or implementation
starts another run. A favorable gate result does not authorize combining treatments.

HelpSteer2 remains a separate opt-in comparison. A 4,096-token study, automatic
teacher generation, KL replay, larger adapters, and new heads need separate
experiments. The control does not infer those benefits from external recipes.

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
| Disclosed loss settings | Smoothed CE plus Brier; coefficients and optimization settings absent | Plain CE control; optional six-arm sweep |

The [4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
and [flash head code](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/joint_schema_model.py)
establish these dimensions and boundaries. The released flash weights include a
merged backbone and separate head, not a recoverable optimizer or adapter recipe.
Its head weights cannot be attached directly to the 4B model. Test a newly trained
isolated reader first; full-schema mixing and larger rank need separate comparisons.

The default run uses unsmoothed CE with zero Brier weight. For a bounded
comparison, set `RUN_LOSS_SWEEP=True` in the notebook. It runs these six arms
with identical admitted data, four seed values, precision, rank, learning rate and update budget:

| Arm | `label_smoothing` | `brier_weight` |
|---|---:|---:|
| CE control | 0.0 | 0.0 |
| Smoothing only | 0.05 | 0.0 |
| Brier only | 0.0 | 0.1 |
| Combined, lower smoothing | 0.02 | 0.1 |
| Combined, middle smoothing | 0.05 | 0.1 |
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

RLCD, ordinal utility rewards, meaning-preserving prompt paraphrases, and a
learned head remain separate experiments. The v4 presentation transforms have
explicit switches and do not generate paraphrases. The public recipe cannot establish their exact
settings or individual effects. Exact-record rewards need multi-field training
records; the current single-question mixture does not provide them.

## Strands Decider training lessons

The [source review](../../docs/RESEARCH.md#training-transfers-for-the-local-4b-pilot)
checks the released v19 configuration, data builders, retention objectives and
later experiments. Its strongest contribution is an inspectable recipe with
skill-specific retention rules. Its 2B Base pointer head differs from our
post-trained 4B vocabulary readout, so its head weights, learning rates and fitted
temperatures do not transfer directly.

For a separate data ablation, set `include_helpsteer2=True` in the notebook's
`CONFIG.update(...)`. This enables only the pinned upstream training file:

- Adequate: helpfulness and correctness both >=3. Inadequate: either <=1.
  Drop the middle band, empty responses and marked multi-turn prompts. Human
  ratings become a binary proxy; they are not exact correctness labels.
- Put every response to the same normalized request in one split group before
  sampling. Ratings stay out of prompts. Admit up to 500 adequate and 500
  inadequate training rows, preferring requests with alternative inadequate
  replies. Evaluation keeps its sampled natural label prevalence.
- Keep the baseline loss, seed, optimizer and 400-update budget fixed. The
  added source raises the mixture cap from 8,000 to 9,000 rows and changes
  baseline-source exposure. Compare those sources on development data as well
  as adequacy; a gain on adequacy alone cannot justify promotion.

Leave the flag off for the existing loss sweep. Source/configuration hashes
prevent this ablation from resuming a baseline run. NVIDIA specifies CC-BY-4.0
for HelpSteer2. This flag imports no Strands synthetic, replay or evaluation file.

The [source audit](STRANDS_SOURCE_AUDIT.json) parsed all 20,324 upstream rows;
12,713 meet the proxy rule, with 10,280 adequate and 2,433 inadequate. Under
the default seed and token cap, an isolated HelpSteer2/rules admission check
admits 1,000 balanced training rows and 64 rows per evaluation role, with
disjoint request groups. The
development sample has only six inadequate examples, so minority-class
estimates are coarse. This is token/data admission, not model-quality evidence.

The same audit checks hashes and exact state overlap for committed generated
v18, adequacy and v20 instruction-flip files. Of 100 sampled training rows per
file, 94 document rows and all adequacy/flip rows fit our 2,048-token cap after
schema conversion. Labels and semantic independence remain unverified. The
full Strands mixture includes PAWS training rows, conflicting with our reserved
PAWS transfer panel; do not import it wholesale.

Test question sensitivity separately: preserve state/options while changing
the question so the correct answer changes, and score both answers together.
Keep meaning-preserving paraphrases as a different diagnostic. Their v20
combined treatment improves generated flip-pair accuracy but fails retention
guards; it is not the released v19 default. False-none errors and unfamiliar
question templates must remain visible alongside the new skill's score.

Base/parent KL, a pointer readout, adjacent-level Score smoothing and per-kind
temperature fitting remain named follow-ups. Strands' broader temperature
refit improves ECE while worsening NLL and is rejected by its own rule.
Retain our proper-score and source-retention gates before adopting any of them.

## Training and validation contract

- Start from the pinned post-trained Qwen3.5-4B, not the Base checkpoint. Train
  rank-16 LoRA on full-attention, DeltaNet and MLP projections. Keep embeddings
  and output rows frozen. Read only the answer-code rows from the final hidden
  state, without generation or a full vocabulary projection.
- Use only upstream training files. Split normalized state groups 75/8/7/5/5
  into training, development, calibration, gate and reserved test before augmentation.
  Exact cross-source state duplicates share a role; rule counterfactuals share a
  group and declared pairs/triplets are sampled atomically. HelpSteer2 response
  siblings share their request group. Exact duplicate
  requests are removed and conflicting labels stop preparation.
  This is not comprehensive semantic near-duplicate decontamination.
- Fix one randomized presentation per admitted row in the control. Optional
  occurrence augmentation changes only the named presentation dimensions. Omit the correct option in approximately 20%
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

Reports include family/kind slices, label and predicted-class distributions,
majority controls, paired correctness and complete-request correctness where
fixtures declare complete multi-question requests. Independent group counts and
paired group-bootstrap intervals accompany comparisons. Sparse slices remain
pilot screening evidence, even when a numerical guard passes.

Maximum-probability and normalized-entropy confidence have separate risk/coverage
reports. Entropy confidence matches the exported Choice/Score helper; Noul has
no confidence field. Neither curve defines an application authorization policy.

All reports include sample counts. The default manifest prepares reserved labels
for serialization but computes no final model scores. Historical OpenKind final
splits remain unopened. Turning on `RUN_FINAL_TEST` runs the new final panel only.

## TypeSafe benchmark, evaluation only

Set `RUN_TYPESAFE_BENCHMARK=True` after the export is frozen. The separate
[benchmark.py](benchmark.py) uses the shared frozen-bundle verifier and loader,
evaluates that exact checkpoint
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
export includes the adapter, tokenizer, readout contract, executed implementation
and hashes. Final evaluation and benchmarking verify and load that implementation
and its tokenizer, adapter and accepted calibration before computing scores.
It contains prediction/target records for calibration and the gate, but no original
source text or base weights. TypeSafe data is never included.

The ZIP preserves `export/` beside `EXPORT_LOCK.json`. Keep both when extracting it:
the lock binds the bundle manifest, including its parent adapter, to the frozen run.
These colocated hashes detect changes; they do not authenticate an untrusted bundle.
When importing Python files directly from an extracted bundle, start Python with
`PYTHONDONTWRITEBYTECODE=1`; added `__pycache__` files violate its frozen inventory.
The loader checks the base configuration and adapter configuration. The caller must
still supply the pinned parent weights; this adapter bundle does not independently
hash every caller-loaded base tensor.

A small synthetic replay pack records
rendered inputs, token/code mappings, logits, temperature, probability vectors and
typed answers for later native qualification. These vectors alone establish no
native parity or model quality. If selection or the gate rejects training,
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

Readable implementation: [train.py](train.py). Offline checks include
[test_train.py](test_train.py) and the v4 data, evaluation and bundle suites.
Regenerate the self-contained notebook with [build_notebook.py](build_notebook.py):

```bash
python3.12 -m venv /tmp/openkind-local-decision-tests
/tmp/openkind-local-decision-tests/bin/python -m pip install \
  torch==2.14.1 transformers==5.17.0 peft==0.21.1 nbformat==5.11.1 \
  'tqdm>=4.66' 'pandas>=2.2'
/tmp/openkind-local-decision-tests/bin/python research/local_decision_training/build_notebook.py
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/openkind-local-decision-tests/bin/python -m unittest discover \
  -s research/local_decision_training -p 'test*.py' -v
```

The tests use tiny randomly initialized models and download no model assets. The
full 4B CUDA training, L4 NF4 path, Google Drive execution and Mac deployment remain
unrun. The authored notebook is an executable experiment, not a completed result.

On 3 October 2026, all 55 v4 offline tests passed in an isolated Python 3.12.11
environment on macOS arm64 CPU with Torch 2.14.1, Transformers 5.17.0 and PEFT
0.21.1. These cover atomic admission, exact generators, occurrence augmentation
and interrupted resume, grouped reports, confidence semantics, tamper rejection
and frozen reserved-input re-encoding. Sweep checks use tiny hybrid models to
verify matched initialization, development-only selection and rejection of a
mismatched parent. The 4B sweep has not been run.

The regenerated notebook validates as 28 cells with 13 compilable code cells.
Its six embedded Python sources match the reviewed files byte for byte; all
55 tests also pass after extracting those sources into an isolated directory.
Regeneration is deterministic. The full Colab workflow was not executed.

TypeSafe checks cover source exclusion, reference schemas and soft targets,
published metric math, failed-row denominators, frozen-export verification and
parent comparison without export mutation. HelpSteer2 checks cover proxy thresholds,
request-group isolation and training-label balance.

A separate v4 check used the cached tokenizer at the pinned Qwen revision, with
external corpora disabled. Both data arms admitted 1,000 generated training rows;
all four evaluation manifests matched exactly. Generated prompts used at most
240 tokens. Twenty training rows also passed token-boundary and length checks
under each of the four presentation modes. This is generated-data token/render
compatibility evidence, not a full-corpus admission audit or neural evaluation.
