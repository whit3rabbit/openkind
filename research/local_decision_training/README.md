# Local decision training

Start with [notebook 35](../35_local_decision_training.ipynb). It trains a
`Qwen/Qwen3.5-4B` decision LoRA on an A100, with an NF4 QLoRA path for L4.
The deployment target is a 16–32 GB Mac after merge, quantization and native
qualification. This is a v4 experiment, not a promoted model or a completed
4B training result. The [local design](../../docs/whitepaper/LOCAL_DECISION_DESIGN.md)
separates this pilot from native integration.

For T4-class Colab GPUs, which have no BF16 support,
[local_decision_training_t4.ipynb](local_decision_training_t4.ipynb) embeds the same
reviewed trainer and test suite under a separate `nf4_fp16` run identity with a
1,024-token cap, shorter budgets and predeclared parameter sweeps. It runs end to
end unattended and is a distinct experiment, not a cheaper notebook 35.

[DECISIONS.md](DECISIONS.md) records the evidence, alternatives and acceptance
conditions for this design.

## Model workflow

| Stage | Notebook | Output |
|---|---|---|
| Compare bounded recipes | [T4 experiments](local_decision_training_t4.ipynb), or [A100/L4 control](../35_local_decision_training.ipynb) | Development-selected configuration and retention evidence |
| Fine-tune one fixed recipe | [Fine-tuning](local_decision_finetuning.ipynb) | One LoRA run, calibration, gate and locked research export |
| Optional decision RL | [Post-training](local_decision_posttraining.ipynb) | Warm-started adapter, reward/reference diagnostics and development retention; acceptance disabled by default |
| Prepare a release | [Release preparation](local_decision_release.ipynb) | Pinned PEFT adapter, tokenizer, decision contract, model card and file manifest |

The fixed-recipe notebook accepts a local `SWEEP_RESULT.json` through
`CHOSEN_RECIPE_PATH`, or runs the bounded control. It starts from the pinned Qwen
base with a fresh optimizer; it does not continue another arm's adapter. Exact
reruns resume committed checkpoints. Gate results must not inform further tuning
while those same groups are presented as fresh acceptance evidence.
An experiment that already produced a suitable frozen export can go straight to
release preparation. Repeating the same fit is optional, not a required stage.
Declare any longer budget or changed recipe before consulting acceptance results.

### Dataset fine-tuning and optional decision RL

The fixed-recipe notebook trains on datasets, not just a configuration file:
MNLI, BoolQ, Banking77, MultiRC, SST-5, Plumb teacher decisions and generated
rules use the same pinned conversion, admission and group splits as the T4
experiment. `include_helpsteer2=True` adds the human-rating adequacy proxy.
`data_intervention="reasoning"` replaces part of the rules training allocation
with exact reasoning cases and complete multi-field requests. These are
declared interventions; neither is enabled implicitly in supervised control
fits. PAWS, SciQ and TypeSafe remain evaluation-only. ContractNLI and QASPER
remain later long-document studies, outside the current bounded mixture.

The [decision-RL trainer](decision_rl.py) implements an optional second stage
from the trained supervised adapter. It uses a fresh optimizer and records the
source checkpoint, adapter digest, original dataset binding, stage configuration
and code hashes. It refuses a step-zero warm start. If the supervised selector
retained its parent, an explicitly supplied positive `SUPERVISED_STEP` can start
an experimental recovery study from a rejected checkpoint. This does not certify
that checkpoint or the RL result. Preserve the full run and prepared-data folder;
the export ZIP does not contain rejected checkpoints or training rows.

Configure `SUPERVISED_RUN` and `SUPERVISED_DATA_DIR`, then declare one method:

| `RL_CONFIG["method"]` | Objective |
|---|---|
| `supervised` | Direct CE + Brier + frozen-reference KL; no utility gradient |
| `expected_utility` | Same anchors plus exact finite-action expected utility |
| `reinforce` | Same anchors plus sampled actions and a leave-one-out policy gradient |

The default reward coefficients are experimental starting values, not recovered
Clef settings. Adjacent numeric ordinal levels receive partial credit; answer
code positions never define distance. `__none__` receives exact credit only.
Complete-record credit requires every field in a declared, exact-label request
to be correct. Counterfactual pairs are not treated as joint records. Teacher
probabilities remain soft targets and retain the supervised half loss weight.
The stage enables the reasoning intervention to supply complete training requests.

CE and Brier use their direct gradients. Assigning the same detached Brier score
to every sampled action would cancel under the leave-one-out baseline and would
not train calibration. A positive `kl_weight` constrains the decision distribution
against cached logits from the supervised checkpoint. Only those finite logits
are cached, so a second 4B backbone is not retained on the T4. Code permutations
and admission bounds bind every reference presentation. Record updates retain
multiple field graphs and can cost more memory than single-question updates.

The initial budget is 50 updates with accumulation over eight complete units.
Compare all three methods with matched inputs and budgets on development first.
Selection still minimizes source-macro development NLL subject to the existing
retention guards. Reward and complete-record utility are diagnostics, not
substitutes for proper scores or acceptance evidence. `RUN_ACCEPTANCE=False`
keeps calibration scores, gate scores and exports closed by default.

Fresh acceptance excludes calibration, gate and test groups from the source
data, plus every additional directory listed in `PRIOR_EVIDENCE_DATA_DIRS`.
Split roles are not reassigned. Exclusion is part of the admission cache identity;
insufficient new groups stop preparation. The notebook cannot discover evidence
you viewed in other runs, so supply those directories. After freezing the method,
enable acceptance once. Failure exports the uncalibrated supervised checkpoint,
which is this stage's step zero, and does not qualify a new RL improvement.

W&B records policy rewards, exact-record rewards, CE, Brier, entropy, reference
KL, development utility and timing on the optimizer-step axis. Local checkpoints,
reference locks, per-update diagnostics and `POSTTRAINING_SUMMARY.json` also work
without W&B. An accepted export locks the RL plan, trainer and supervised lineage
and works with the existing release notebook. No public upload is automatic.

This is an open calibration-aware decision optimization experiment inspired by
[Clef's disclosure](https://blog.cloudflare.com/clef-decision-models/), not a
reproduction of its unreleased reward implementation. Offline tiny-model tests
cover policy-gradient agreement with an exact expectation, ordinal remapping,
complete requests, warm-start binding, interruption/resume, reference tampering,
fresh-group admission and frozen export. Full 4B CUDA/T4 quality, memory and speed
remain unmeasured.

The release notebook uses a frozen export directory or its portable ZIP. It
rejects step-zero exports as a new trained model, preserves the locked bundle,
and pins the base revision in the conventional PEFT entry files. A local package
is the default. Optional upload requires an explicit token and a new private Hub
repository, then checks the inventory at the returned commit. It does not merge
the base, publish publicly, or install a registry profile. Review the model card
and source terms before public distribution; SST-5's mirror still declares no
license. [Hub uploads](https://huggingface.co/docs/huggingface_hub/guides/upload)
and [model cards](https://huggingface.co/docs/huggingface_hub/guides/model-cards)
define the external packaging contract.

### What counts as our model

The first deliverable should be a named, versioned **decision adapter** on the
pinned Qwen3.5-4B base. The trained weights, renderer, dynamic-option readout,
semantic-none rule, calibration and qualification evidence form the model.
This is fine-tuning of pretrained weights, not pretraining a new architecture.
Its custom readout needs the supplied reference code; a generic generation
pipeline does not implement the decision contract.

After an adapter passes meaningful held-out comparisons, load the unquantized
pinned base, merge with PEFT, and recheck probabilities, calibration and task
quality. Then quantize and qualify the native Mac path with matched memory and
latency evidence. Merge changes the artifact and quantization changes numerics;
neither inherits the training screen automatically.
[PEFT checkpoint guidance](https://huggingface.co/docs/peft/developer_guides/checkpoint)
describes adapter and merged artifacts.

A later architecture study could compare a smaller pretrained encoder with a
dynamic option-ranking head against this 4B baseline. That head must learn
abstention, preserve option descriptions and expose typed score distributions.
Treat it as a new family with its own loss, renderer, calibration, native runtime
and qualification fixtures. Keep pretraining from scratch in a separate campaign
with a declared corpus and compute budget. The current capped mixture and pilot
update count do not establish a useful general backbone.

### W&B and speed

`USE_WANDB=True` in the T4 or fixed-recipe notebook logs live loss, gradient norm,
learning rate, update timing, token throughput and development metrics. The
optimizer-step axis is separate from W&B's internal log sequence. Online runs use
a stable ID and skip already logged events on resume; offline runs need no login
and produce separate logs in the same group. `WANDB_LOG_EVERY=10` limits update
logging. W&B observes local selection and never runs its own search over gate or
test data. After export it records the actual gate decision, exported step and
calibration, including parent fallback. `WANDB_LOG_REPORTS=True` optionally
attaches small qualification reports; weights and row-level predictions are excluded.

Prepared-data reuse checks the fast tokenizer, chat template, source pins,
configuration, trainer hash, provenance manifest and every role's stored content hash. A completed
sweep verifies each arm's final and selected checkpoints before returning its
saved winner, without reloading model weights or rerunning preflight. Loss totals
stay on device until an update completes. `PERFORMANCE.json` separates training,
development evaluation and checkpoint time for the latest invocation; it is not
a GPU benchmark or a quality result.

The pasted run shows a 150-update loop taking 1:13:28, with a final progress-rate
estimate near 83 seconds per update. That smoothed estimate includes stalls and
does not measure isolated update time. The reference
DeltaNet/convolution warnings identify a plausible bottleneck, but no kernel
timing attributes that wall time yet. Keep the math attention/checkpoint path as
the control. Test optimized kernels, SDPA changes or larger microbatches in
separate paired runs on the target GPU, including real backward, memory,
probability and quality checks. Do not assume an A100 result applies to T4.
FP16-only training also needs a dedicated scaling/numerics qualification; this
audit has not qualified a GradScaler path or established T4 quality gains.

Regenerate and validate locally:

```bash
python research/local_decision_training/build_notebook.py
python research/local_decision_training/build_t4_notebook.py
python research/local_decision_training/build_finetuning_notebook.py
python research/local_decision_training/build_posttraining_notebook.py
python research/local_decision_training/build_release_notebook.py
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 python -m unittest discover -s research/local_decision_training -p 'test*.py' -v
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 python research/local_decision_training/validate_notebooks.py
```

Validation uses offline fixtures. It cannot establish a complete 4B Colab run,
native Mac qualification or a successful Hub upload.

When development selection retains step zero, the gate's `candidate_raw` table
and selected-checkpoint schema panel evaluate the frozen parent. A parent can
also violate an absolute false-none cap; retaining it does not qualify it.
Identical step-zero schema diagnostics at temperature 1 are reused. The summary
separates the lowest observed development NLL from the retention-selected
checkpoint, and candidate evaluation temperature from the actual export temperature.
The export ZIP contains the exported weights, not rejected trained checkpoints.
Preserve the full run directory to keep those checkpoints for further inspection.

### Colab setup and loading diagnostics

The training notebooks remove the unused preinstalled `torchao` before installing
the pinned stack. PEFT 0.21.1 rejects `torchao` 0.10.0 even during ordinary LoRA
dispatch; NF4 in this recipe uses bitsandbytes. Keep Colab's Torch/CUDA build.
If upgrading an already imported stack, restart the Python session and rerun setup.
[PEFT's compatibility check](https://github.com/huggingface/peft/blob/v0.21.1/src/peft/import_utils.py)
defines this dependency constraint.

Transformers 5.17.0 returns sets in `output_loading_info`. The loader normalizes
them to deterministic JSON arrays before writing `ENVIRONMENT.json`, including
empty sets. Missing or mismatched weights and unreviewed text keys still fail the
loading checks. Regression tests load an actual local tiny checkpoint; they do
not download the 4B weights.

## Quickstart

Upload the notebook at [Google Colab](https://colab.research.google.com/), select
an A100 GPU, and run all cells. Mount Drive when prompted. No prior experiment
folder, teacher API key or repository clone is needed. Training resumes from
the last verified optimizer checkpoint when rerun with the same configuration.

**T4, unattended.** Upload [local_decision_training_t4.ipynb](local_decision_training_t4.ipynb),
select a T4 GPU, and run all cells without further interaction. Nothing prompts:
outputs stay on the Colab disk and are lost when the runtime dies (set
`USE_DRIVE=True` to persist on Drive, which requires interactive authorization).
The default run performs a three-arm learning-rate sweep, selects the development
winner automatically, calibrates, gates and exports. Interrupted arms resume from
committed checkpoints when rerun with unchanged settings.

## T4 fp16 pilot differences

| Setting | Notebook 35 (A100/L4) | T4 notebook |
|---|---|---|
| Precision identity | `bf16` or `nf4` (BF16 compute) | `nf4_fp16` (FP16 compute), resolved by `resolve_precision` |
| Token admission cap | 2,048 total prompt tokens | 1,024 total prompt tokens |
| Default updates | 400 × effective batch 16 | 150 × effective batch 16 |
| Default study | single plain-CE control | three learning rates, then automatic selection |
| Storage | Google Drive | local Colab disk; Drive optional and interactive |
| Minimum GPU memory | 35 GiB BF16 / 20 GiB NF4 | 14 GiB |

The 1,024-token cap rejects overlength records whole, never truncates them, and
the admission audit quantifies the difference. Changing precision, token cap,
update budget, seeds, data settings or the embedded source creates a different run
identity, so a T4 run neither resumes nor numerically compares against a notebook 35
run. `SWEEP_MODE` selects one predeclared study per run identity: `"learning_rate"`
or `"rank"` use the bounded single-dimension parameter sweep, `"loss"` runs the
existing six-arm loss sweep (which can exceed one T4 session), and `"none"` runs a
single control fit. `CUSTOM_SWEEP_ARMS` may replace the default arms with 2–6
alternatives that include the control value and vary nothing else. FP16 activations
and gradients on T4 are a hypothesis this notebook exists to measure; its gradient
preflight still requires finite nonzero adapter gradients before any training, and
a completed run is not T4 deployment qualification or Mac evidence.

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
The T4 notebook's `SWEEP_MODE` implements those separate studies directly:
`learning_rate` and `rank` run the bounded single-dimension parameter sweep
(`PARAMETER_SWEEP_ARMS`, control value required, `alpha` pinned to twice `rank`)
with the loss sweep's matched seeds, fresh optimizers, step-zero parity checks,
protected-role isolation and development-NLL selection. Rank arms rebuild the
adapters; each arm's identity binds its shapes. Only the selected arm reaches
calibration and the gate.

Do not compare raw training losses across objectives or promote an arm from a
leaderboard claim. Separate run identities prevent incompatible resume. A repeated
search using gate results spends that gate and needs fresh acceptance groups.

Decision RL and ordinal/complete-record utility rewards now have their own
optional post-training notebook. Meaning-preserving prompt paraphrases and a
learned head remain separate experiments. The v4 presentation transforms have
explicit switches and do not generate paraphrases. The public recipe cannot establish their exact
settings or individual effects. Exact-record rewards use the reasoning generator's
declared multi-field requests; the single-question control does not provide them.

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
Regenerate the self-contained notebooks with [build_notebook.py](build_notebook.py)
for the A100/L4 run and [build_t4_notebook.py](build_t4_notebook.py) for the
unattended T4 run:

```bash
python3.12 -m venv /tmp/openkind-local-decision-tests
/tmp/openkind-local-decision-tests/bin/python -m pip install \
  torch==2.14.1 transformers==5.17.0 peft==0.21.1 nbformat==5.11.1 \
  'tqdm>=4.66' 'pandas>=2.2'
/tmp/openkind-local-decision-tests/bin/python research/local_decision_training/build_notebook.py
/tmp/openkind-local-decision-tests/bin/python research/local_decision_training/build_t4_notebook.py
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/openkind-local-decision-tests/bin/python -m unittest discover \
  -s research/local_decision_training -p 'test*.py' -v
```

The tests use tiny randomly initialized models and download no model assets. The
full 4B CUDA training, L4 NF4 path, T4 NF4 FP16 path, Google Drive execution and
Mac deployment remain unrun. The authored notebooks are executable experiments,
not completed results.

On 3 October 2026, all 58 v4 offline tests passed in an isolated Python 3.12.11
environment on macOS arm64 CPU with Torch 2.14.1, Transformers 5.17.0 and PEFT
0.21.1. These cover atomic admission, exact generators, occurrence augmentation
and interrupted resume, grouped reports, confidence semantics, tamper rejection
and frozen reserved-input re-encoding. Sweep checks use tiny hybrid models to
verify matched initialization, development-only selection and rejection of a
mismatched parent, now including the new learning-rate and rank parameter sweeps
and the `resolve_precision` capability floors. The 4B sweep has not been run.

The regenerated notebooks each validate as 28 cells with 13 compilable code cells.
Their six embedded Python sources match the reviewed files byte for byte; all 58
tests also pass after extracting those sources from either notebook into an
isolated directory. Regeneration is deterministic. The full Colab workflows were
not executed.

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
