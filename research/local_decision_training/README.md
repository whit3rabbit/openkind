# Local decision training

Start with [notebook 35](../35_local_decision_training.ipynb). It trains a
`Qwen/Qwen3.5-4B` decision LoRA on an A100, with an NF4 QLoRA path for L4.
The deployment target is a 16–32 GB Mac after merge, quantization and native
qualification. This is a new experiment, not a promoted model or a completed
4B training result.

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

The common approach is broad decision supervision plus checked hard examples.
Benchmark test sets are not interchangeable with training datasets. In particular,
this notebook uses neither JevBench items nor Plumb's calibration split for training.

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
- Select on source-macro development NLL with per-source accuracy/NLL and
  sufficiently populated class-recall constraints. Fit temperature on calibration
  only. A separate gate accepts or rejects calibration and the candidate without
  selecting another checkpoint. The small default panels support screening,
  not tight error-rate guarantees.
- Optionally evaluate the frozen export on reserved groups, PAWS, SciQ and an
  unseen rule composition. Public corpora may be in Qwen pretraining; the split
  protects this fine-tune, not an unknown pretraining history.

All reports include sample counts. The default manifest prepares reserved labels
for serialization but computes no final model scores. Historical OpenKind final
splits remain unopened. Turning on `RUN_FINAL_TEST` runs the new final panel only.

## Outputs and verification

Drive receives the admitted dataset manifest, baseline, training history, committed
checkpoints, selected checkpoint, calibration/gate rows and an export ZIP. The
export includes the adapter, tokenizer, readout contract and hashes. It contains
neither base weights nor source datasets. If selection or the gate rejects training,
the export contains the zero-update adapter and the candidate remains inspectable.

The notebook does not merge or quantize the model, register an OpenKind profile,
or claim Mac speed. Those steps require paired native quality and probability
checks after conversion. `Score` supervision here covers finite ordinal scales;
it does not establish arbitrary continuous-score semantics.

Readable implementation: [train.py](train.py). Offline checks: [test_train.py](test_train.py).
Regenerate the self-contained notebook with [build_notebook.py](build_notebook.py):

```bash
python research/local_decision_training/build_notebook.py
python -m unittest discover -s research/local_decision_training -p 'test_train.py' -v
```

The tests use tiny randomly initialized models and download no model assets. The
full 4B CUDA training, L4 NF4 path, Google Drive execution and Mac deployment remain
unrun. The authored notebook is an executable experiment, not a completed result.
