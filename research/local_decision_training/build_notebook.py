"""Regenerate the self-contained Colab notebook from reviewed source files."""
from pathlib import Path
import textwrap

import nbformat as nbf

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "35_local_decision_training.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text, hidden=False):
    cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    if hidden:
        cell.metadata["cellView"] = "form"
    cells.append(cell)


md("""
# Train a compact local decision model

**OpenKind experiment 35, version 4. Reviewed 3 October 2026.**

Train `Qwen/Qwen3.5-4B` with rank-16 LoRA for typed decisions. The model sees the state,
question, and all options together. Training applies cross-entropy to selected answer-code
logits from one forward pass with ordinary unsmoothed CE. New reasoning data, presentation
augmentation and the existing six-arm loss sweep are separate opt-in comparisons. It does not generate reasoning or answer text.

**Recommended runtime: Colab A100.** An L4 uses NF4 QLoRA automatically. Both require native
BF16; T4-class GPUs have no BF16 support and use the dedicated unattended
[local_decision_training_t4.ipynb](local_decision_training/local_decision_training_t4.ipynb)
instead, whose `nf4_fp16` run identity and 1,024-token pilot are separate experiments. Deployment
remains targeted at a 16–32 GB Mac, after adapter merge, quantization and native-runtime
evaluation.

Upload this `.ipynb` to [Colab](https://colab.research.google.com/), select a GPU, and choose
**Runtime > Run all**. Mount Drive when prompted. No teacher API key or previous OpenKind
experiment folder is needed. The first run downloads public training data and the pinned
4B checkpoint. Allow room for the base weights, dataset cache, and multiple adapter/optimizer
checkpoints; Drive checkpoints can occupy several GB.

This is a new, bounded training recipe. Local offline tests exercise data semantics, actual
tiny Qwen hybrid gradients, checkpoint resume, selection, calibration and export. **A full
4B CUDA training run and Mac quality/speed measurements have not been performed.** Successful
training is not evidence that the adapted model beats its parent.
""")

md("""
## 1. Data and experiment contract

| Source | Purpose | Maximum training rows before runtime admission |
|---|---|---:|
| MultiNLI | Support, contradiction, genuinely insufficient evidence | 1,000 |
| BoolQ | Natural-language yes/no decisions using a supplied passage | 1,000 |
| Banking77 | Dynamic option sets, 2/4/8 candidate intents plus `__none__` | 1,000 |
| MultiRC | Reading across sentences; each answer assessed independently | 1,000 |
| SST-5 | Five-level ordinal sentiment; optional source | 1,000 |
| Plumb decisions | Precomputed Qwen-teacher policy, numeric, rubric and reasoning cases | 2,000 |
| Generated rules | Exact labels, threshold boundaries, missing facts, explicit score rubrics | 1,000 |
| HelpSteer2 | Optional answer-adequacy proxy from human ratings; disabled by default | 1,000 if enabled |

All public training records come from upstream **train** files. Split by normalized state
(request groups for HelpSteer2 responses) before augmentation: 75% training, 8% development, 7% calibration, 5% acceptance gate,
5% reserved test. Split groups prevent leakage; declared atomic pairs/triplets survive
deduplication, admission and cap sampling together. Ordinary document rows remain individually
sampleable within their role. Incomplete units are rejected and unused capacity is reported.
Actual admitted counts
are recorded; the table gives caps, not guaranteed counts or a balanced-label promise.

The first run uses 400 updates × 16 microbatches = 6,400 presentations. The deterministic
training schedule records source exposure and preserves non-target exposure in the reasoning
data comparison. Teacher rows receive loss weight 0.5.
They remain teacher labels, not human gold. Keep explanations out of inputs. Preserve soft
targets, renormalizing only four-decimal rounding errors within 0.001 total mass.

For eligible one-hot Choice records, deterministically remove the correct option in about
20% of cases and supervise `__none__`. Shuffle option/code assignments. Rule groups also
include a missing conjunct with a known failure and a missing disjunct with a known success.
Missing facts alone do not imply none. Never infer semantic
none from confidence, nor label a truncated passage unanswerable. Reject inputs exceeding
2,048 **total prompt tokens**, including question, options and template.

E43/E44 diagnose numeric-capping and chronology errors, collateral regressions after a targeted
eligibility repair, and majority-class collapse. `data_intervention="control"` preserves the
broad mixture. The optional `reasoning` arm replaces half the rule allocation with exact
capping, chronology, decisive-known-fact and answer-changing-instruction examples. Facts have
new identities and evaluation uses held-out renderings. Keep evaluation manifests, total
presentations and other-source exposure fixed across this comparison. Historical evaluation
cases remain protected. These are hypotheses, not measured improvements.

The separate optional final cell evaluates reserved groups, PAWS, SciQ and an unseen rule
composition. Source labels are processed when preparing reserved files; their model results
are not consulted for selection. No historical OpenKind final corpus or QASPER labels are
read. Public datasets can occur in base-model pretraining; these are fine-tuning holdouts,
not a claim of pretraining-clean evaluation.

TypeSafe datasets are benchmark-only. They never enter training, development, temperature
calibration, the acceptance gate or loss-sweep selection. A separate optional cell opens
pinned test snapshots only after freezing the export, with no refitting or promotion.

`include_helpsteer2=False` preserves the baseline mixture for the loss sweep. In a separate
data ablation, set it to `True` to judge request/response adequacy from the pinned NVIDIA
train file. Adequate means helpfulness and correctness >=3; inadequate means either <=1;
drop the middle band and multi-turn prompts. These are human-rating proxies, not exact
logical labels. Group all responses to the same request before splitting. Balance the
admitted training classes, prefer alternative responses to requests with an inadequate
reply, and keep evaluation at its sampled natural prevalence. Ratings never enter prompts.
The extra source adds up to 1,000 rows; keep the 400-update budget fixed and compare the
same baseline sources on development data before accepting a gain. Strands' generated
files, replay targets, PAWS training rows and v20 changes are not imported by this flag.

Dataset terms remain source-specific. The source manifest records pins, file hashes and
license metadata. SST-5's mirror does not specify a license, and MultiNLI/MultiRC have
source-specific terms. Turning off `include_sst5` is supported. This notebook does not
relicense or redistribute the source datasets.
""")

md("""
## 2. Install the training stack

Keep Colab's existing Torch/CUDA installation. Remove Colab's unused `torchao`, whose older
preinstalled version can make PEFT's ordinary LoRA dispatch fail, then install the pinned
Transformers/PEFT/data stack before importing it. NF4 uses bitsandbytes here.
If Colab asks for a restart, restart the session and run all again.
Optional custom DeltaNet kernels are deliberately absent from this initial recipe: the
Transformers reference path is slower but avoids an unqualified backward kernel change.
""")
code("""
import importlib.metadata
import subprocess
import sys

# PEFT probes installed torchao even for ordinary LoRA; this recipe uses bitsandbytes instead.
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
requirements = [
    "transformers==5.17.0", "peft==0.21.1", "datasets==5.0.1",
    "accelerate==1.15.0", "bitsandbytes==0.50.2", "huggingface_hub==1.33.0",
    "safetensors==0.8.0", "tokenizers==0.23.2", "tqdm>=4.66", "pandas>=2.2",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *requirements])
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > A100 or L4 GPU."
assert torch.cuda.is_bf16_supported(including_emulation=False), "Use A100/L4. T4's FP16 route is not qualified here."
print("Torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))
""")

md("""
## 3. Mount Drive and unpack the complete source

The next cell contains the complete trainer and offline tests. It writes only this experiment's
working directory. Nothing executable is fetched from a moving Git branch. Expand the cell
to inspect it; the same readable files are in `research/local_decision_training/` in the repo.
""")
code("""
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
OUTPUT_ROOT = Path("/content/drive/MyDrive/OpenKind_Local_Decision_Training")
WORK = Path("/content/openkind_local_decision_training")
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
""")
embedded_names = ("train.py", "benchmark.py", "wandb_tracking.py", "release.py", "decision_rl.py", *sorted(p.name for p in HERE.glob("test*.py")))
code("# @title Embedded training implementation and offline tests\n"
     + f"EMBEDDED_SOURCE_FILES = {embedded_names!r}\n" + "\n".join(
    f"(WORK / {name!r}).write_text({(HERE/name).read_text()!r}, encoding='utf-8')"
    for name in embedded_names), hidden=True)

md("""
## 4. Parameters

Leave these defaults for the first pilot. An A100 uses BF16 weights; an L4 uses 4-bit NF4
weights with BF16 compute. `max_length`, precision, data settings and source code are part of
the run identity. Changing them creates a new run rather than resuming incompatible work.

`RUN_FINAL_TEST=False` leaves the final evaluation cell inactive. Turn it on only once the
recipe and selection are frozen. Further tuning after viewing that result needs a new test.

The main run uses `label_smoothing=0.0, brier_weight=0.0`. Establish that control first.
Then compare `data_intervention="reasoning"` with semantic data changes only. A separate
presentation comparison can set `presentation_augmentation` to `option_order`, `code_assignment`,
`opaque_keys`, or `all`; try individual transforms before combining them. Occurrences derive
reproducible presentations from `augmentation_seed`, including after checkpoint resume.
Split, initialization, sampling and augmentation seeds are independent run inputs.

After fixing data and presentation, set `RUN_LOSS_SWEEP=True` to compare the six predeclared
settings in `recipe.LOSS_SWEEP_ARMS`. Nonzero coefficients are pilot choices, not recovered
Cloudflare settings. Fix all seeds, admitted data, rank, learning rate, readout, precision and
update budget across arms. The sweep loads one model at a time and resumes each arm's
committed checkpoints. At defaults, it costs 2,400 updates and 38,400 presentations, excluding
evaluation. Leave the switch false for one plain-CE fit. Do not automatically combine
successful-looking interventions or select again after seeing gate/final results.

Smooth the CE target;
compute Brier against the original soft/hard distribution, summed across outcomes. Smoothing
changes the optimum away from the original target, so evaluate NLL/Brier on unsmoothed labels.
Do not add RLCD without a disclosed reward, estimator and matched CE comparison.
""")
code("""
import importlib
import json
import random
import os
sys.path.insert(0, str(WORK))
import train as recipe
importlib.reload(recipe)

CONFIG = dict(recipe.DEFAULT_CONFIG)
CONFIG.update(
    max_length=2048,
    max_steps=400,
    accumulation=16,
    learning_rate=2e-5,
    train_per_source=1000,
    teacher_train_cap=2000,
    eval_per_source=64,
    precision="auto",
    include_sst5=True,
    include_teacher=True,
    include_helpsteer2=False,
    split_seed=17,
    initialization_seed=17,
    sampling_seed=17,
    augmentation_seed=17,
    data_intervention="control",  # "reasoning" replaces half the capped rules allocation.
    presentation_augmentation="none",  # "option_order", "code_assignment", "opaque_keys", or "all".
    label_smoothing=0.0,
    brier_weight=0.0,
)
RUN_LOSS_SWEEP = False  # True runs all six matched loss arms instead of one fit.
RUN_FINAL_TEST = False
RUN_TYPESAFE_BENCHMARK = False  # Evaluation only, after freezing the export.
TYPESAFE_CASES_PER_SOURCE = 10  # None evaluates all cases. Sample whole cases before token admission.
TYPESAFE_MAX_LENGTH = 12288  # Benchmark-only long-input panel; None enforces the export's 2,048-token cap.
PREPARE_ONLY = False  # True stops before loading model weights or training.
CONTROL_CONFIG = dict(CONFIG)
initialization_seed = recipe.experiment_seed(CONFIG, "initialization")
random.seed(initialization_seed)
torch.manual_seed(initialization_seed)
torch.cuda.manual_seed_all(initialization_seed)
assert CONFIG["max_steps"] > 0 and CONFIG["accumulation"] > 0
print(json.dumps(CONFIG, indent=2))
""")

md("""
## 5. Run offline correctness tests

These use a tiny, randomly initialized Qwen model with both DeltaNet and full attention.
They test gradients through both families, selected-logit projection, checkpoint recovery,
selection and export, loss ablations, atomic admission, exact counterfactual labels,
occurrence augmentation/resume, group uncertainty, confidence semantics and tamper rejection. They do
not download model assets or establish 4B model quality.
""")
code("""
import os
test_environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
subprocess.check_call([sys.executable, "-m", "unittest", "discover", "-s", str(WORK), "-p", "test*.py", "-v"], env=test_environment)
""")

md("""
## 6. Prepare and audit the pinned data

Only the tokenizer is loaded here. Whole source groups are assigned before augmentation.
Review the admitted counts, none prevalence and sample cases. `PREPARE_ONLY=True` is useful
for this inspection before spending training compute. Overlength counts reveal what this
short-context pilot leaves untested.
""")
code("""
from transformers import AutoTokenizer
import pandas as pd

CONFIG = dict(CONTROL_CONFIG)
tokenizer = AutoTokenizer.from_pretrained(recipe.MODEL_ID, revision=recipe.MODEL_REVISION, trust_remote_code=False)
source_hashes = {name: recipe.file_digest(WORK / name) for name in EMBEDDED_SOURCE_FILES}
source_sha = source_hashes["train.py"]
data_key = recipe.digest(dict(version=recipe.VERSION, config=CONFIG, code=source_hashes, model=recipe.MODEL_REVISION, sources=recipe.SOURCES))[:16]
DATA_DIR = OUTPUT_ROOT / ("data_" + data_key)
data, manifest = recipe.prepare_data(CONFIG, tokenizer, DATA_DIR)
counts = []
for role, rows in data.items():
    # Aggregate composition is inspectable; no reserved model scores are computed here.
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        counts.append(dict(role=role, source=source, n=len(subset),
                           none=sum(r["answer_key"] == "__none__" for r in subset),
                           max_tokens=max(len(r["input_ids"]) for r in subset)))
display(pd.DataFrame(counts))
print("Admission audit:", json.dumps(manifest["audit"], indent=2))
print("Inspect examples:", DATA_DIR / "INSPECT_TRAINING_EXAMPLES.json")
""")

md("""
## 7. Load the model and lock the run

The complete run identity includes resolved precision, package versions, data and source
hashes. The preflight performs a real forward/backward pass and requires finite nonzero
gradient signal in attention, DeltaNet and MLP adapters. An OOM stops the run: no example is
silently dropped. Lowering the length cap creates a separate experiment with new admission
counts. Resume an interrupted run by rerunning with unchanged settings and the same precision.
In sweep mode, model loading and gradient preflight occur separately for each arm below.
""")
code("""
import platform
import time
if not PREPARE_ONLY:
    import gc
    if "model" in globals():
        del model
    gc.collect(); torch.cuda.empty_cache()
    software = {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "huggingface_hub", "safetensors", "tokenizers"]}
    for package in ["flash-linear-attention", "causal-conv1d", "kernels"]:
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = "absent"
    if RUN_LOSS_SWEEP:
        sweep_context = dict(source_sha=source_sha, source_hashes=source_hashes, software=software, python=platform.python_version(),
                             gpu=torch.cuda.get_device_name(0), compute_capability=list(torch.cuda.get_device_capability(0)),
                             total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3)
        print("Sweep arms:", json.dumps(recipe.LOSS_SWEEP_ARMS, indent=2))
    else:
        random.seed(initialization_seed); torch.manual_seed(initialization_seed); torch.cuda.manual_seed_all(initialization_seed)
        model, model_report = recipe.load_model(CONFIG)
        identity = recipe.digest(dict(version=recipe.VERSION, config=CONFIG, source_hashes=source_hashes,
                                      data_hashes=manifest["hashes"], model=recipe.MODEL_REVISION,
                                      precision=model_report["precision"], compute_capability=model_report["compute_capability"], software=software))
        RUN = OUTPUT_ROOT / ("run_" + identity[:16])
        RUN.mkdir(parents=True, exist_ok=True)
        recipe.immutable_json(RUN / "CONFIG.json", dict(identity=identity, config=CONFIG, software=software, source_sha=source_sha, source_hashes=source_hashes,
                                                        data_manifest_sha256=recipe.digest(manifest), data_directory=str(DATA_DIR)))
        recipe.atomic_json(RUN / "ENVIRONMENT.json", dict(python=platform.python_version(), software=software, model=model_report))
        started = time.perf_counter()
        preflight = recipe.gradient_preflight(model, data["train"][0], CONFIG)
        torch.cuda.synchronize()
        preflight.update(seconds=time.perf_counter()-started, peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        recipe.atomic_json(RUN / "GRADIENT_PREFLIGHT.json", preflight)
        print("Run directory:", RUN)
        print(json.dumps(model_report, indent=2))
        print("Gradient preflight:", preflight)
""")

md("""
## 8. Train, checkpoint and select

Single-example microbatches avoid padding and keep memory bounded. Gradients accumulate for
16 examples per update; adapters train in FP32 with gradient clipping. All backbone weights
and vocabulary rows remain frozen. Attention checkpoint forward and recomputation both use
the math SDPA backend, carrying forward the repair from experiment 27.

Select the lowest source-macro development NLL among checkpoints passing per-source
accuracy/NLL/Brier and adequately populated class-recall constraints. Step zero can win. The
false-none guard applies separately where the source contains enough answerable choices.
Thresholds are preset pilot rules, not statistical guarantees with 64 examples per source.

Checkpoints include adapter, optimizer, scheduler, RNG, history and hashes. A committed
manifest is written last. An interrupted update is replayed from the last complete checkpoint.

With `RUN_LOSS_SWEEP=True`, select one arm by the same unsmoothed development NLL and retention
guards, including Brier. Each arm starts from the same seeded parent, with a fresh optimizer;
step-zero development reports must match. Ties prefer an earlier checkpoint, then arm order.
Only the selected arm reaches calibration and the gate. Gate failure retains its frozen
parent, never another sweep arm. These six fits are a pilot, not a statistical guarantee or
a search over acceptance/final labels.
""")
code("""
if not PREPARE_ONLY:
    if RUN_LOSS_SWEEP:
        CONFIG = dict(CONTROL_CONFIG)
        if "model" in globals():
            del model
        gc.collect(); torch.cuda.empty_cache()
        sweep = recipe.run_loss_sweep(recipe.load_model, data, CONFIG, OUTPUT_ROOT, sweep_context, tokenizer=tokenizer)
        display(pd.DataFrame([dict(arm=r["name"], smoothing=r["config"]["label_smoothing"],
                                  brier_weight=r["config"]["brier_weight"], selected_step=r["selected_step"],
                                  macro_nll=r["metrics"]["macro_nll"], macro_accuracy=r["metrics"]["macro_accuracy"],
                                  macro_brier=sum(v["brier"] for v in r["metrics"]["by_source"].values())/len(r["metrics"]["by_source"]))
                              for r in sweep["results"]]))
        CONFIG = dict(sweep["selected_config"])
        RUN = Path(sweep["selected_run"])
        identity = sweep["selected_identity"]
        selection = json.loads((RUN / "SELECTION.json").read_text())
        model, model_report = recipe.load_model(CONFIG)
        assert {k: model_report[k] for k in ("precision", "compute_capability")} == sweep["execution"]
        recipe.restore(RUN / "checkpoints" / f"step_{selection['selected_step']:06d}", model, identity)
        print("Selected loss arm:", sweep["selected_arm"])
    else:
        selection = recipe.fit(model, data, CONFIG, RUN, identity, tokenizer=tokenizer)
    history = pd.DataFrame([dict(step=h["step"], macro_nll=h["metrics"]["macro_nll"],
                                macro_accuracy=h["metrics"]["macro_accuracy"],
                                passes_retention=h["retention"]["passed"],
                                reasons="; ".join(h["retention"]["reasons"])) for h in selection["history"]])
    display(history)
    print("Selected update:", selection["selected_step"], "(0 means retain the frozen parent)")
""")

md("""
## 9. Fit calibration, then use a separate acceptance gate

Fit one scalar temperature on calibration rows after checkpoint selection. The acceptance
gate compares it with temperature 1, requiring non-regression in NLL and Brier separately
for every source. It then tests the candidate against the matched frozen parent. Gate
failures retain the parent; they do not trigger another checkpoint or temperature search.

Reports include source/family/kind metrics, predicted-class distributions, majority controls,
paired correctness and complete-request correctness for declared multi-question fixtures.
Independent group counts and paired group-bootstrap intervals expose uncertainty. Existing
source/class, NLL/Brier and false-none thresholds remain pilot screens; sparse slices cannot
confirm retention. Report maximum-probability and exported entropy-confidence risk/coverage
separately. Neither defines an application authorization policy. Noul has no confidence field.
""")
code("""
if not PREPARE_ONLY:
    gate = recipe.calibrate_and_gate(model, data, CONFIG, RUN, identity, selection)
    print("Evaluated checkpoint:", selection["selected_step"],
          "(frozen parent)" if selection["selected_step"] == 0 else "(trained checkpoint)")
    display(pd.DataFrame(gate["candidate_raw"]["by_source"]).T[["n", "accuracy", "nll", "brier", "ece", "none_recall", "false_none_rate"]])
    print("Decision:", gate["decision"])
    print("Evaluated temperature:", gate["deployed_temperature"], "accepted:", gate["calibration_accepted"])
    print("Retention gate:", gate["retention"])
""")

md("""
### Schema sensitivity on gate groups

E42 found substantial order/key sensitivity despite process isolation. Test option display
order and code assignment separately, their combined reversal, and opaque Choice keys while
preserving descriptions and semantic none. Compare the selected candidate with its frozen parent on the same gate groups.
Log canonical probability vectors, total variation and winner changes per source; reject
overlength variants rather than truncate. Question IDs never enter this trainer's prompt.

These are descriptive diagnostics, not newly invented numerical parity thresholds. They do
not select another checkpoint or change the export. High sensitivity remains a deployment
qualification failure to investigate; a passing small gate is not native qualification.
They do not test sibling-field interactions or semantic paraphrases. Occurrence-based
presentation augmentation is separately selected in the training configuration.
""")
code("""
if not PREPARE_ONLY and CONFIG["schema_diagnostics"]:
    selected_path = RUN / "checkpoints" / f"step_{selection['selected_step']:06d}"
    recipe.restore(selected_path, model, identity)
    candidate_schema = recipe.schema_diagnostics(model, tokenizer, data["gate"], CONFIG["max_length"], gate["deployed_temperature"])
    same_parent = selection["selected_step"] == 0 and gate["deployed_temperature"] == 1.0
    if same_parent:
        parent_schema = candidate_schema
    else:
        recipe.restore(RUN / "checkpoints/step_000000", model, identity)
        parent_schema = recipe.schema_diagnostics(model, tokenizer, data["gate"], CONFIG["max_length"])
    recipe.restore(selected_path, model, identity)
    recipe.immutable_json(RUN / "SCHEMA_DIAGNOSTICS.json", dict(identity=identity, selected_step=selection["selected_step"], candidate=candidate_schema, parent=parent_schema))
    print(f"Selected checkpoint schema sensitivity (step {selection['selected_step']}):", json.dumps(candidate_schema["by_transform_and_source"], indent=2))
    if same_parent:
        print("Selected checkpoint is the frozen parent; reusing the identical schema diagnostics.")
    else:
        print("Parent schema sensitivity:", json.dumps(parent_schema["by_transform_and_source"], indent=2))
""")

md("""
## 10. Export a reproducible research bundle

The export contains a PEFT adapter, tokenizer, exact prompt/readout contract, source pins,
selection/calibration reports, executed implementation and file hashes. A small synthetic
replay pack records rendered inputs, token/code mappings, logits, accepted temperature,
probability vectors and typed answers for later native qualification. A failed gate exports step zero, with the
trained candidate preserved under checkpoints for inspection. The adapter needs the pinned
base weights. The archive keeps `export/` beside `EXPORT_LOCK.json`; preserve that layout
when extracting it. It includes calibration/gate prediction targets, but no original source
text, TypeSafe data or base checkpoint. Colocated hashes check consistency, not authenticity.
For direct imports from an extracted bundle, start Python with `PYTHONDONTWRITEBYTECODE=1`
so importing does not add untracked bytecode files to the frozen inventory.
The caller supplies the pinned parent weights; the loader binds base/adapter configurations
without independently hashing every caller-loaded base tensor.

This is not a registry installation or a ready-made GGUF. For Mac deployment, merge the
adapter into the pinned unquantized base, convert/quantize in the target runtime, reproduce
the finite-code prompt and row selection, then rerun paired quality, calibration, memory and
latency checks. In particular NF4 training and Mac Q4 inference are different numerical
paths. A selected-token finite-code readout still needs native profile qualification.

Frozen final evaluation and benchmarking share a verifier/loader that binds the executed
trainer, tokenizer, adapter and calibration contract to the export before scoring. The
bundle's `train.py:decide` reference helper accepts a caller-loaded pinned base and verified adapter. It admits the entire request before inference, enforces the training token
cap and described semantic none, and evaluates each question separately. It returns finite
distributions and normalized-entropy confidence (no Noul confidence). It is not a wire adapter,
shared-prefix cache or CLEF head implementation. Do not load the base's usual generation API
or OpenKind's fitted candidate scorer and call that the same readout.
""")
code("""
if not PREPARE_ONLY:
    export = recipe.export_bundle(model, tokenizer, CONFIG, manifest, RUN, identity, selection, gate)
    archive = recipe.archive_export(RUN, identity)
    print("Export folder:", export)
    print("Archive on Drive:", archive)
    print(json.dumps(json.loads((export / "DECISION_CONTRACT.json").read_text()), indent=2))
""")

md("""
## 11. Optional final evaluation after freezing the export

Runs only if `RUN_FINAL_TEST=True`. The candidate and matched parent are evaluated on
reserved source groups plus PAWS, SciQ and a held-out lookup/override composition. PAWS and
SciQ never enter training, development, calibration or the acceptance gate. Official Plumb
`test` is not used because its author used it for calibration. These small panels measure
transfer, not universal reasoning. No public JevBench items are used by this notebook.

The final report is descriptive and cannot change the exported model. If its findings guide
another recipe, treat this test as spent. Existing OpenKind final splits remain untouched.
""")
code("""
if not PREPARE_ONLY and RUN_FINAL_TEST:
    final = recipe.final_evaluation(model, data, tokenizer, CONFIG, RUN, identity, gate)
    display(pd.DataFrame(final["candidate"]["by_source"]).T[["n", "accuracy", "nll", "brier", "ece"]])
    print("Final report:", RUN / "FINAL_REPORT.json")
else:
    print("Final evaluation not run. Set RUN_FINAL_TEST=True only after freezing this recipe.")
""")

md("""
## 12. Optional TypeSafe benchmark after freezing the export

Set `RUN_TYPESAFE_BENCHMARK=True` to compare the frozen export and zero-update parent on
the five pinned `typesafe/evalsafe-*` test datasets. The default samples ten complete cases
per source before checking token length; set `TYPESAFE_CASES_PER_SOURCE=None` for all cases.
`TYPESAFE_MAX_LENGTH=12288` tests a declared longer context than the 2,048-token training
and export cap. Set it to `None` to enforce the export cap, which admitted zero invoice
questions in the tokenizer audit. Lengths above the export cap remain unqualified for
deployment and can need more GPU memory; an OOM stops evaluation. The report records both
caps and never changes the export contract. There is no training, calibration, checkpoint
selection or gate decision in this cell.
Viewing this benchmark spends it for subsequent tuning. Unknown base pretraining exposure
is not excluded.

O*NET uses `1 - JSD` with base-2 logs for Noul/Choice and range-normalized expected-score
agreement for Score, averaged equally across question types. References are synthetic
Astra/Fable consensus distributions, not human ground truth. The other four datasets are
fixed-input question diagnostics here: their official metrics require executing complete
policies and comparing actions, including arguments. Those workflow scores are not produced.

OpenKind requires described `__none__` for Choice, so missing none outcomes are appended with
zero reference mass. Predicted none mass remains in the divergence. This changes those
schemas and prevents a direct CLEF/TypeSafe leaderboard comparison. Report coverage and
all admission failures: the declared token and 16-outcome bounds reject some published records.
Overlength, unsupported and failed rows score zero in the selected-case denominator; do
not silently replace them with shorter examples. Model results never modify the export.
Local serial wall time includes Python overhead and is not the published service or
workflow latency. Dataset pins, file hashes, case sampling and probability vectors are saved.
""")
code("""
if not PREPARE_ONLY and RUN_TYPESAFE_BENCHMARK:
    import benchmark as typesafe_benchmark
    importlib.reload(typesafe_benchmark)
    prepared_groups = {row["group"] for role_rows in data.values() for row in role_rows}
    typesafe_result = typesafe_benchmark.run_benchmark(
        model, tokenizer, CONFIG, RUN, identity,
        excluded_groups=prepared_groups, cases_per_source=TYPESAFE_CASES_PER_SOURCE,
        max_length=TYPESAFE_MAX_LENGTH,
    )
    display(pd.DataFrame({name: {"question_agreement": report["equal_type_agreement"],
                                "answered": sum(k["answered"] for k in report["by_kind"].values()),
                                "selected_questions": sum(k["n"] for k in report["by_kind"].values())}
                         for name, report in typesafe_result["candidate"].items()}).T)
    print("TypeSafe question diagnostics saved. Official workflow action scores were not measured.")
else:
    print("TypeSafe references unopened. Enable RUN_TYPESAFE_BENCHMARK only after freezing the export.")
""")

md("""
## Sources and interpretation

- [Research history](https://github.com/whit3rabbit/openkind/blob/main/research/README.md): experiments 27–28 expose cross-task forgetting; experiment 29 distinguishes active MoE parameters from weight residency.
- [JevK5 v0.2 recipe](https://github.com/allebee/jevk5/blob/v0.2.0/README.md): checked teacher cases mixed with human-labeled tasks; temperature fit on held-out domains.
- [Jeeves pinned data manifest](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/data/manifest.json): explicitly distinguishes trainable sources from evaluation-only tasks.
- [Plumb dataset](https://huggingface.co/datasets/crh225/plumb-decisions/tree/718a9f006f3beaecfa7f66a819553f99ed7b94f6): teacher-generated cases and the actual purpose of each published split. Same-teacher double checking is not independent gold verification.
- [Imajev model card](https://huggingface.co/mohit67890/imajev-4b): broad labeled data, hard examples and replay motivate a mixed recipe, not an expected score for this notebook.
- [Winnow model card](https://huggingface.co/EldanRing/Winnow-12B): private training data; the named public tasks here do not reproduce Winnow.
- [Qwen3.5 text model](https://huggingface.co/docs/transformers/model_doc/qwen3_5) and [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization): framework contracts.
- [Cloudflare CLEF announcement](https://blog.cloudflare.com/clef-decision-models/) and [source review](../docs/RESEARCH.md#cloudflare-clef-and-linked-decision-models-reviewed-2026-10-01): learned routing head, one prefill, CE/Brier disclosure and reproduction limits. This notebook does not reproduce its rank-256 or RLCD recipe.
- [Pinned CLEF-flash head settings](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/joint_head_config.json): 1,024-wide head, two evidence-routing layers, four decoder layers and 16 heads. This is a 9B architecture reference, not a compatible 4B head or a disclosed loss coefficient.
- [TypeSafe datasets](https://huggingface.co/typesafe/datasets): five pinned evaluation-only sources, separated from every fitting and selection stage.
- [Strands Decider release](https://strandsagents.com/blog/introducing-strands-decider/) and [pinned v19 artifact](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19/tree/bb282d786bc251fd4e3068de3ada9ddbb38127cd): pointer head, rank-16 LoRA, base/parent KL retention and adequacy training. The released checkpoint uses a different readout and does not establish 4B gains.
- [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/990b2711a36180dd19d9c94b8627844866f8982a): optional human-rating proxy, source-specific CC-BY-4.0 terms, request-grouped splits. Imported only from the upstream train file.
- [Working paper](../docs/whitepaper/WORKING_PAPER.md) and [whitepaper §24](../docs/whitepaper/WHITEPAPER.md): E43/E44 capping, chronology, collateral errors and recovered encoder controls motivate diagnostics, without establishing that this training fixes them.
- [Decision records](local_decision_training/DECISIONS.md): evidence, alternatives, experiments and acceptance conditions for every v4 change.

This notebook distills decision behavior, including some teacher-computed distributions. It
does not transplant a reasoning module or prove that hidden reasoning transferred. Teacher
labels, one-hot human labels, and exact programmatic labels remain attributable in every row.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": TARGET.name, "provenance": []},
    "accelerator": "GPU",
})
for i, cell in enumerate(notebook.cells):
    cell.id = f"local-decisions-{i:02d}"
nbf.validate(notebook)
nbf.write(notebook, TARGET)
print(TARGET)
