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

**OpenKind experiment 35, version 1. Prepared 29 September 2026.**

Train `Qwen/Qwen3.5-4B` with rank-16 LoRA for typed decisions. The model sees the state,
question, and all options together. Training applies cross-entropy to selected answer-code
logits from one forward pass. It does not generate reasoning or answer text.

**Recommended runtime: Colab A100.** An L4 uses NF4 QLoRA automatically. Both require native
BF16; this notebook does not qualify T4 training. Deployment remains targeted at a 16–32 GB
Mac, after adapter merge, quantization and native-runtime evaluation.

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

All public training records come from upstream **train** files. Split by normalized state
before constructing options: 75% training, 8% development, 7% calibration, 5% acceptance gate,
5% reserved test. Counterfactual siblings share their original group. Actual admitted counts
are recorded; the table gives caps, not guaranteed counts or a balanced-label promise.

The first run uses at most 400 updates × 16 microbatches = 6,400 presentations, sampled from
the prepared mixture in a deterministic shuffled order. Teacher rows receive loss weight 0.5.
They remain teacher labels, not human gold. Keep explanations out of inputs. Preserve soft
targets, renormalizing only four-decimal rounding errors within 0.001 total mass.

For eligible one-hot Choice records, deterministically remove the correct option in about
20% of cases and supervise `__none__`. Shuffle option/code assignments. Never infer semantic
none from confidence, nor label a truncated passage unanswerable. Reject inputs exceeding
2,048 **total prompt tokens**, including question, options and template.

The separate optional final cell evaluates reserved groups, PAWS, SciQ and an unseen rule
composition. Source labels are processed when preparing reserved files; their model results
are not consulted for selection. No historical OpenKind final corpus or QASPER labels are
read. Public datasets can occur in base-model pretraining; these are fine-tuning holdouts,
not a claim of pretraining-clean evaluation.

Dataset terms remain source-specific. The source manifest records pins, file hashes and
license metadata. SST-5's mirror does not specify a license, and MultiNLI/MultiRC have
source-specific terms. Turning off `include_sst5` is supported. This notebook does not
relicense or redistribute the source datasets.
""")

md("""
## 2. Install the training stack

Keep Colab's existing Torch/CUDA installation. Install the pinned Transformers/PEFT/data
stack before importing it. If Colab asks for a restart, restart the session and run all again.
Optional custom DeltaNet kernels are deliberately absent from this initial recipe: the
Transformers reference path is slower but avoids an unqualified backward kernel change.
""")
code("""
import importlib.metadata
import subprocess
import sys

requirements = [
    "transformers==5.17.0", "peft==0.21.1", "datasets==5.0.1",
    "accelerate==1.15.0", "bitsandbytes==0.50.2", "huggingface_hub==1.33.0",
    "safetensors==0.8.0", "tokenizers==0.23.2", "tqdm>=4.66", "pandas>=2.2",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *requirements])
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > A100 or L4 GPU."
assert torch.cuda.is_bf16_supported(), "Use A100/L4. T4's FP16 route is not qualified here."
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
code("# @title Embedded training implementation and offline tests\n" + "\n".join(
    f"(WORK / {name!r}).write_text({(HERE/name).read_text()!r}, encoding='utf-8')"
    for name in ("train.py", "test_train.py")), hidden=True)

md("""
## 4. Parameters

Leave these defaults for the first pilot. An A100 uses BF16 weights; an L4 uses 4-bit NF4
weights with BF16 compute. `max_length`, precision, data settings and source code are part of
the run identity. Changing them creates a new run rather than resuming incompatible work.

`RUN_FINAL_TEST=False` leaves the final evaluation cell inactive. Turn it on only once the
recipe and selection are frozen. Further tuning after viewing that result needs a new test.
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
)
RUN_FINAL_TEST = False
PREPARE_ONLY = False  # True stops before loading model weights or training.
random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])
assert CONFIG["max_steps"] > 0 and CONFIG["accumulation"] > 0
print(json.dumps(CONFIG, indent=2))
""")

md("""
## 5. Run offline correctness tests

These use a tiny, randomly initialized Qwen model with both DeltaNet and full attention.
They test gradients through both families, selected-logit projection, checkpoint recovery,
selection and export. They do not download model assets or establish 4B model quality.
""")
code("""
subprocess.check_call([sys.executable, "-m", "unittest", "discover", "-s", str(WORK), "-p", "test_train.py", "-v"])
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

tokenizer = AutoTokenizer.from_pretrained(recipe.MODEL_ID, revision=recipe.MODEL_REVISION, trust_remote_code=False)
source_sha = recipe.file_digest(WORK / "train.py")
data_key = recipe.digest(dict(config=CONFIG, code=source_sha, model=recipe.MODEL_REVISION, sources=recipe.SOURCES))[:16]
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
""")
code("""
import platform
import time
if not PREPARE_ONLY:
    model, model_report = recipe.load_model(CONFIG)
    software = {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "huggingface_hub", "safetensors", "tokenizers"]}
    for package in ["flash-linear-attention", "causal-conv1d", "kernels"]:
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = "absent"
    identity = recipe.digest(dict(version=recipe.VERSION, config=CONFIG, source_sha=source_sha,
                                  data_hashes=manifest["hashes"], model=recipe.MODEL_REVISION,
                                  precision=model_report["precision"], compute_capability=model_report["compute_capability"], software=software))
    RUN = OUTPUT_ROOT / ("run_" + identity[:16])
    RUN.mkdir(parents=True, exist_ok=True)
    recipe.immutable_json(RUN / "CONFIG.json", dict(identity=identity, config=CONFIG, software=software, source_sha=source_sha))
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
accuracy/NLL and adequately populated class-recall constraints. Step zero can win. The
false-none guard applies separately where the source contains enough answerable choices.
Thresholds are preset pilot rules, not statistical guarantees with 64 examples per source.

Checkpoints include adapter, optimizer, scheduler, RNG, history and hashes. A committed
manifest is written last. An interrupted update is replayed from the last complete checkpoint.
""")
code("""
if not PREPARE_ONLY:
    selection = recipe.fit(model, data, CONFIG, RUN, identity)
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

Reports include per-source accuracy, NLL, Brier, ECE, per-class recall, semantic-none recall,
false-none rates, ordinal expected-value error, and risk/coverage at several thresholds.
Treat ECE and high-confidence accuracy cautiously when their denominators are small.
""")
code("""
if not PREPARE_ONLY:
    gate = recipe.calibrate_and_gate(model, data, CONFIG, RUN, identity, selection)
    display(pd.DataFrame(gate["candidate_raw"]["by_source"]).T[["n", "accuracy", "nll", "brier", "ece", "none_recall", "false_none_rate"]])
    print("Decision:", gate["decision"])
    print("Temperature:", gate["deployed_temperature"], "accepted:", gate["calibration_accepted"])
    print("Retention gate:", gate["retention"])
""")

md("""
## 10. Export a reproducible research bundle

The export contains a PEFT adapter, tokenizer, exact prompt/readout contract, source pins,
selection/calibration reports and file hashes. A failed gate exports step zero, with the
trained candidate preserved under checkpoints for inspection. The adapter needs the pinned
base weights. No base checkpoint or source dataset is copied into the export archive.

This is not a registry installation or a ready-made GGUF. For Mac deployment, merge the
adapter into the pinned unquantized base, convert/quantize in the target runtime, reproduce
the finite-code prompt and row selection, then rerun paired quality, calibration, memory and
latency checks. In particular NF4 training and Mac Q4 inference are different numerical
paths. A selected-token finite-code readout still needs native profile qualification.
""")
code("""
import shutil
if not PREPARE_ONLY:
    export = recipe.export_bundle(model, tokenizer, CONFIG, manifest, RUN, identity, selection, gate)
    archive = shutil.make_archive(str(RUN / "openkind_decision_export"), "zip", root_dir=export)
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
## Sources and interpretation

- [Research history](https://github.com/whit3rabbit/openkind/blob/main/research/README.md): experiments 27–28 expose cross-task forgetting; experiment 29 distinguishes active MoE parameters from weight residency.
- [JevK5 v0.2 recipe](https://github.com/allebee/jevk5/blob/v0.2.0/README.md): checked teacher cases mixed with human-labeled tasks; temperature fit on held-out domains.
- [Jeeves pinned data manifest](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/data/manifest.json): explicitly distinguishes trainable sources from evaluation-only tasks.
- [Plumb dataset](https://huggingface.co/datasets/crh225/plumb-decisions/tree/718a9f006f3beaecfa7f66a819553f99ed7b94f6): teacher-generated cases and the actual purpose of each published split. Same-teacher double checking is not independent gold verification.
- [Imajev model card](https://huggingface.co/mohit67890/imajev-4b): broad labeled data, hard examples and replay motivate a mixed recipe, not an expected score for this notebook.
- [Winnow model card](https://huggingface.co/EldanRing/Winnow-12B): private training data; the named public tasks here do not reproduce Winnow.
- [Qwen3.5 text model](https://huggingface.co/docs/transformers/model_doc/qwen3_5) and [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization): framework contracts.

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
