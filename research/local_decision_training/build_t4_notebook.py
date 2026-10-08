"""Regenerate the self-contained, unattended T4 Colab notebook from reviewed source files."""
from pathlib import Path
import textwrap

import nbformat as nbf

HERE = Path(__file__).resolve().parent
FIXED_RECIPE_NOTEBOOK = globals().get("FIXED_RECIPE_NOTEBOOK", False)
TARGET = globals().get("NOTEBOOK_TARGET", HERE / ("local_decision_finetuning.ipynb" if FIXED_RECIPE_NOTEBOOK else "local_decision_training_t4.ipynb"))
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text, hidden=False):
    cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    if hidden:
        cell.metadata["cellView"] = "form"
    cells.append(cell)


md("""
# Train a local decision model on a free Colab T4, unattended

**OpenKind experiment 40, version 4. Reviewed 4 October 2026.**

This notebook is the T4 sibling of [notebook 35](../35_local_decision_training.ipynb). It embeds
the same reviewed trainer and test suite and trains a `Qwen/Qwen3.5-4B` rank-16 LoRA for typed
decisions, then sweeps, evaluates, calibrates, gates and exports a candidate. You do not need to
run notebook 35 first; nothing is read from it. "Relying on it" means this notebook inherits its
data recipe, split rules, retention guards and export contract.

**Built for an unattended T4 session.** Choose **Runtime > Change runtime type > T4 GPU**, then
**Runtime > Run all**, and leave it alone. Every choice is predeclared:

- No Drive authorization and no other prompts. Outputs stay on the Colab disk under
  `/content/openkind_local_decision_outputs` and are lost when the runtime dies. Set
  `USE_DRIVE=True` in section 3 if you will be present to authorize Drive and want the export
  to survive; interrupted runs resume from committed checkpoints when rerun unchanged.
- A T4 has no BF16 support, so the model runs the separate `nf4_fp16` run identity: 4-bit NF4
  weights with FP16 compute, FP32 LoRA adapters and the 1,024-token admission cap instead of
  notebook 35's 2,048. This is a different experiment, not a cheaper notebook 35.
- The default sweep compares three predeclared learning rates with matched seeds, data and
  budgets, then automatically continues with the development-selected winner: calibration,
  acceptance gate, schema diagnostics and export. The gate accepts or rejects; it never tries
  another arm. Set `SWEEP_MODE` to `"rank"`, `"loss"`, or `"none"` for other studies.

Budget for downloads, data admission, three 150-update fits, development evaluation,
calibration, gate and export. A full sweep can exceed a free session. Use this session's
`PERFORMANCE.json` to separate update, evaluation and checkpoint time before extrapolating;
T4 wall time cannot be estimated reliably from A100 evidence. A completed `SWEEP_RESULT.json`
and `GATE_RESULT.json` are the record.

**Status discipline.** Offline tests exercise data semantics, tiny hybrid gradients, sweeps,
resume, selection, calibration and export. This revised notebook has not completed 4B CUDA
validation, FP16 numerics qualification on T4 or Mac quality/speed measurements. Successful training is not evidence
that the adapted model beats its parent, and the exported adapter is a research bundle, not a
registered OpenKind profile.
""")

md("""
## 1. Experiment contract

The data recipe, source pins, caps and split rules are identical to
[notebook 35](../35_local_decision_training.ipynb) and its
[run guide](README.md): MultiNLI, BoolQ, Banking77, MultiRC, optional SST-5, the Plumb
Qwen-teacher set (half loss weight), and exact generated rules, capped at 8,000 admitted
training rows, split 75/8/7/5/5 into training, development, calibration, gate and reserved
test by normalized state group before augmentation. Teacher explanations never enter prompts;
TypeSafe datasets are benchmark-only. Read that guide and
[DECISIONS.md](DECISIONS.md) for the evidence behind each rule; this notebook changes the
execution substrate, not the recipe.

| Setting | Notebook 35 (A100/L4) | This T4 notebook |
|---|---|---|
| Precision identity | `bf16` or `nf4` (BF16 compute) | `nf4_fp16` (FP16 compute), resolved automatically |
| Token admission cap | 2,048 total prompt tokens | 1,024 total prompt tokens |
| Default updates | 400 × batch 16 | 150 × batch 16 |
| Default study | single plain-CE control | three learning rates, then automatic selection |
| Storage | Google Drive | local Colab disk (Drive optional, interactive) |

Changing precision, token cap, update budget, seeds, data settings or the embedded source
creates a different run identity rather than resuming incompatible work. Sweep rules follow the
same governance as the loss sweep: one bounded study per run identity, the configured control is
always one arm, every arm trains on the same admitted data with matched seeds and a fresh
optimizer, step-zero development reports must match across arms, and selection uses unsmoothed
source-macro development NLL subject to the existing per-source retention guards. Ties prefer an
earlier checkpoint, then arm order. Only the winner reaches calibration and the gate; a failed
gate retains the frozen parent. A second search that uses gate or final results spends those
panels and needs fresh acceptance groups.
""")

md("""
## 2. Install the training stack

Keep Colab's existing Torch/CUDA installation and pin the reviewed Transformers/PEFT/data stack
around it. Remove Colab's unused `torchao`: an older preinstalled version can make PEFT's
ordinary LoRA dispatch fail. This recipe uses bitsandbytes for NF4.
If Colab asks for a restart, restart the session and choose Run all again. Optional
custom DeltaNet kernels are deliberately absent; the Transformers reference path is slower but
avoids an unqualified backward kernel change.
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
    "wandb==0.28.1",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *requirements])
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU."
properties = torch.cuda.get_device_properties(0)
gib = properties.total_memory / 1024**3
assert gib >= 14, f"Need at least 14 GiB of GPU memory; reported {gib:.1f} GiB."
print("Torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), f"({gib:.1f} GiB)")
print("Native BF16 support:", torch.cuda.is_bf16_supported(including_emulation=False),
      "(T4 resolves to the nf4_fp16 run identity; larger GPUs resolve to bf16/nf4)")
""")

md("""
## 3. Workspace and outputs

The next hidden cell contains the complete trainer and offline tests; nothing executable is
fetched from a moving Git branch, and the same readable files live in
`research/local_decision_training/` in the repository. Expand it to inspect.

With the default `USE_DRIVE=False` nothing prompts: outputs land on the Colab disk and vanish
with the runtime, which is the price of a truly unattended run. Setting `USE_DRIVE=True` keeps
checkpoints, reports and the export on Drive across disconnects, but Drive requires interactive
authorization, so it is not suitable for an unsupervised session.
""")
code("""
from pathlib import Path

USE_DRIVE = False  # True persists outputs on Drive but needs interactive authorization.
WORK = Path("/content/openkind_local_decision_training")
OUTPUT_ROOT = (Path("/content/drive/MyDrive/OpenKind_Local_Decision_Training_T4") if USE_DRIVE
               else Path("/content/openkind_local_decision_outputs"))
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
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

These defaults run the full unattended pipeline: a three-arm learning-rate study (1e-5, the 2e-5
control, 4e-5), automatic selection, calibration, gate, diagnostics and export. The 1,024-token
cap bounds T4 memory with the MATH attention path; overlength records are rejected whole, never
truncated, and the admission audit reports them. `SWEEP_MODE` selects the one predeclared study:

- `"learning_rate"` (default) or `"rank"` — the new single-dimension parameter sweep.
- `"loss"` — the existing six-arm loss sweep; on a T4 it can exceed one session.
- `"none"` — one plain-CE control fit.

`CUSTOM_SWEEP_ARMS` may replace the default arms with 2–6 alternatives that include the
configured control value and vary nothing else; the validators reject mixed dimensions. Leave
`RUN_FINAL_TEST` and `RUN_TYPESAFE_BENCHMARK` off until a recipe is frozen; viewing those
results spends the reserved panels.

`USE_WANDB=True` tracks live optimizer metrics and development reports for each sweep arm,
or for a single fixed recipe. Local checkpoint and selection files govern the result.
`WANDB_MODE="offline"` saves logs without login. For `"online"`, set `WANDB_API_KEY` in the
environment or Colab Secrets with notebook access enabled, and optionally set `WANDB_ENTITY`.
Missing credentials stop before data preparation. Online runs resume under a stable run ID;
offline reruns produce a new log in the same group. `WANDB_LOG_EVERY` bounds update logging.
""")
code("""
import importlib
import json
import random
import sys

sys.path.insert(0, str(WORK))
import train as recipe
importlib.reload(recipe)

CONFIG = dict(recipe.DEFAULT_CONFIG)
CONFIG.update(
    max_length=1024,       # T4 memory bound; a separate experiment from the 2,048-token A100/L4 pilot.
    max_steps=150,
    accumulation=16,
    learning_rate=2e-5,
    rank=16,
    alpha=32,
    train_per_source=1000,
    teacher_train_cap=2000,
    eval_per_source=64,
    precision="auto",      # T4 resolves to nf4_fp16; A100/L4 resolve to bf16/nf4.
    include_sst5=True,
    include_teacher=True,
    include_helpsteer2=False,
    split_seed=17,
    initialization_seed=17,
    sampling_seed=17,
    augmentation_seed=17,
    data_intervention="control",        # "reasoning" replaces half the capped rules allocation.
    presentation_augmentation="none",   # "option_order", "code_assignment", "opaque_keys", or "all".
    label_smoothing=0.0,
    brier_weight=0.0,
)
SWEEP_MODE = "learning_rate"  # "learning_rate", "rank", "loss", or "none".
CUSTOM_SWEEP_ARMS = None      # Optional 2-6 arms including the control value; one dimension only.
USE_WANDB = False
WANDB_PROJECT = "openkind-local-decisions"
WANDB_ENTITY = None          # Set your W&B account or team for online tracking.
WANDB_MODE = "offline"       # "online" requires WANDB_API_KEY; neither mode asks for login.
WANDB_LOG_EVERY = 10
WANDB_LOG_REPORTS = False    # Optional small qualification reports; no weights or row-level predictions.
RUN_FINAL_TEST = False
RUN_TYPESAFE_BENCHMARK = False  # Evaluation only, after freezing the export.
TYPESAFE_CASES_PER_SOURCE = 10  # None evaluates all cases. Sample whole cases before token admission.
TYPESAFE_MAX_LENGTH = 12288     # Benchmark-only long-input panel; None enforces the export's 1,024-token cap.
PREPARE_ONLY = False            # True stops before loading model weights or training.
assert SWEEP_MODE in ("learning_rate", "rank", "loss", "none"), SWEEP_MODE
assert CONFIG["max_steps"] > 0 and CONFIG["accumulation"] > 0
# Validate the complete study before downloads, model loading or tracking initialization.
if SWEEP_MODE == "loss":
    sweep_configs = recipe.loss_sweep_configs(CONFIG, CUSTOM_SWEEP_ARMS)
elif SWEEP_MODE in ("learning_rate", "rank"):
    sweep_configs = recipe.parameter_sweep_configs(
        CONFIG, recipe.PARAMETER_SWEEP_ARMS[SWEEP_MODE] if CUSTOM_SWEEP_ARMS is None else CUSTOM_SWEEP_ARMS, SWEEP_MODE)
else:
    assert CUSTOM_SWEEP_ARMS is None, "CUSTOM_SWEEP_ARMS requires a sweep mode"
    sweep_configs = []
CONTROL_CONFIG = dict(CONFIG)  # Later cells use the winner; rerunning this study keeps the original control.
if USE_WANDB:
    assert WANDB_MODE in ("offline", "online"), WANDB_MODE
    assert type(WANDB_LOG_EVERY) is int and WANDB_LOG_EVERY > 0
    import os
    if WANDB_MODE == "online" and not os.environ.get("WANDB_API_KEY"):
        from google.colab import userdata
        try:
            os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
        except (userdata.SecretNotFoundError, userdata.NotebookAccessError) as exc:
            raise ValueError("Set WANDB_API_KEY in Colab Secrets with notebook access, or use WANDB_MODE='offline'.") from exc
    import wandb_tracking
    importlib.reload(wandb_tracking)
initialization_seed = recipe.experiment_seed(CONFIG, "initialization")
random.seed(initialization_seed)
import torch
torch.manual_seed(initialization_seed)
torch.cuda.manual_seed_all(initialization_seed)
print(json.dumps(CONFIG, indent=2))
""")

md("""
## 5. Run offline correctness tests

Tiny, randomly initialized Qwen models with both DeltaNet and full attention verify gradients
through both families, selected-logit projection, checkpoint recovery, sweep matching and
selection, calibration, export and tamper rejection. They download no model assets and do not
measure 4B quality. A failure here stops the session before any GPU time is spent.
""")
code("""
import os
test_environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
subprocess.check_call([sys.executable, "-m", "unittest", "discover", "-s", str(WORK), "-p", "test*.py", "-v"], env=test_environment)
""")

md("""
## 6. Prepare and audit the pinned data

Only the tokenizer is loaded here. Whole source groups are assigned before augmentation; review
the admitted counts, none prevalence and overlength rejections before the sweep starts. The
1,024-token cap admits fewer long rows than notebook 35's 2,048 and its audit quantifies exactly
what this pilot leaves untested. `PREPARE_ONLY=True` inspects this without training.
""")
code("""
from transformers import AutoTokenizer
import pandas as pd

CONFIG = dict(CONTROL_CONFIG)
tokenizer = AutoTokenizer.from_pretrained(recipe.MODEL_ID, revision=recipe.MODEL_REVISION, trust_remote_code=False)
source_hashes = {name: recipe.file_digest(WORK / name) for name in EMBEDDED_SOURCE_FILES}
source_sha = source_hashes["train.py"]
data_key = recipe.digest(dict(version=recipe.VERSION, config=CONFIG, code=source_hashes,
                              model=recipe.MODEL_REVISION, sources=recipe.SOURCES))[:16]
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
## 7. Run the predeclared sweep, or one control fit

In sweep modes the study loads one freshly seeded model per arm, runs the gradient preflight,
trains with a fresh optimizer, and resumes each arm's committed checkpoints if the session was
rerun. Every step-zero development report must match; precision drift or a mismatched parent
stops the sweep. A T4 runs arms sequentially; at defaults this cell performs three 150-update
fits plus periodic development evaluation.

With `SWEEP_MODE="none"`, a single control fit runs instead. In both cases this cell ends with
the selected checkpoint restored into `model`, the selected run directory in `RUN`, and the
frozen selection in `selection`. Step zero can win selection, which means the frozen parent is
kept; the gate still evaluates that outcome.
""")
code("""
import gc
import platform
import time
from pathlib import Path

if not PREPARE_ONLY:
    if "model" in globals():
        del model
    gc.collect(); torch.cuda.empty_cache()
    software = {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "huggingface_hub", "safetensors", "tokenizers"]}
    for package in ["flash-linear-attention", "causal-conv1d", "kernels"]:
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = "absent"
    sweep_context = dict(source_sha=source_sha, source_hashes=source_hashes, software=software,
                         python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
                         compute_capability=list(torch.cuda.get_device_capability(0)),
                         total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3)
    CONFIG = dict(CONTROL_CONFIG)
    sweep = None
    if USE_WANDB and SWEEP_MODE != "none":
        sweep = wandb_tracking.run_sweep_with_wandb(
            recipe.load_model, data, CONFIG, OUTPUT_ROOT, sweep_context, SWEEP_MODE,
            arms=CUSTOM_SWEEP_ARMS, tokenizer=tokenizer,
            project=WANDB_PROJECT, entity=WANDB_ENTITY, mode=WANDB_MODE, log_every=WANDB_LOG_EVERY)
    elif SWEEP_MODE == "loss":
        sweep = recipe.run_loss_sweep(recipe.load_model, data, CONFIG, OUTPUT_ROOT, sweep_context,
                                      arms=CUSTOM_SWEEP_ARMS, tokenizer=tokenizer)
    elif SWEEP_MODE in ("learning_rate", "rank"):
        sweep = recipe.run_parameter_sweep(recipe.load_model, data, CONFIG, OUTPUT_ROOT, sweep_context,
                                           SWEEP_MODE, arms=CUSTOM_SWEEP_ARMS, tokenizer=tokenizer)
    else:
        model, model_report = recipe.load_model(CONFIG)
        identity = recipe.digest(dict(version=recipe.VERSION, config=CONFIG, source_hashes=source_hashes,
                                      data_hashes=manifest["hashes"], model=recipe.MODEL_REVISION,
                                      precision=model_report["precision"],
                                      compute_capability=model_report["compute_capability"], software=software))
        RUN = OUTPUT_ROOT / ("run_" + identity[:16])
        RUN.mkdir(parents=True, exist_ok=True)
        recipe.immutable_json(RUN / "CONFIG.json", dict(identity=identity, config=CONFIG, software=software,
                                                        source_sha=source_sha, source_hashes=source_hashes,
                                                        data_manifest_sha256=recipe.digest(manifest), data_directory=str(DATA_DIR)))
        recipe.atomic_json(RUN / "ENVIRONMENT.json", dict(python=platform.python_version(), software=software,
                                                          model=model_report))
        started = time.perf_counter()
        preflight = recipe.gradient_preflight(model, data["train"][0], CONFIG)
        torch.cuda.synchronize()
        preflight.update(seconds=time.perf_counter()-started, peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        recipe.atomic_json(RUN / "GRADIENT_PREFLIGHT.json", preflight)
        print("Run directory:", RUN)
        print("Gradient preflight:", preflight)
        if USE_WANDB:
            selection = wandb_tracking.run_finetune_with_wandb(
                model, data, CONFIG, RUN, identity, tokenizer=tokenizer,
                project=WANDB_PROJECT, entity=WANDB_ENTITY, mode=WANDB_MODE, log_every=WANDB_LOG_EVERY)
        else:
            selection = recipe.fit(model, data, CONFIG, RUN, identity, tokenizer=tokenizer)
    if sweep is not None:
        varying = sorted({k for r in sweep["results"] for k in r["config"]
                          if len({other["config"].get(k) for other in sweep["results"]}) > 1})
        display(pd.DataFrame([dict(arm=r["name"], **{k: r["config"][k] for k in varying},
                                   selected_step=r["selected_step"], macro_nll=r["metrics"]["macro_nll"],
                                   macro_accuracy=r["metrics"]["macro_accuracy"],
                                   macro_brier=sum(v["brier"] for v in r["metrics"]["by_source"].values())
                                                  / len(r["metrics"]["by_source"]))
                              for r in sweep["results"]]))
        CONFIG = dict(sweep["selected_config"])
        RUN = Path(sweep["selected_run"])
        identity = sweep["selected_identity"]
        selection = json.loads((RUN / "SELECTION.json").read_text())
        model, model_report = recipe.load_model(CONFIG)
        assert {k: model_report[k] for k in ("precision", "compute_capability")} == sweep["execution"]
        recipe.restore(RUN / "checkpoints" / f"step_{selection['selected_step']:06d}", model, identity)
        print("Selected arm:", sweep["selected_arm"])
    history = pd.DataFrame([dict(step=h["step"], macro_nll=h["metrics"]["macro_nll"],
                                 macro_accuracy=h["metrics"]["macro_accuracy"],
                                 passes_retention=h["retention"]["passed"],
                                 reasons="; ".join(h["retention"]["reasons"])) for h in selection["history"]])
    display(history)
    print("Selected update:", selection["selected_step"], "(0 means retain the frozen parent)")
    if (RUN / "PERFORMANCE.json").exists():
        print("Timing (latest training invocation):", json.loads((RUN / "PERFORMANCE.json").read_text()))
""")

md("""
## 8. Fit calibration, then use a separate acceptance gate

One scalar temperature is fitted on calibration rows after checkpoint selection. The acceptance
gate requires non-regression in NLL and Brier separately for every source, both against
temperature 1 and against the matched frozen parent. Gate failures retain the parent; they do
not trigger another checkpoint, temperature or sweep-arm search.
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

E42 found substantial order/key sensitivity despite process isolation. The paired diagnostics
reverse option display order and code assignment and rename opaque Choice keys while preserving
descriptions and semantic none, comparing the selected candidate with its frozen parent on the
same gate groups. They are descriptive, change no selection, and do not establish native parity.
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
## 9. Export a reproducible research bundle

The export contains the PEFT adapter, tokenizer, exact prompt/readout contract, source pins,
sweep plan and result, selection/calibration reports, the executed implementation and file
hashes. A failed gate exports step zero. Trained checkpoints remain in the run directory,
outside the export ZIP; preserve that full directory to keep their weights for inspection.
Keep `export/` beside `EXPORT_LOCK.json` when extracting. The caller supplies the pinned parent
weights; the bundle does not redistribute base tensors. This is not a registry installation: Mac
deployment still needs adapter merge, quantization, and paired native quality, calibration,
memory and latency checks. NF4 FP16 training and Mac Q4 inference are different numerical paths.
""")
code("""
if not PREPARE_ONLY:
    export = recipe.export_bundle(model, tokenizer, CONFIG, manifest, RUN, identity, selection, gate)
    archive = recipe.archive_export(RUN, identity)
    print("Export folder:", export)
    print("Archive:", archive)
    print(json.dumps(json.loads((export / "DECISION_CONTRACT.json").read_text()), indent=2))
""")

md("""
## 10. Best-candidate summary

A compact record of what the unattended session decided and why. The selected arm and step come
from development data only; the gate compares the calibrated candidate with the frozen parent.
A candidate that fails the gate is not a worse notebook — it is the guard doing its job. If the
winner sits at a sweep boundary (the lowest or highest value tried), extending that range is a
new bounded study, not an edit to this one.
""")
code("""
if not PREPARE_ONLY:
    winner = selection["selected_step"] > 0 and gate["decision"] == "experimental_candidate_passed"
    best_observed = min(selection["history"], key=lambda h: (h["metrics"]["macro_nll"], h["step"]))
    exported_contract = json.loads((export / "DECISION_CONTRACT.json").read_text())
    summary = dict(schema=recipe.VERSION, sweep_mode=SWEEP_MODE,
                   selected_arm=sweep["selected_arm"] if sweep else "control",
                   selected_step=selection["selected_step"],
                   macro_nll=selection["history"][[h["step"] for h in selection["history"]]
                                                 .index(selection["selected_step"])]["metrics"]["macro_nll"],
                   best_observed_development_step=best_observed["step"],
                   best_observed_development_macro_nll=best_observed["metrics"]["macro_nll"],
                   best_observed_development_passes_retention=best_observed["retention"]["passed"],
                   development_rejections=[dict(step=h["step"], reasons=h["retention"]["reasons"])
                                           for h in selection["history"] if not h["retention"]["passed"]],
                   fitted_temperature=gate["fitted_temperature"],
                   evaluated_temperature=gate["deployed_temperature"],
                   deployed_temperature=exported_contract["temperature"],
                   gate_decision=gate["decision"], experimental_candidate_exported=bool(winner),
                   model_promoted=False,
                   run_directory=str(RUN), export_directory=str(export), archive=str(archive))
    recipe.atomic_json(RUN / "BEST_MODEL_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
    if USE_WANDB:
        wandb_tracking.log_qualification(RUN, identity, project=WANDB_PROJECT, entity=WANDB_ENTITY,
                                        mode=WANDB_MODE, log_reports=WANDB_LOG_REPORTS)
""")

md("""
## 11. Optional final evaluation after freezing the export

Runs only if `RUN_FINAL_TEST=True`, which an unattended default run leaves off. It evaluates the
frozen export on reserved source groups plus PAWS, SciQ and a held-out composition. Viewing that
result spends it for subsequent tuning.
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

Runs only if `RUN_TYPESAFE_BENCHMARK=True`. Compares the frozen export and zero-update parent on
the five pinned `typesafe/evalsafe-*` snapshots, evaluation only. `TYPESAFE_MAX_LENGTH=12288`
tests a declared longer context than the 1,024-token export cap; extended context remains
unqualified for deployment and an OOM stops the benchmark. There is no training, selection or
gate decision in this cell.
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

- [Notebook 35](../35_local_decision_training.ipynb) and the [run guide](README.md): the A100/L4
  recipe, dataset rationale and mixture caps this notebook inherits unchanged.
- [Decision records](DECISIONS.md): evidence, alternatives and acceptance conditions, including
  the fp16 execution path and the single-dimension parameter sweep introduced for T4 sessions.
- [Local design](../../docs/whitepaper/LOCAL_DECISION_DESIGN.md): separates this pilot from
  native integration. [Whitepaper §24](../../docs/whitepaper/WHITEPAPER.md) carries E43/E44
  diagnostics this notebook measures but does not fix.
- [QLoRA](https://huggingface.co/docs/peft/developer_guides/quantization) and
  [bitsandbytes](https://huggingface.co/docs/bitsandbytes/index): 4-bit NF4 training with FP16
  compute is a standard path for FP16-only GPUs; its numerics here are a hypothesis this
  notebook exists to test, not a qualified configuration.
- [Qwen3.5 text model](https://huggingface.co/docs/transformers/model_doc/qwen3_5): the pinned
  backbone and its hybrid attention/DeltaNet layers.
- [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/): Turing architecture, 16 GB
  GDDR6, no BF16 support — the hardware constraint behind the separate run identity.

This notebook distills decision behavior, including some teacher-computed distributions. It does
not transplant a reasoning module, prove that the adapted model beats its parent, or authorize
any action. The Rust engine keeps its existing qualified profiles; experiment 40 needs merged
artifacts, offline probability fixtures and a new renderer/profile before any integration.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": TARGET.name, "provenance": []},
    "accelerator": "GPU",
})
for i, cell in enumerate(notebook.cells):
    cell.id = f"local-decisions-t4-{i:02d}"
if FIXED_RECIPE_NOTEBOOK:
    notebook.cells[0].source = """# Fine-tune a fixed OpenKind decision recipe

Choose one recipe from development evidence before running this notebook. It performs one
Qwen3.5-4B LoRA fit, selection, calibration, the retention gate and a frozen research export.
It shares the trainer and tests with the T4 experiment notebook. Defaults are a bounded
T4 control (1,024 tokens, 150 updates); larger BF16 GPUs can use a separately declared recipe.

This is fine-tuning of pretrained weights. No model is pretrained from scratch here.
Use `CHOSEN_RECIPE_PATH` for a local `SWEEP_RESULT.json` or a complete configuration JSON.
The notebook trains from the pinned base, without another arm's optimizer or adapter.
Changing the recipe creates a new run identity. If gate results informed any tuning,
those groups cannot serve as fresh acceptance evidence.

Fine-tuning consumes the pinned MNLI, BoolQ, Banking77, MultiRC, SST-5, Plumb and generated
rules dataset mix. HelpSteer2 adequacy and generated reasoning are optional dataset interventions.
The separate [decision-RL notebook](local_decision_posttraining.ipynb) continues a trained
supervised checkpoint with task rewards and a frozen reference; RL is never enabled implicitly.

The [release notebook](local_decision_release.ipynb) prepares a verified adapter package
after export. An experiment's existing suitable export can also go straight to that notebook.
Hugging Face publishing is a separate, explicit step; this notebook never uploads.
"""
    for cell in notebook.cells:
        if cell.cell_type == "code" and 'SWEEP_MODE = "learning_rate"' in cell.source:
            cell.source = cell.source.replace('SWEEP_MODE = "learning_rate"', '''CHOSEN_RECIPE_PATH = ""  # Optional local SWEEP_RESULT.json or complete configuration JSON.
if CHOSEN_RECIPE_PATH:
    from pathlib import Path
    declared = json.loads(Path(CHOSEN_RECIPE_PATH).read_text())
    CONFIG = dict(declared.get("selected_config", declared))
    assert set(CONFIG) == set(recipe.DEFAULT_CONFIG), "Use a complete reviewed configuration"
SWEEP_MODE = "none"''', 1)
            cell.source += '\nassert SWEEP_MODE == "none", "Use the experiment notebook for sweeps"\n'
            cell.source += 'if not torch.cuda.is_bf16_supported(including_emulation=False):\n    assert CONFIG["max_length"] <= 1024, "The T4 recipe is bounded to 1,024 tokens"\n'
        elif cell.cell_type == "markdown" and cell.source.startswith("## 4. Parameters"):
            cell.source = """## 4. Declare the fixed recipe

Leave `CHOSEN_RECIPE_PATH` empty for the bounded plain-CE control, or point it at a local
development-selected sweep result. Inspect the complete configuration before training.
Use Drive for persistent checkpoints when interactive authorization is available.
`USE_WANDB=True` enables live training and development metrics; offline mode needs no login.
Keep final evaluation and TypeSafe benchmarks off until the recipe and export are frozen.
"""
        elif cell.cell_type == "markdown" and cell.source.startswith("## 7."):
            cell.source = """## 7. Fine-tune the declared recipe

One freshly seeded model trains with a fresh optimizer, or resumes this exact run's last
committed checkpoint. Development selection may retain step zero. Calibration and the
gate evaluate the selected outcome once; they never search alternative recipes.
"""
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace("three learning rates, then automatic selection", "one fixed plain-CE recipe")
    for i, cell in enumerate(notebook.cells):
        cell.id = f"local-decisions-finetune-{i:02d}"
nbf.validate(notebook)
nbf.write(notebook, TARGET)
print(TARGET)
