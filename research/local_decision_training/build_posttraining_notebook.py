"""Build a portable, opt-in decision-RL stage using the shared Colab setup."""
from pathlib import Path
import runpy
import textwrap

import nbformat

HERE = Path(__file__).resolve().parent
TARGET = HERE / "local_decision_posttraining.ipynb"
namespace = runpy.run_path(str(HERE / "build_t4_notebook.py"),
                          init_globals={"FIXED_RECIPE_NOTEBOOK": True, "NOTEBOOK_TARGET": TARGET})
notebook = namespace["notebook"]

notebook.cells[0].source = """# Optional decision RL after supervised fine-tuning

Load a verified trained OpenKind Qwen3.5-4B checkpoint, then run one declared calibration-aware
decision optimization experiment. This is an open RLCD-inspired experiment, not a reproduction
of Cloudflare's undisclosed reward implementation. No text rollouts or text generation are used.

The objective combines sampled decision utility (RLOO REINFORCE), adjacent ordinal credit,
complete-record rewards, direct CE/Brier gradients, and KL to the frozen supervised reference.
Reference logits are cached once, so a second 4B model does not occupy GPU memory.
`RL_CONFIG['method']` can instead select `supervised` (CE/Brier/KL only) or `expected_utility`
(exact finite-action expectation) for matched comparisons. Declare each experiment separately.

Supply the full supervised run and its original prepared-data directory, not only an export ZIP.
The default uses its development-selected trained checkpoint. If selection retained step zero,
an explicit positive `SUPERVISED_STEP` may start an experimental recovery study from a rejected
checkpoint. Its provenance records that choice; it does not promote that checkpoint.

The dataset mix is inherited from supervised fine-tuning. Generated reasoning training is
enabled for complete multi-field requests. HelpSteer2 remains optional; PAWS, SciQ and TypeSafe
remain evaluation-only. New acceptance groups exclude the supplied earlier evidence groups
without reassigning their roles. Supply every earlier data directory you consulted.

Defaults train and compare on development only. `RUN_ACCEPTANCE=False` keeps calibration,
gate scores and release export closed until the method is frozen. This stage has offline tiny-model
coverage; full 4B T4 numerics, memory, quality and speed remain unmeasured.
"""

parameters = '''
import importlib
import json
import sys
sys.path.insert(0, str(WORK))
import train as recipe
import decision_rl
import wandb_tracking
importlib.reload(recipe)
importlib.reload(decision_rl)
importlib.reload(wandb_tracking)

SUPERVISED_RUN = ""       # Full run directory containing CONFIG.json and checkpoints/.
SUPERVISED_DATA_DIR = ""  # Original data_... directory containing DATA_MANIFEST.json and role JSONs.
SUPERVISED_STEP = None    # None uses selected_step; an explicit positive step is experimental.
PRIOR_EVIDENCE_DATA_DIRS = []  # Also list every other earlier dataset directory whose gate/test you viewed.
assert SUPERVISED_RUN and SUPERVISED_DATA_DIR, "Set SUPERVISED_RUN and SUPERVISED_DATA_DIR before Run all"
SOURCE_CONFIG, SOURCE_CHECKPOINT, SOURCE = decision_rl.source_checkpoint(SUPERVISED_RUN, SUPERVISED_STEP)
CONFIG = dict(SOURCE_CONFIG)
CONFIG.update(max_steps=50, accumulation=8, learning_rate=5e-6,
              evaluate_every=25, checkpoint_every=25, data_intervention="reasoning")
# Existing public datasets stay enabled as in the supervised recipe. Optional adequacy data:
# CONFIG["include_helpsteer2"] = True
RL_CONFIG = dict(decision_rl.DEFAULT_RL_CONFIG)
RL_CONFIG.update(method="reinforce")  # "supervised", "expected_utility", or "reinforce".
decision_rl.validate_config(RL_CONFIG)
CONTROL_CONFIG = dict(CONFIG)
SWEEP_MODE, sweep = "none", None
PREPARE_ONLY = False
RUN_ACCEPTANCE = False  # Freeze the method on development before opening a fresh gate.
USE_WANDB = False
WANDB_PROJECT = "openkind-local-decisions"
WANDB_ENTITY = None
WANDB_MODE = "offline"
WANDB_LOG_EVERY = 10
WANDB_LOG_REPORTS = False
RUN_FINAL_TEST = False
RUN_TYPESAFE_BENCHMARK = False
TYPESAFE_CASES_PER_SOURCE = 10
TYPESAFE_MAX_LENGTH = 12288
if USE_WANDB:
    assert WANDB_MODE in ("offline", "online", "disabled")
    if WANDB_MODE == "online":
        import os
        assert os.environ.get("WANDB_API_KEY"), "Set WANDB_API_KEY for online tracking"
if not torch.cuda.is_bf16_supported(including_emulation=False):
    assert CONFIG["max_length"] <= 1024, "Use the bounded T4 source recipe"
print("Supervised checkpoint:", SOURCE_CHECKPOINT)
print("RL objective:", json.dumps(RL_CONFIG, indent=2))
print("Acceptance enabled:", RUN_ACCEPTANCE)
'''

data_cell = '''
from transformers import AutoTokenizer
import pandas as pd

CONFIG = dict(CONTROL_CONFIG)
tokenizer = AutoTokenizer.from_pretrained(recipe.MODEL_ID, revision=recipe.MODEL_REVISION, trust_remote_code=False)
original_data, original_manifest = decision_rl.read_source_data(SUPERVISED_DATA_DIR, SUPERVISED_RUN)
excluded = set(decision_rl.fresh_evidence_groups(original_data))
for directory in PRIOR_EVIDENCE_DATA_DIRS:
    previous = {role: json.loads((Path(directory) / f"{role}.json").read_text()) for role in recipe.ROLES}
    previous_manifest = json.loads((Path(directory) / "DATA_MANIFEST.json").read_text())
    assert all(recipe.digest(rows) == previous_manifest["hashes"][role] for role, rows in previous.items())
    excluded.update(decision_rl.fresh_evidence_groups(previous))
source_hashes = {name: recipe.file_digest(WORK / name) for name in EMBEDDED_SOURCE_FILES}
source_sha = source_hashes["train.py"]
data_key = recipe.digest(dict(config=CONFIG, sources=recipe.SOURCES, source=SOURCE,
                              excluded=sorted(excluded), trainer=source_sha))[:16]
DATA_DIR = OUTPUT_ROOT / ("rl_data_" + data_key)
data, manifest = recipe.prepare_data(CONFIG, tokenizer, DATA_DIR, excluded_groups=excluded)
assert not excluded.intersection(row["group"] for role in recipe.ROLES for row in data[role])
units = decision_rl.training_units(data["train"])
if RL_CONFIG["exact_record_weight"] and RL_CONFIG["method"] != "supervised":
    assert any(unit["is_record"] for unit in units), "No complete records were admitted"
display(pd.DataFrame([dict(role=role, source=source, rows=sum(row["source"] == source for row in rows))
                      for role, rows in data.items() for source in sorted({row["source"] for row in rows})]))
print("Complete multi-field training requests:", sum(unit["is_record"] for unit in units))
print("Excluded earlier evidence groups:", len(excluded))
print("Prepared data:", DATA_DIR)
'''

training = '''
import gc
import platform
from contextlib import nullcontext

if not PREPARE_ONLY:
    if "model" in globals():
        del model
    gc.collect(); torch.cuda.empty_cache()
    software = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "huggingface_hub", "safetensors", "tokenizers")}
    model, model_report = recipe.load_model(CONFIG)
    source_environment = json.loads((Path(SUPERVISED_RUN) / "ENVIRONMENT.json").read_text())
    if "precision" in source_environment.get("model", {}):
        assert source_environment["model"]["precision"] == model_report["precision"], "Use the supervised precision for this paired experiment"
    recipe.restore(SOURCE_CHECKPOINT, model, SOURCE["identity"])
    identity = recipe.digest(dict(stage=decision_rl.VERSION, config=CONFIG, rl_config=RL_CONFIG,
                                  source=SOURCE, code=source_hashes, data=manifest["hashes"],
                                  software=software, precision=model_report["precision"],
                                  compute_capability=model_report["compute_capability"]))
    study_identity = recipe.digest(dict(config=CONFIG, rl_config={key: value for key, value in RL_CONFIG.items() if key != "method"},
                                       source=SOURCE, code=source_hashes, data=manifest["hashes"], software=software,
                                       precision=model_report["precision"], compute_capability=model_report["compute_capability"]))
    RUN = OUTPUT_ROOT / ("decision_rl_" + identity[:16])
    RUN.mkdir(parents=True, exist_ok=True)
    recipe.immutable_json(RUN / "CONFIG.json", dict(identity=identity, config=CONFIG, rl_config=RL_CONFIG,
                          posttraining_study_identity=study_identity,
                          software=software, source_sha=source_sha, source_hashes=source_hashes,
                          data_manifest_sha256=recipe.digest(manifest), data_directory=str(DATA_DIR)))
    recipe.atomic_json(RUN / "ENVIRONMENT.json", dict(python=platform.python_version(), software=software, model=model_report))
    tracking = (wandb_tracking.track_run(dict(name=RL_CONFIG["method"], config={**CONFIG, "decision_rl": RL_CONFIG, "supervised_parent": SOURCE}),
                    RUN, identity, "openkind_decision_rl_" + study_identity[:16], project=WANDB_PROJECT,
                    entity=WANDB_ENTITY, mode=WANDB_MODE, log_every=WANDB_LOG_EVERY)
                if USE_WANDB else nullcontext(None))
    with tracking as logger:
        selection = decision_rl.fit(model, data, CONFIG, RL_CONFIG, RUN, identity, SOURCE, tokenizer=tokenizer,
                                    on_event=logger.on_event if logger else None)
        selected = next(entry for entry in selection["history"] if entry["step"] == selection["selected_step"])
        if logger:
            logger(selection, dict(selected_step=selected["step"], metrics=selected["metrics"]))
    display(pd.DataFrame([dict(step=entry["step"], nll=entry["metrics"]["macro_nll"],
                         accuracy=entry["metrics"]["macro_accuracy"], retention=entry["retention"]["passed"],
                         exact_record_accuracy=entry["utility"]["exact_record_accuracy"]) for entry in selection["history"]]))
    recipe.atomic_json(RUN / "POSTTRAINING_SUMMARY.json", dict(stage=decision_rl.VERSION, supervised_parent=SOURCE,
                         objective=RL_CONFIG, selected_rl_step=selection["selected_step"],
                         development_utility=selected["utility"], run_directory=str(RUN), data_directory=str(DATA_DIR),
                         acceptance_requested=RUN_ACCEPTANCE, model_promoted=False, native_qualified=False))
    print("Selected RL update:", selection["selected_step"], "(0 retains the supervised checkpoint)")
    print("Run directory:", RUN)
    print("Acceptance remains closed." if not RUN_ACCEPTANCE else "Proceeding to the declared fresh acceptance panel.")
'''

for cell in notebook.cells:
    if cell.cell_type == "code":
        if "EMBEDDED_SOURCE_FILES =" in cell.source:
            continue
        if 'CHOSEN_RECIPE_PATH = ""' in cell.source:
            cell.source = textwrap.dedent(parameters).strip()
        elif "data, manifest = recipe.prepare_data(" in cell.source:
            cell.source = textwrap.dedent(data_cell).strip()
        elif "software = {p:" in cell.source and "sweep_context =" in cell.source:
            cell.source = textwrap.dedent(training).strip()
        elif "if not PREPARE_ONLY" in cell.source and any(text in cell.source for text in
                ("calibrate_and_gate", "candidate_schema =", "export_bundle", "BEST_MODEL_SUMMARY.json", "final_evaluation", "typesafe_benchmark.run_benchmark")):
            cell.source = cell.source.replace("if not PREPARE_ONLY", "if not PREPARE_ONLY and RUN_ACCEPTANCE")
            cell.source = cell.source.replace("(frozen parent)", "(supervised checkpoint)")
            cell.source = cell.source.replace("Selected checkpoint is the frozen parent", "Selected checkpoint is the supervised parent")
    elif cell.source.startswith("## 1."):
        cell.source = """## 1. Stage contract

The warm start is a trained supervised adapter, with a fresh RL optimizer. Step zero means
that supervised checkpoint. The frozen reference binds its logits to the exact admitted
presentations. All arms keep the same source data and development retention rules.

The default fits 50 updates, batch 8 complete units. A record may contain multiple fields,
so this is not the same field exposure or memory budget as 50 single-question updates.
Ordinal credit is attached to numeric semantic levels; `__none__` gets only exact credit.
The proper-score terms are differentiated directly, and no shared detached Brier reward
is added to every sampled action. Record rewards apply only to complete exact-label records.

Compare `supervised`, `expected_utility` and `reinforce` on development with acceptance
disabled. Freeze the method before setting `RUN_ACCEPTANCE=True`; record any earlier
acceptance directories in `PRIOR_EVIDENCE_DATA_DIRS`. A failed gate retains the supervised
checkpoint and is not evidence that RL improved it.
"""
    elif cell.source.startswith("## 4."):
        cell.source = """## 4. Declare the supervised input and RL experiment

Set the full source run and its prepared-data directory. Keep these on Drive across sessions.
An export ZIP alone omits rejected checkpoints and the prepared training datasets.
Changing method, rewards, budget, data, source checkpoint or software creates a new identity.
"""
    elif cell.source.startswith("## 6."):
        cell.source = """## 6. Inherit datasets and reserve fresh acceptance groups

Verify the original data manifest against its supervised run. Keep its split seed, public
dataset pins and token cap. Generated reasoning contributes complete multi-field requests.
Exclude every earlier calibration, gate and test group supplied here, before admission.
If fresh groups are insufficient, stop and declare a new data plan rather than reuse them.
"""
    elif cell.source.startswith("## 7."):
        cell.source = """## 7. Optimize discrete decisions

Restore the verified supervised checkpoint and cache its reference logits once. Each update
processes complete requests; committed checkpoints include the optimizer, scheduler and RNG
state. Reruns restore this exact experiment. W&B logs rewards, KL, proper scores and speed.
Development selects by NLL subject to retention, never by training reward alone.
"""
    elif cell.source.startswith("## 8."):
        cell.source += "\n\nRuns only with `RUN_ACCEPTANCE=True`. The parent here is the supervised checkpoint."
    elif cell.source.startswith("## 9."):
        cell.source += "\n\nRequires acceptance. The locked export includes the supervised lineage, RL plan and trainer."
    elif cell.source.startswith("## 10."):
        cell.source = """## 10. Frozen acceptance summary

Only written after acceptance and export. The development-only stage always writes
`POSTTRAINING_SUMMARY.json`. The existing release notebook accepts a frozen trained export;
a stage that retains RL update zero does not qualify as a newly improved RL model.
"""
for index, cell in enumerate(notebook.cells):
    cell.id = f"local-decisions-rl-{index:02d}"
nbformat.validate(notebook)
nbformat.write(notebook, TARGET)
print(TARGET)
