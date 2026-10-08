"""Validate generated notebooks and execute their T4 sweep cell with offline fixtures."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import nbformat

import train as recipe
import test_wandb_tracking

HERE = Path(__file__).resolve().parent


class NotebookTests(unittest.TestCase):
    def test_decision_rl_notebook_warm_start_tracking_resume_and_acceptance_boundary(self):
        import copy
        import pandas as pd
        import torch
        import transformers
        import decision_rl
        import wandb_tracking
        from test_decision_rl import WarmStartTests, record_fields
        from test_train import ReloadableTinyTokenizer
        WarmStartTests.setUpClass()
        self.addCleanup(WarmStartTests.tearDownClass)
        fixture = WarmStartTests()
        notebook = nbformat.read(HERE / "local_decision_posttraining.ipynb", as_version=4)
        parameters = next(c.source for c in notebook.cells if "SUPERVISED_RUN =" in c.source)
        preparation = next(c.source for c in notebook.cells if "original_data, original_manifest" in c.source)
        training = next(c.source for c in notebook.cells if "selection = decision_rl.fit" in c.source)
        after_training = [c.source for c in notebook.cells if c.cell_type == "code" and "if not PREPARE_ONLY and RUN_ACCEPTANCE" in c.source]
        for acceptance in (False, True):
            with self.subTest(acceptance=acceptance), tempfile.TemporaryDirectory() as root:
                tracker = test_wandb_tracking.FakeWandb()
                settings = parameters.replace('SUPERVISED_RUN = ""', f"SUPERVISED_RUN = {str(fixture.source_run)!r}")
                settings = settings.replace('SUPERVISED_DATA_DIR = ""', f"SUPERVISED_DATA_DIR = {str(fixture.data_dir)!r}")
                settings = settings.replace('SUPERVISED_STEP = None', 'SUPERVISED_STEP = 2')
                settings = settings.replace('max_steps=50, accumulation=8, learning_rate=5e-6', 'max_steps=2, accumulation=1, learning_rate=1e-3')
                settings = settings.replace('data_intervention="reasoning")', 'data_intervention="reasoning", schema_diagnostics=False)')
                settings = settings.replace('RUN_ACCEPTANCE = False', f'RUN_ACCEPTANCE = {acceptance!r}')
                settings = settings.replace('USE_WANDB = False', 'USE_WANDB = True')
                namespace = dict(WORK=HERE, OUTPUT_ROOT=Path(root), Path=Path, torch=torch, importlib=__import__("importlib"),
                                 pd=pd, display=lambda value: None,
                                 EMBEDDED_SOURCE_FILES=("train.py", "benchmark.py", "wandb_tracking.py", "decision_rl.py"))
                with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True), contextlib.redirect_stdout(io.StringIO()):
                    exec(settings, namespace)
                fresh = copy.deepcopy(fixture.data)
                fresh["development"] = record_fields()
                for role, rows in fresh.items():
                    for row in rows:
                        row["group"] = f"fresh-{role}-" + row["group"]
                manifest = dict(schema=recipe.VERSION, sources={}, config=namespace["CONFIG"],
                                hashes={role: recipe.digest(rows) for role, rows in fresh.items()})
                with mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=ReloadableTinyTokenizer()), \
                     mock.patch.object(recipe, "prepare_data", return_value=(fresh, manifest)), contextlib.redirect_stdout(io.StringIO()):
                    exec(preparation, namespace)
                export_function = recipe.export_bundle
                def export_fixture(*args, **kwargs):
                    return export_function(*args, **kwargs, tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)
                with mock.patch.object(recipe, "load_model", side_effect=lambda config: (fixture.model(), dict(precision="fp32-cpu", compute_capability=[]))), \
                     mock.patch("importlib.metadata.version", return_value="offline-fixture"), \
                     mock.patch.object(recipe, "export_bundle", side_effect=export_fixture), \
                     mock.patch.object(recipe, "calibrate_and_gate", wraps=recipe.calibrate_and_gate) as gate, \
                     mock.patch.dict("sys.modules", wandb=tracker), contextlib.redirect_stdout(io.StringIO()):
                    exec(training, namespace)
                    selected = copy.deepcopy(namespace["selection"])
                    exec(training, namespace)
                    self.assertEqual(namespace["selection"], selected)
                    if not acceptance:
                        namespace.update(RUN_FINAL_TEST=True, RUN_TYPESAFE_BENCHMARK=True)
                    for source in after_training:
                        exec(source, namespace)
                run = namespace["RUN"]
                self.assertEqual(gate.call_count, int(acceptance))
                self.assertEqual((run / "export").exists(), acceptance)
                self.assertFalse((run / "TEST_OPENED.json").exists())
                self.assertTrue((run / "POSTTRAINING_SUMMARY.json").exists())
                self.assertTrue(all(item.exit_codes == [0] for item in tracker.runs))
                self.assertTrue(all(item.config["name"] == "reinforce" for item in tracker.runs))
                self.assertEqual(len({item.config["group"] for item in tracker.runs}), 1)
                logs = [values for tracked in tracker.runs for _, values in tracked.logs]
                self.assertTrue(any("decision_rl/reference_kl" in values for values in logs))
                self.assertTrue(any("development/exact_record_accuracy" in values for values in logs))
                plan = json.loads((run / "POSTTRAINING_PLAN.json").read_text())
                self.assertEqual(plan["source"]["adapter_sha256"], fixture.source["adapter_sha256"])

    def test_schema_cells_reuse_only_an_identical_parent_and_label_the_evaluated_step(self):
        import pandas as pd
        for target in (HERE.parent / "35_local_decision_training.ipynb",
                       HERE / "local_decision_training_t4.ipynb", HERE / "local_decision_finetuning.ipynb"):
            notebook = nbformat.read(target, as_version=4)
            schema_cell = next(c.source for c in notebook.cells if c.source.startswith("if not PREPARE_ONLY and CONFIG")
                               and "candidate_schema =" in c.source)
            gate_cell = next(c.source for c in notebook.cells if c.source.startswith("if not PREPARE_ONLY:")
                             and "gate = recipe.calibrate_and_gate(" in c.source)
            for step, temperature, expected_calls in ((0, 1., 1), (0, 1.5, 2), (100, 1., 2)):
                with self.subTest(notebook=target.name, step=step, temperature=temperature), tempfile.TemporaryDirectory() as root:
                    metrics = dict(n=64, accuracy=.75, nll=.5, brier=.2, ece=.1, none_recall=None, false_none_rate=None)
                    gate = dict(candidate_raw=dict(by_source=dict(tiny=metrics)), decision="retain_parent",
                                deployed_temperature=temperature, calibration_accepted=False, retention=dict(passed=False))
                    namespace = dict(PREPARE_ONLY=False, CONFIG=recipe.DEFAULT_CONFIG, RUN=Path(root), identity="fixture",
                                     recipe=recipe, selection=dict(selected_step=step), model=object(), tokenizer=None,
                                     data=dict(gate=[]), pd=pd, display=lambda value: None, json=json)
                    output = io.StringIO()
                    with mock.patch.object(recipe, "calibrate_and_gate", return_value=gate), \
                         mock.patch.object(recipe, "restore"), \
                         mock.patch.object(recipe, "schema_diagnostics", return_value=dict(by_transform_and_source={}, rows=[])) as diagnose, \
                         contextlib.redirect_stdout(output):
                        exec(gate_cell, namespace)
                        exec(schema_cell, namespace)
                    self.assertEqual(diagnose.call_count, expected_calls)
                    self.assertIn(f"Evaluated checkpoint: {step}", output.getvalue())
                    self.assertIn("(frozen parent)" if step == 0 else "(trained checkpoint)", output.getvalue())
                    self.assertEqual("reusing the identical schema diagnostics" in output.getvalue(), expected_calls == 1)

    def test_summary_distinguishes_rejected_development_gain_and_actual_export_temperature(self):
        for target in (HERE / "local_decision_training_t4.ipynb", HERE / "local_decision_finetuning.ipynb"):
            notebook = nbformat.read(target, as_version=4)
            source = next(c.source for c in notebook.cells if c.source.startswith("if not PREPARE_ONLY:")
                          and "BEST_MODEL_SUMMARY.json" in c.source)
            for selected_step, decision, deployed_temperature in ((0, "retain_parent", 1.),
                                                                 (100, "retain_parent", 1.),
                                                                 (100, "experimental_candidate_passed", 1.5)):
                with self.subTest(notebook=target.name, selected_step=selected_step, decision=decision), tempfile.TemporaryDirectory() as root:
                    run = Path(root)
                    export = run / "export"
                    recipe.atomic_json(export / "DECISION_CONTRACT.json", dict(temperature=deployed_temperature))
                    history = [dict(step=0, metrics=dict(macro_nll=1.), retention=dict(passed=True, reasons=[])),
                               dict(step=100, metrics=dict(macro_nll=.7), retention=dict(passed=selected_step == 100, reasons=["source regression"] if selected_step == 0 else [])),
                               dict(step=150, metrics=dict(macro_nll=.5), retention=dict(passed=False, reasons=["recall regression"]))]
                    namespace = dict(PREPARE_ONLY=False, recipe=recipe, selection=dict(selected_step=selected_step, history=history),
                                     gate=dict(decision=decision, fitted_temperature=1.5, deployed_temperature=1.5),
                                     SWEEP_MODE="none", sweep=None, RUN=run, export=export, archive=run / "export.zip",
                                     USE_WANDB=False, json=json)
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(source, namespace)
                    summary = json.loads((run / "BEST_MODEL_SUMMARY.json").read_text())
                    self.assertEqual(summary["selected_step"], selected_step)
                    self.assertEqual(summary["best_observed_development_step"], 150)
                    self.assertFalse(summary["best_observed_development_passes_retention"])
                    self.assertEqual(summary["development_rejections"][-1], dict(step=150, reasons=["recall regression"]))
                    self.assertEqual(summary["evaluated_temperature"], 1.5)
                    self.assertEqual(summary["deployed_temperature"], deployed_temperature)
                    self.assertEqual(summary["experimental_candidate_exported"], decision == "experimental_candidate_passed")
                    self.assertFalse(summary["model_promoted"])

    def test_training_install_cells_remove_incompatible_torchao_before_peft_dispatch(self):
        for target in (HERE.parent / "35_local_decision_training.ipynb",
                       HERE / "local_decision_training_t4.ipynb", HERE / "local_decision_finetuning.ipynb"):
            notebook = nbformat.read(target, as_version=4)
            setup = next(c.source for c in notebook.cells if c.cell_type == "code" and "requirements = [" in c.source)
            with self.subTest(notebook=target.name), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                script = root / "bootstrap.py"
                # Model packages are already pinned in the test environment. Only pip is replaced;
                # package discovery and PEFT's failing/successful adapter dispatch remain real.
                script.write_text(f'''from pathlib import Path
import importlib, shutil, subprocess, sys
from unittest import mock
site = Path({str(root / "old-site")!r})
(site / "torchao").mkdir(parents=True)
(site / "torchao/__init__.py").write_text("")
(site / "torchao-0.10.0.dist-info").mkdir()
(site / "torchao-0.10.0.dist-info/METADATA").write_text("Metadata-Version: 2.1\\nName: torchao\\nVersion: 0.10.0\\n")
sys.path[:0] = [str(site), {str(HERE)!r}]
from test_train import tiny_model
import train as recipe
config = dict(recipe.DEFAULT_CONFIG, rank=2, alpha=4)
try:
    tiny_model(config)
except ImportError as error:
    assert "incompatible version of torchao" in str(error) and "0.10.0" in str(error), str(error)
else:
    raise AssertionError("The incompatible Colab package was not reproduced")
commands = []
def pip(command):
    commands.append(command)
    if command[3] == "uninstall":
        assert command == [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"]
        shutil.rmtree(site / "torchao")
        shutil.rmtree(site / "torchao-0.10.0.dist-info")
        importlib.invalidate_caches()
        from peft.import_utils import is_torchao_available
        is_torchao_available.cache_clear()
    else:
        assert command[3:5] == ["install", "-q"]
        assert not any(p.startswith("torch==") or p.startswith("torchao") for p in command[5:])
import torch
with mock.patch.object(subprocess, "check_call", side_effect=pip), \\
     mock.patch.object(torch.cuda, "is_available", return_value=True), \\
     mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True), \\
     mock.patch.object(torch.cuda, "get_device_properties", return_value=mock.Mock(total_memory=16 * 1024**3)), \\
     mock.patch.object(torch.cuda, "get_device_name", return_value="offline CUDA boundary"):
    exec({setup!r})
assert len(commands) == 2
assert any(p.requires_grad for p in tiny_model(config).parameters())
print("PASS: incompatible torchao reproduced, setup removed it, and PEFT adapter dispatch succeeded")
''')
                result = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("PEFT adapter dispatch succeeded", result.stdout)

    def test_generated_notebooks_compile_embed_current_sources_and_regenerate_identically(self):
        for builder, target in (("build_notebook.py", HERE.parent / "35_local_decision_training.ipynb"),
                                ("build_t4_notebook.py", HERE / "local_decision_training_t4.ipynb"),
                                ("build_finetuning_notebook.py", HERE / "local_decision_finetuning.ipynb"),
                                ("build_posttraining_notebook.py", HERE / "local_decision_posttraining.ipynb"),
                                ("build_release_notebook.py", HERE / "local_decision_release.ipynb")):
            with self.subTest(notebook=target.name):
                notebook = nbformat.read(target, as_version=4)
                nbformat.validate(notebook)
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type == "code":
                        compile(cell.source, f"{target.name}:cell-{index}", "exec")
                        self.assertIsNone(cell.execution_count)
                        self.assertEqual(cell.outputs, [])
                embedded = next(c.source for c in notebook.cells if "EMBEDDED_SOURCE_FILES =" in c.source)
                with tempfile.TemporaryDirectory() as root:
                    namespace = {"WORK": Path(root)}
                    exec(embedded, namespace)
                    expected = ({"train.py", "benchmark.py", "release.py"} if "release" in target.name
                                else {"train.py", "benchmark.py", "wandb_tracking.py", "release.py", "decision_rl.py", *[p.name for p in HERE.glob("test*.py")]})
                    self.assertEqual(set(namespace["EMBEDDED_SOURCE_FILES"]), expected)
                    for name in expected:
                        self.assertEqual((Path(root) / name).read_bytes(), (HERE / name).read_bytes())
                with mock.patch.object(nbformat, "write") as write, contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(HERE / builder))
                self.assertEqual(nbformat.writes(write.call_args.args[0]), nbformat.writes(notebook))

    def test_t4_training_cell_executes_all_sweep_modes_with_and_without_tracking_and_reruns(self):
        import pandas as pd
        import torch
        notebook = nbformat.read(HERE / "local_decision_training_t4.ipynb", as_version=4)
        parameters = next(c.source for c in notebook.cells if "CONTROL_CONFIG = dict(CONFIG)" in c.source)
        training = next(c.source for c in notebook.cells if "sweep = None" in c.source)
        for dimension in ("learning_rate", "rank", "loss", "none"):
            for tracking in (False, True):
                with self.subTest(dimension=dimension, tracking=tracking), tempfile.TemporaryDirectory() as root:
                    fixture = test_wandb_tracking.TrackingTests()
                    fixture.setUp()
                    namespace = dict(WORK=HERE, OUTPUT_ROOT=Path(root), pd=pd, display=lambda value: None,
                                     data=fixture.data, tokenizer=None, manifest={"hashes": {}},
                                     source_sha="offline-notebook", source_hashes={}, DATA_DIR=Path(root) / "data")
                    settings = parameters.replace('SWEEP_MODE = "learning_rate"', f"SWEEP_MODE = {dimension!r}")
                    settings = settings.replace("USE_WANDB = False", f"USE_WANDB = {tracking!r}")
                    with contextlib.redirect_stdout(io.StringIO()), \
                         mock.patch.dict("sys.modules", wandb=fixture.tracker):
                        exec(settings, namespace)
                    self.assertEqual(namespace["CONTROL_CONFIG"], namespace["CONFIG"])
                    # The artifact cell runs unchanged; fixtures replace only external/expensive boundaries.
                    def fit_and_save(*args, **kwargs):
                        fit_args = list(args)
                        if dimension == "none":
                            self.assertIs(fit_args[1], fixture.data)
                            fit_args[1] = {role: fixture.data[role] for role in ("train", "development")}
                        selection = fixture.fit(*fit_args, **kwargs)
                        recipe.atomic_json(Path(args[3]) / "SELECTION.json", selection)
                        return selection
                    with mock.patch.object(recipe, "load_model", fixture.factory), \
                         mock.patch.object(recipe, "gradient_preflight", return_value={}), \
                         mock.patch.object(recipe, "fit", side_effect=fit_and_save), \
                         mock.patch.object(recipe, "restore"), \
                         mock.patch("importlib.metadata.version", return_value="offline-fixture"), \
                         mock.patch.object(torch.cuda, "get_device_name", return_value="offline-fixture"), \
                         mock.patch.object(torch.cuda, "get_device_capability", return_value=[]), \
                         mock.patch.object(torch.cuda, "get_device_properties", return_value=mock.Mock(total_memory=16 * 1024**3)), \
                         mock.patch.object(torch.cuda, "synchronize"), \
                         mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                         mock.patch.dict("sys.modules", wandb=fixture.tracker), \
                         contextlib.redirect_stdout(io.StringIO()):
                        exec(training, namespace)
                        first = namespace["sweep"]
                        first_identity = namespace["identity"]
                        exec(training, namespace)
                    self.assertEqual(namespace["sweep"], first)
                    self.assertEqual(namespace["identity"], first_identity)
                    if dimension == "none":
                        self.assertIsNone(first)
                        self.assertEqual(namespace["CONFIG"], namespace["CONTROL_CONFIG"])
                        self.assertEqual(len(fixture.tracker.runs), 2 if tracking else 0)
                        continue
                    self.assertEqual(namespace["CONFIG"], first["selected_config"])
                    count = 6 if dimension == "loss" else 3
                    self.assertEqual(len(first["results"]), count)
                    self.assertEqual(len(fixture.tracker.runs), count * 2 if tracking else 0)
                    self.assertTrue(all(run.exit_codes == [0] for run in fixture.tracker.runs))

    def test_fixed_recipe_cell_accepts_selected_config_and_trains_once(self):
        import pandas as pd
        import torch
        notebook = nbformat.read(HERE / "local_decision_finetuning.ipynb", as_version=4)
        parameters = next(c.source for c in notebook.cells if "CHOSEN_RECIPE_PATH =" in c.source)
        training = next(c.source for c in notebook.cells if "sweep = None" in c.source)
        fixture = test_wandb_tracking.TrackingTests()
        fixture.setUp()
        config = dict(fixture.config, max_length=1024)
        with tempfile.TemporaryDirectory() as root:
            chosen = Path(root) / "chosen.json"
            chosen.write_text(json.dumps(dict(selected_config=config)))
            settings = parameters.replace('CHOSEN_RECIPE_PATH = ""', f"CHOSEN_RECIPE_PATH = {str(chosen)!r}")
            settings = settings.replace("USE_WANDB = False", "USE_WANDB = True")
            namespace = dict(WORK=HERE, OUTPUT_ROOT=Path(root), pd=pd, display=lambda value: None,
                             data=fixture.data, tokenizer=None, manifest={"hashes": {}},
                             source_sha="offline-fixed-recipe", source_hashes={}, DATA_DIR=Path(root) / "data")
            def fit_and_save(*args, **kwargs):
                fit_args = list(args)
                fit_args[1] = {role: fixture.data[role] for role in ("train", "development")}
                return fixture.fit(*fit_args, **kwargs)
            with mock.patch.dict("sys.modules", wandb=fixture.tracker), contextlib.redirect_stdout(io.StringIO()):
                exec(settings, namespace)
            with mock.patch.object(recipe, "load_model", fixture.factory), \
                 mock.patch.object(recipe, "gradient_preflight", return_value={}), \
                 mock.patch.object(recipe, "fit", side_effect=fit_and_save), \
                 mock.patch("importlib.metadata.version", return_value="offline-fixture"), \
                 mock.patch.object(torch.cuda, "get_device_name", return_value="offline-fixture"), \
                 mock.patch.object(torch.cuda, "get_device_capability", return_value=[]), \
                 mock.patch.object(torch.cuda, "get_device_properties", return_value=mock.Mock(total_memory=16 * 1024**3)), \
                 mock.patch.object(torch.cuda, "synchronize"), \
                 mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                 mock.patch.dict("sys.modules", wandb=fixture.tracker), contextlib.redirect_stdout(io.StringIO()):
                exec(training, namespace)
            self.assertEqual(namespace["CONTROL_CONFIG"], config)
            self.assertEqual(namespace["CONFIG"], config)
            self.assertIsNone(namespace["sweep"])
            fixture.factory.assert_called_once_with(config)
            self.assertEqual(len(fixture.tracker.runs), 1)

    def test_release_cells_prepare_and_reuse_locked_package_without_torch_or_upload(self):
        from test_release import ReleaseTests
        ReleaseTests.setUpClass()
        self.addCleanup(ReleaseTests.tearDownClass)
        notebook = nbformat.read(HERE / "local_decision_release.ipynb", as_version=4)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            sources = []
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                source = cell.source
                source = source.replace('subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub==1.33.0"])', "pass  # No upload requested; verification uses the standard library.")
                source = source.replace('Path("/content/openkind_release_tools")', f"Path({str(root / 'tools')!r})")
                source = source.replace('"/content/openkind_local_decision_export.zip"', repr(str(ReleaseTests.archive)))
                source = source.replace('Path("/content/openkind_release_input")', f"Path({str(root / 'input')!r})")
                source = source.replace('Path("/content/openkind_adapter_release")', f"Path({str(root / 'stage')!r})")
                sources.append(source)
            script = root / "smoke.py"
            script.write_text("\n".join(sources) + '\nassert "torch" not in sys.modules\nassert UPLOAD_TO_HUB is False\n')
            for _ in range(2):
                result = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Local package verified", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
