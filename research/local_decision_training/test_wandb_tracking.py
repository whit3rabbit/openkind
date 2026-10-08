"""Offline sweep tracking regressions, with no W&B login, model or dataset downloads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import train as recipe
import wandb_tracking


class FakeRun:
    def __init__(self, config):
        self.config = config
        self.summary = {}
        self.logs = []
        self.exit_codes = []
        self.artifacts = []

    def log(self, values, step=None):
        self.logs.append((values.get("optimizer_step", step), values))

    def define_metric(self, *args, **kwargs):
        pass

    def finish(self, exit_code=0):
        self.exit_codes.append(exit_code)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


class FakeArtifact:
    def __init__(self, name, type):
        self.name, self.type, self.files = name, type, []

    def add_file(self, path, name):
        self.files.append((path, name))


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self):
        self.runs = []

    def init(self, **config):
        run = FakeRun(config)
        if config.get("resume"):
            previous = next((r for r in self.runs if r.config.get("id") == config["id"]), None)
            if previous is not None:
                run.summary = dict(previous.summary)
        self.runs.append(run)
        return run


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(recipe.DEFAULT_CONFIG, max_steps=2)
        self.data = {role: [] for role in recipe.ROLES}
        self.data["train"] = [{"source": "tiny"}]
        self.factory = mock.Mock(side_effect=lambda config: (object(), dict(precision="fp32-cpu", compute_capability=[])))
        self.tracker = FakeWandb()
        self.parent_only = False

    def fit(self, model, data, config, run, identity, **kwargs):
        self.assertEqual(set(data), {"train", "development"})
        def metrics(nll):
            return dict(macro_nll=nll, macro_accuracy=.75,
                        by_source={"tiny": dict(nll=nll, accuracy=.75, brier=.2)})
        parent = dict(step=0, metrics=metrics(1.), retention=dict(passed=True, reasons=[]))
        candidate = dict(step=config["max_steps"], metrics=metrics(.5 + config["learning_rate"]),
                         retention=dict(passed=not self.parent_only, reasons=[]))
        selection = dict(identity=identity, selected_step=0 if self.parent_only else candidate["step"], history=[parent, candidate])
        run = Path(run)
        for step, history in ((0, [parent]), (candidate["step"], selection["history"])):
            folder = run / "checkpoints" / f"step_{step:06d}"
            for name in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors", "adapter/README.md", "resume.pt"):
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"offline fixture")
            recipe.atomic_json(folder / "COMMIT.json", dict(step=step, history=history, identity=identity,
                files={str(path.relative_to(folder)): recipe.file_digest(path) for path in folder.rglob("*") if path.is_file()}))
        recipe.atomic_json(run / "SELECTION.json", selection)
        recipe.atomic_json(run / "BASELINE_DEVELOPMENT.json", parent["metrics"])
        if kwargs.get("on_event"):
            kwargs["on_event"]("development", parent)
            kwargs["on_event"]("update", dict(step=candidate["step"], train_loss=.5, grad_norm=1., learning_rate=1e-5, seconds=1., tokens=100))
            kwargs["on_event"]("development", candidate)
        return selection

    def tracked(self, root, dimension, arms=None):
        return wandb_tracking.run_sweep_with_wandb(self.factory, self.data, self.config, root, {}, dimension,
                                                  arms, wandb_module=self.tracker)

    def test_complete_default_studies_match_untracked_selection(self):
        with mock.patch.object(recipe, "gradient_preflight", return_value={}), \
             mock.patch.object(recipe, "fit", side_effect=self.fit):
            for dimension in ("learning_rate", "rank", "loss"):
                with self.subTest(dimension=dimension), tempfile.TemporaryDirectory() as root:
                    self.tracker = FakeWandb()
                    self.factory.reset_mock()
                    config_before = copy.deepcopy(self.config)
                    result = self.tracked(root, dimension)
                    if dimension == "loss":
                        plain = recipe.run_loss_sweep(self.factory, self.data, self.config, root, {})
                    else:
                        plain = recipe.run_parameter_sweep(self.factory, self.data, self.config, root, {}, dimension)
                    self.assertEqual(result, plain)
                    self.assertEqual(self.config, config_before)
                    count = 6 if dimension == "loss" else 3
                    self.assertEqual(len(result["results"]), count)
                    self.assertEqual(len(self.tracker.runs), count)
                    self.assertEqual(self.factory.call_count, count)
                    self.assertTrue((Path(result["selected_run"]).parent / "SWEEP_RESULT.json").exists())
                    for run, arm in zip(self.tracker.runs, result["results"]):
                        self.assertEqual(run.config["name"], arm["name"])
                        self.assertEqual(run.config["config"], arm["config"])
                        self.assertEqual(run.config["group"], "openkind_sweep_" + result["sweep_identity"][:16])
                        self.assertEqual(run.config["reinit"], "finish_previous")
                        self.assertEqual(run.config["mode"], "offline")
                        self.assertEqual([step for step, _ in run.logs], [0, 2, 2])
                        self.assertEqual(run.logs[1][1]["train/loss"], .5)
                        self.assertEqual(run.summary["selected_step"], arm["selected_step"])
                        self.assertEqual(run.exit_codes, [0])

    def test_invalid_studies_fail_before_model_or_tracking_initialization(self):
        with tempfile.TemporaryDirectory() as root:
            for dimension in ("learning_rate", "rank", "loss"):
                defaults = recipe.LOSS_SWEEP_ARMS if dimension == "loss" else recipe.PARAMETER_SWEEP_ARMS[dimension]
                for arms in ([], defaults[:1], list(defaults) * 7):
                    with self.subTest(dimension=dimension, count=len(arms)):
                        with self.assertRaisesRegex(AssertionError, "2 to 6 arms"):
                            self.tracked(root, dimension, arms)
            self.factory.assert_not_called()
            self.assertEqual(self.tracker.runs, [])

    def test_training_failure_finishes_tracking_and_stops_study(self):
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.object(recipe, "gradient_preflight", return_value={}), \
             mock.patch.object(recipe, "fit", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                self.tracked(root, "learning_rate")
            self.factory.assert_called_once()
            self.assertEqual(len(self.tracker.runs), 1)
            self.assertEqual(self.tracker.runs[0].exit_codes, [1])
            self.assertFalse(list(Path(root).glob("sweep_*/SWEEP_RESULT.json")))

    def test_parent_selection_remains_eligible_and_is_logged(self):
        self.parent_only = True
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.object(recipe, "gradient_preflight", return_value={}), \
             mock.patch.object(recipe, "fit", side_effect=self.fit):
            result = self.tracked(root, "learning_rate")
        self.assertEqual(result["selected_arm"], "lr_1e_05")
        self.assertTrue(all(arm["selected_step"] == 0 for arm in result["results"]))
        for run in self.tracker.runs:
            self.assertTrue(run.summary["retained_parent"])
            self.assertFalse(run.logs[-1][1]["development/passes_retention"])

    def test_completed_sweep_rejects_corrupt_checkpoint_without_loading_or_logging(self):
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.object(recipe, "gradient_preflight", return_value={}), \
             mock.patch.object(recipe, "fit", side_effect=self.fit):
            result = self.tracked(root, "learning_rate")
            path = Path(result["results"][1]["run"]) / "checkpoints/step_000002/adapter/adapter_model.safetensors"
            path.write_bytes(b"corrupt")
            self.factory.reset_mock()
            self.tracker = FakeWandb()
            with self.assertRaisesRegex(AssertionError, "Corrupt checkpoint"):
                self.tracked(root, "learning_rate")
            self.factory.assert_not_called()
            self.assertEqual(self.tracker.runs, [])

    def test_online_resume_skips_saved_events_and_uses_optimizer_axis(self):
        arm = dict(name="control", config=self.config)
        entry = dict(step=2, metrics=dict(macro_nll=.5, macro_accuracy=.75, by_source={}), retention=dict(passed=True))
        with tempfile.TemporaryDirectory() as root:
            for _ in range(2):
                with wandb_tracking.track_run(arm, root, "stable-identity", "group", mode="online", wandb_module=self.tracker) as logger:
                    logger.on_event("development", entry)
                    logger.on_event("update", dict(step=2, train_loss=.5, grad_norm=1., learning_rate=1e-5, seconds=1., tokens=100))
        first, second = self.tracker.runs
        self.assertEqual(first.config["id"], second.config["id"])
        self.assertEqual(second.config["resume"], "allow")
        self.assertEqual(len(first.logs), 2)
        self.assertEqual(second.logs, [])
        self.assertTrue(all(values["optimizer_step"] == 2 for _, values in first.logs))

    def test_qualification_records_actual_parent_fallback_and_optional_reports(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "lr_1e_05"
            recipe.atomic_json(run / "CONFIG.json", dict(config=self.config, sweep_identity="study"))
            recipe.atomic_json(run / "GATE_RESULT.json", dict(decision="frozen_parent_retained", confirmed_retention=False))
            for name in ("BEST_MODEL_SUMMARY.json", "PERFORMANCE.json", "EXPORT_LOCK.json", "GATE_ROWS.json"):
                recipe.atomic_json(run / name, {})
            with mock.patch.object(recipe, "verify_export", return_value=({}, dict(exported_step=0, temperature=1.), {})) as verify:
                wandb_tracking.log_qualification(run, "identity", wandb_module=self.tracker)
                wandb_tracking.log_qualification(run, "identity", log_reports=True, wandb_module=self.tracker)
            verify.assert_called_with(run / "export", "identity")
            first, second = self.tracker.runs
            self.assertEqual(first.config["group"], "openkind_sweep_study")
            self.assertEqual(first.config["name"], "lr_1e_05")
            self.assertEqual(first.summary["exported_step"], 0)
            self.assertFalse(first.summary["experimental_candidate_exported"])
            self.assertFalse(first.summary["model_promoted"])
            self.assertEqual(first.artifacts, [])
            self.assertEqual({name for _, name in second.artifacts[0].files},
                             {"BEST_MODEL_SUMMARY.json", "GATE_RESULT.json", "PERFORMANCE.json", "EXPORT_LOCK.json"})


if __name__ == "__main__":
    unittest.main()
