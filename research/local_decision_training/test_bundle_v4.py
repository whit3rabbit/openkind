"""Offline bundle provenance and replay checks on a randomly initialized tiny hybrid."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import benchmark
import train as recipe
from test_train import ReloadableTinyTokenizer, benchmark_fixture, encoded_rows, tiny_model


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        cls.identity = "offline-v4-bundle"
        cls.config = {**recipe.DEFAULT_CONFIG, "rank": 2, "alpha": 4, "max_steps": 1,
                      "accumulation": 1, "evaluate_every": 1, "checkpoint_every": 1,
                      "learning_rate": 1e-3}
        cls.data = {role: copy.deepcopy(encoded_rows()) for role in recipe.ROLES}
        cls.manifest = dict(sources={}, hashes={role: recipe.digest(rows) for role, rows in cls.data.items()})
        for name in ("CONFIG.json", "ENVIRONMENT.json"):
            recipe.atomic_json(cls.root / name, {})
        model = tiny_model(cls.config)
        cls.selection = recipe.fit(model, cls.data, cls.config, cls.root, cls.identity)
        cls.gate = recipe.calibrate_and_gate(model, cls.data, cls.config, cls.root, cls.identity, cls.selection)
        recipe.export_bundle(model, ReloadableTinyTokenizer(), cls.config, cls.manifest,
                             cls.root, cls.identity, cls.selection, cls.gate,
                             tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        shutil.copytree(self.root, self.run)
        self.export = self.run / "export"
        self.model = tiny_model(self.config)

    def load(self):
        return recipe.load_frozen_bundle(self.model, self.config, self.run, self.identity,
                                         tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)

    def final(self, data=None, gate=None):
        return recipe.final_evaluation(self.model, self.data if data is None else data, mock.Mock(),
                                       self.config, self.run, self.identity, self.gate if gate is None else gate,
                                       tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)

    def test_frozen_source_and_serialized_replay_execute(self):
        with mock.patch.dict("sys.modules", {"train": None}):
            bundle = self.load()
        self.assertEqual(Path(bundle.recipe.__file__), (self.export / "train.py").resolve())
        self.assertEqual(Path(bundle.benchmark.__file__), (self.export / "benchmark.py").resolve())
        self.assertIs(bundle.benchmark.recipe, bundle.recipe)
        self.assertIsNot(bundle.recipe.logits, recipe.logits)
        self.assertIsNot(bundle.tokenizer, ReloadableTinyTokenizer())
        replay = json.loads((self.export / "REPLAY.json").read_text())
        self.assertFalse(replay["native_qualification"])
        self.assertFalse(replay["source_examples_included"])
        self.assertEqual(len(replay["cases"]), 3)
        for case in replay["cases"]:
            self.assertNotIn("failure", case)
            self.assertNotIn("target", case)
            self.assertEqual(bundle.tokenizer.encode(case["rendered_input"]), case["input_ids"])
            self.assertEqual(recipe.probabilities(case["logits"], case["temperature"]), case["probabilities"])
            question = case["question"]
            actual = bundle.recipe.decide(self.model, bundle.tokenizer, case["state"], [question], bundle.contract)
            self.assertEqual(actual[question["id"]], case["typed_answer"])
            self.assertEqual(list(case["typed_answer"]["probabilities"]), [entry["key"] for entry in case["mapping"]])
            self.assertEqual("confidence" in case["typed_answer"], question["kind"] != "noul")
        for case in replay["failures"]:
            with self.assertRaises(AssertionError):
                bundle.recipe.decide(self.model, bundle.tokenizer, case["state"], case["questions"],
                                     {**bundle.contract, **case["contract_overrides"]})

    def test_corrupt_assets_rejected_before_final_access(self):
        for name in ("train.py", "benchmark.py", "tokenizer/tokenizer.json", "adapter/adapter_model.safetensors",
                     "parent_adapter/adapter_model.safetensors", "DECISION_CONTRACT.json", "GATE_RESULT.json",
                     "CALIBRATION_ROWS.json", "REPLAY.json"):
            with self.subTest(name=name):
                path = self.export / name
                saved = path.read_bytes()
                try:
                    path.write_bytes(saved + b" ")
                    with self.assertRaisesRegex(AssertionError, "Frozen export changed"):
                        self.final()
                    self.assertFalse((self.run / "TEST_OPENED.json").exists())
                finally:
                    path.write_bytes(saved)

    def test_portable_archive_retains_manifest_lock_without_checkpoints(self):
        import zipfile
        path = recipe.archive_export(self.run, self.identity)
        destination = Path(self.temp.name) / "unpacked"
        with zipfile.ZipFile(path) as archive:
            self.assertIn("EXPORT_LOCK.json", archive.namelist())
            self.assertIn("export/EXPORT.json", archive.namelist())
            self.assertFalse(any("checkpoints" in name or name.endswith("resume.pt") for name in archive.namelist()))
            archive.extractall(destination)
        metadata, contract, config = recipe.verify_export(destination / "export", self.identity)
        self.assertEqual(metadata["identity"], self.identity)
        bundle = recipe.load_frozen_bundle(self.model, config, destination, self.identity,
                                          tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)
        self.assertEqual(bundle.contract, contract)

    def test_rehashed_manifest_requires_original_run_lock(self):
        ((self.export / "train.py").resolve()).write_text("raise RuntimeError('not executed')")
        metadata = json.loads((self.export / "EXPORT.json").read_text())
        metadata["files"]["train.py"] = recipe.file_digest((self.export / "train.py").resolve())
        recipe.atomic_json(self.export / "EXPORT.json", metadata)
        with self.assertRaisesRegex(AssertionError, "manifest changed"):
            self.load()

    def test_extra_assets_and_symlinks_are_rejected(self):
        path = self.export / "extra.py"
        path.write_text("pass")
        with self.assertRaisesRegex(AssertionError, "inventory changed"):
            self.load()
        path.unlink()
        path.symlink_to("train.py")
        with self.assertRaisesRegex(AssertionError, "Symlinks"):
            self.load()

    def test_caller_code_and_configuration_cannot_replace_export(self):
        changed = self.run / "changed.py"
        changed.write_text("raise RuntimeError('not executed')")
        with mock.patch.object(recipe, "__file__", str(changed)):
            with self.assertRaisesRegex(AssertionError, "Trainer implementation"):
                self.load()
        with mock.patch.object(benchmark, "__file__", str(changed)):
            with self.assertRaisesRegex(AssertionError, "Benchmark implementation"):
                benchmark.run_benchmark(self.model, None, self.config, self.run, self.identity,
                                        tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)
        with self.assertRaisesRegex(AssertionError, "Configuration differs"):
            recipe.load_frozen_bundle(self.model, {**self.config, "max_length": 4096}, self.run, self.identity)
        with self.assertRaisesRegex(AssertionError, "Gate differs"):
            self.final(gate={**self.gate, "deployed_temperature": 2.0})
        self.model.peft_config[self.model.active_adapter].lora_alpha += 1
        with self.assertRaisesRegex(AssertionError, "adapter configuration"):
            self.load()

    def test_tokenizer_requires_explicit_offline_loader(self):
        with self.assertRaisesRegex(AssertionError, "explicit tokenizer loader"):
            recipe.load_frozen_bundle(self.model, self.config, self.run, self.identity)
        class WrongTokenizer(ReloadableTinyTokenizer):
            def encode(self, text, **kwargs):
                return [x + 1 for x in super().encode(text, **kwargs)]
        with self.assertRaisesRegex(AssertionError, "code mapping changed"):
            recipe.load_frozen_bundle(self.model, self.config, self.run, self.identity,
                                     tokenizer_loader=lambda path: WrongTokenizer())

    def test_reserved_tokenization_rechecks_saved_template_and_target_alignment(self):
        tokenizer = ReloadableTinyTokenizer()
        raw = recipe.record("fixture", "rendering", "Known facts", "Assess",
                            [("false", "False"), ("true", "True")], "true", "noul", target=[.2, .8])
        row = recipe.encode(raw, tokenizer, 17, permutation=[1, 0])
        recipe.verify_reserved_encoding([row], tokenizer)
        class ChangedTemplate(ReloadableTinyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return "changed " + super().apply_chat_template(messages, **kwargs)
        with self.assertRaisesRegex(AssertionError, "Frozen tokenizer/rendering differs"):
            recipe.verify_reserved_encoding([row], ChangedTemplate())
        with self.assertRaisesRegex(AssertionError, "original render contract"):
            recipe.verify_reserved_encoding(encoded_rows(), tokenizer)
        recipe.verify_reserved_encoding(encoded_rows(), tokenizer, offline_fixture=True)

    def test_reserved_rows_bound_before_even_cached_final(self):
        recipe.atomic_json(self.run / "FINAL_REPORT.json", {"cached": True})
        altered = copy.deepcopy(self.data)
        altered["test"][0]["target"] = [1., 0.]
        with self.assertRaisesRegex(AssertionError, "Prepared rows differ"):
            self.final(data=altered)
        self.assertFalse((self.run / "TEST_OPENED.json").exists())
        with mock.patch.object(recipe, "predict", side_effect=AssertionError("live prediction must not execute")):
            self.assertEqual(self.final(), {"cached": True})

    def test_fresh_final_executes_saved_implementation_on_local_sources(self):
        def source_loader(spec):
            if spec["repo"].endswith("paws"):
                rows = [dict(sentence1=f"A duck number {i} swims.", sentence2=f"Duck {i} is swimming.", label=1)
                        for i in range(16)]
            else:
                rows = [dict(question=f"Which is a planet? Fixture {i}.", correct_answer="Earth",
                             distractor1="Oak", distractor2="Slate", distractor3="Cloud") for i in range(16)]
            return rows, {**spec, "offline_fixture": True}
        with mock.patch.object(recipe, "predict", side_effect=AssertionError("live final implementation executed")):
            result = recipe.final_evaluation(self.model, self.data, mock.Mock(), self.config,
                                            self.run, self.identity, self.gate,
                                            tokenizer_loader=ReloadableTinyTokenizer.from_pretrained,
                                            source_loader=source_loader)
        self.assertTrue(result["offline_fixture"])
        self.assertFalse(result["final_used_for_selection"])
        self.assertFalse(result["model_promoted"])
        self.assertIn("paired_group_comparison", result)
        self.assertTrue({"tiny", "paws", "sciq", "rules_unseen_composition"} <= set(result["candidate"]["by_source"]))
        with mock.patch.object(recipe, "predict", side_effect=AssertionError("cached final recomputed")):
            self.assertEqual(self.final(), result)

    def test_benchmark_uses_saved_code_tokenizer_and_adapters(self):
        # Local checkpoint mutation and live monkeypatches cannot alter the frozen candidate or parent.
        for path in (self.run / "checkpoints").glob("*/adapter/adapter_model.safetensors"):
            path.write_bytes(b"changed after export")
        source_loader = mock.Mock(return_value=([benchmark_fixture()], {"offline_fixture": True}))
        with mock.patch.object(recipe, "logits", side_effect=AssertionError("live inference executed")), \
             mock.patch.object(benchmark, "evaluate", side_effect=AssertionError("live benchmark executed")):
            result = benchmark.run_benchmark(self.model, mock.Mock(), self.config, self.run, self.identity,
                                            tokenizer_loader=ReloadableTinyTokenizer.from_pretrained,
                                            source_loader=source_loader)
        self.assertEqual(source_loader.call_count, 5)
        self.assertTrue(all(value["by_kind"]["noul"]["answered"] == 1 for value in result["candidate"].values()))
        self.assertEqual(set(result["candidate"]), set(result["parent"]))
        self.assertFalse(result["model_promoted"])

    def test_retained_parent_exports_step_zero_and_unit_temperature(self):
        shutil.rmtree(self.export)
        (self.run / "EXPORT_LOCK.json").unlink()
        gate = {**self.gate, "decision": "retain_parent", "deployed_temperature": 1.7}
        recipe.atomic_json(self.run / "GATE_RESULT.json", gate)
        recipe.export_bundle(self.model, ReloadableTinyTokenizer(), self.config, self.manifest,
                             self.run, self.identity, self.selection, gate,
                             tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)
        bundle = self.load()
        self.assertEqual(bundle.contract["temperature"], 1.0)
        self.assertEqual(bundle.metadata["exported_step"], 0)
        self.assertEqual(recipe.file_digest(self.export / "adapter/adapter_model.safetensors"),
                         recipe.file_digest(self.export / "parent_adapter/adapter_model.safetensors"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
