"""Offline decision-RL gradients, dataset boundaries and warm-start/resume evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from peft import get_peft_model_state_dict

import decision_rl as rl
import train as recipe
from test_train import encoded_rows, tiny_model, FakeTokenizer, ReloadableTinyTokenizer


def record_fields():
    fields = copy.deepcopy(encoded_rows())
    for index, row in enumerate(fields):
        row.update(role="train", supervision="human", request_id="request", request_size=len(fields),
                   question_id=f"q{index}", state="shared facts", group="shared", id=f"field:{index}")
    return fields


class ObjectiveTests(unittest.TestCase):
    def test_direct_proper_score_and_reference_gradients(self):
        fields = record_fields()
        prediction = torch.tensor([.2, -.3], requires_grad=True)
        fields = [fields[0]]
        config = dict(rl.DEFAULT_RL_CONFIG, method="supervised")
        loss, metrics = rl.objective([prediction], fields, [[-.1, .4]], config)
        probability = prediction.softmax(-1)
        target = torch.tensor(fields[0]["target"])
        expected = -(target * prediction.log_softmax(-1)).sum() + .1 * (probability - target).square().sum()
        expected += .05 * (probability * (prediction.log_softmax(-1) - torch.tensor([-.1, .4]).log_softmax(-1))).sum()
        torch.testing.assert_close(loss, expected)
        actual_gradient = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
        torch.testing.assert_close(actual_gradient, torch.autograd.grad(expected, prediction)[0])
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        self.assertGreater(float(metrics["reference_kl"]), 0)

    def test_sampled_policy_gradient_matches_finite_action_expectation_with_record_reward(self):
        fields = record_fields()
        references = [[0., 0.] for _ in fields]
        config = dict(rl.DEFAULT_RL_CONFIG, samples=64, policy_weight=1., exact_record_weight=.5)
        exact = [torch.tensor([.2, -.1], requires_grad=True) for _ in fields]
        loss, _ = rl.objective(exact, fields, references, {**config, "method": "expected_utility"})
        expected = torch.stack(torch.autograd.grad(loss, exact))
        torch.manual_seed(91)
        gradients = []
        for _ in range(180):
            predictions = [value.detach().clone().requires_grad_() for value in exact]
            loss, _ = rl.objective(predictions, fields, references, config)
            gradients.append(torch.stack(torch.autograd.grad(loss, predictions)))
        torch.testing.assert_close(torch.stack(gradients).mean(0), expected, atol=.025, rtol=0)

    def test_ordinal_credit_survives_code_permutation_and_preserves_none(self):
        row = dict(kind="score", keys=["2", "__none__", "0", "1"], target=[0., 0., 1., 0.])
        utilities = rl.action_utilities(row, rl.DEFAULT_RL_CONFIG).tolist()
        self.assertEqual(dict(zip(row["keys"], utilities)), {"0": 1., "1": .5, "2": 0., "__none__": 0.})
        row["target"] = [0., 1., 0., 0.]
        self.assertEqual(rl.action_utilities(row, rl.DEFAULT_RL_CONFIG).tolist(), [0., 1., 0., 0.])

    def test_requests_stay_complete_and_soft_targets_cannot_claim_exact_records(self):
        fields = record_fields()
        self.assertEqual(len(rl.training_units(fields)), 1)
        with self.assertRaisesRegex(AssertionError, "Incomplete"):
            rl.training_units(fields[:1])
        fields[0]["target"] = [.6, .4]
        with self.assertRaisesRegex(AssertionError, "exact field"):
            rl.training_units(fields)
        fields[0]["benchmark_only"] = True
        with self.assertRaisesRegex(AssertionError, "Benchmark"):
            rl.training_units(fields)
        with self.assertRaises(AssertionError):
            rl.validate_config({**rl.DEFAULT_RL_CONFIG, "samples": 1})

    def test_atomic_counterfactuals_stay_whole_without_receiving_record_rewards(self):
        fields = record_fields()
        for index, row in enumerate(fields):
            row.update(request_id=f"single-{index}", request_size=1, atomic_unit="pair", atomic_size=len(fields))
        units = rl.training_units(fields)
        self.assertEqual(len(units), 1)
        self.assertFalse(units[0]["is_record"])
        _, metrics = rl.objective([torch.zeros(2, requires_grad=True) for _ in fields], fields,
                                  [[0., 0.] for _ in fields], rl.DEFAULT_RL_CONFIG)
        self.assertEqual(float(metrics["exact_record_reward"]), 0.)
        with self.assertRaisesRegex(AssertionError, "Incomplete"):
            rl.training_units(fields[:1])

    def test_fresh_preparation_excludes_spent_groups_without_reassigning_roles(self):
        config = dict(recipe.DEFAULT_CONFIG, max_length=10000, train_per_source=32,
                      eval_per_source=16, synthetic_groups=600, data_intervention="reasoning")
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(recipe.SOURCES, {}, clear=True):
            original, _ = recipe.prepare_data(config, FakeTokenizer(), Path(root) / "old")
            excluded = rl.fresh_evidence_groups(original)
            fresh, manifest = recipe.prepare_data(config, FakeTokenizer(), Path(root) / "fresh", excluded_groups=excluded)
        self.assertEqual(original["train"], fresh["train"])
        self.assertEqual(original["development"], fresh["development"])
        self.assertEqual(manifest["excluded_groups_sha256"], recipe.digest(excluded))
        for role, rows in fresh.items():
            for row in rows:
                self.assertNotIn(row["group"], excluded)
                self.assertEqual(recipe.split_group(row["group"], 17), role)
        self.assertTrue(any(unit["is_record"] for unit in rl.training_units(fresh["train"])))


class WarmStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source_run = cls.root / "supervised"
        cls.config = dict(recipe.DEFAULT_CONFIG, rank=2, alpha=4, max_steps=2, accumulation=1,
                          evaluate_every=1, checkpoint_every=1, learning_rate=1e-3,
                          max_accuracy_drop=1., max_class_recall_drop=1., max_nll_increase=1., max_brier_increase=1.)
        model = tiny_model(cls.config)
        model.peft_config["default"].base_model_name_or_path = recipe.MODEL_ID
        cls.data = {role: copy.deepcopy(encoded_rows()) for role in recipe.ROLES}
        cls.data["train"] = record_fields()
        cls.original_manifest = dict(schema=recipe.VERSION, config=cls.config, sources={},
                                    counts={role: len(rows) for role, rows in cls.data.items()},
                                    hashes={role: recipe.digest(rows) for role, rows in cls.data.items()})
        recipe.atomic_json(cls.source_run / "CONFIG.json", dict(identity="supervised-fixture", config=cls.config,
                            data_manifest_sha256=recipe.digest(cls.original_manifest)))
        recipe.atomic_json(cls.source_run / "ENVIRONMENT.json", {})
        recipe.fit(model, cls.data, cls.config, cls.source_run, "supervised-fixture")
        cls.config, cls.checkpoint, cls.source = rl.source_checkpoint(cls.source_run, 2)
        cls.data_dir = cls.root / "data"
        recipe.atomic_json(cls.data_dir / "DATA_MANIFEST.json", cls.original_manifest)
        for role, rows in cls.data.items():
            recipe.atomic_json(cls.data_dir / f"{role}.json", rows)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model(self):
        model = tiny_model(self.config)
        model.peft_config["default"].base_model_name_or_path = recipe.MODEL_ID
        recipe.restore(self.checkpoint, model, self.source["identity"])
        return model

    def test_source_data_binding_and_step_zero_rejection(self):
        data, _ = rl.read_source_data(self.data_dir, self.source_run)
        self.assertEqual(data, self.data)
        with self.assertRaisesRegex(AssertionError, "No trained"):
            rl.source_checkpoint(self.source_run, 0)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            for role, rows in self.data.items():
                recipe.atomic_json(path / f"{role}.json", rows)
            recipe.atomic_json(path / "DATA_MANIFEST.json", {**self.original_manifest, "config": {}})
            with self.assertRaisesRegex(AssertionError, "binding"):
                rl.read_source_data(path, self.source_run)

    def test_fresh_adapter_cannot_impersonate_a_trained_warm_start(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(recipe, "logits") as readout:
            with self.assertRaisesRegex(AssertionError, "declared supervised warm start"):
                rl.fit(tiny_model(self.config), self.data, self.config, rl.DEFAULT_RL_CONFIG,
                       Path(root), "wrong-warm-start", self.source)
            readout.assert_not_called()

    def test_real_warm_start_interruption_resume_and_reference_tamper(self):
        config = {**self.config, "max_steps": 3}
        data = self.data
        class Protected(dict):
            def __getitem__(self, role):
                if role not in ("train", "development"):
                    raise AssertionError("Protected role opened: " + role)
                return super().__getitem__(role)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            full = self.model()
            expected = rl.fit(full, Protected(data), config, rl.DEFAULT_RL_CONFIG, root / "full", "paired", self.source)
            interrupted = self.model()
            def stop(kind, entry):
                if kind == "update" and entry["step"] == 2:
                    raise RuntimeError("disconnect")
            with self.assertRaisesRegex(RuntimeError, "disconnect"):
                rl.fit(interrupted, Protected(data), config, rl.DEFAULT_RL_CONFIG, root / "resumed", "paired", self.source, on_event=stop)
            actual = rl.fit(interrupted, Protected(data), config, rl.DEFAULT_RL_CONFIG, root / "resumed", "paired", self.source)
            self.assertEqual(actual, expected)
            for name, tensor in get_peft_model_state_dict(full).items():
                torch.testing.assert_close(tensor, get_peft_model_state_dict(interrupted)[name], atol=0, rtol=0)
            before = recipe.file_digest(self.checkpoint / "adapter/adapter_model.safetensors")
            self.assertEqual(before, self.source["adapter_sha256"])
            cache = root / "resumed/REFERENCE_LOGITS.json"
            cache.write_text(cache.read_text() + " ")
            with self.assertRaisesRegex(AssertionError, "reference cache changed"):
                rl.fit(self.model(), Protected(data), config, rl.DEFAULT_RL_CONFIG, root / "resumed", "paired", self.source)

    def test_stage_export_locks_rl_source_and_supervised_lineage(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            model = self.model()
            recipe.atomic_json(run / "CONFIG.json", dict(config=self.config))
            recipe.atomic_json(run / "ENVIRONMENT.json", {})
            selection = rl.fit(model, self.data, self.config, rl.DEFAULT_RL_CONFIG, run, "rl-export", self.source)
            gate = recipe.calibrate_and_gate(model, self.data, self.config, run, "rl-export", selection)
            export = recipe.export_bundle(model, ReloadableTinyTokenizer(), self.config, self.original_manifest,
                                         run, "rl-export", selection, gate, tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)
            metadata, contract, _ = recipe.verify_export(export, "rl-export")
            self.assertEqual(contract["supervised_parent"], self.source)
            self.assertEqual(contract["training_stage"], rl.VERSION)
            self.assertEqual(contract["loss"]["brier_weight"], rl.DEFAULT_RL_CONFIG["brier_weight"])
            self.assertIn("decision_rl.py", metadata["files"])
            self.assertIn("POSTTRAINING_PLAN.json", metadata["files"])
            (export / "decision_rl.py").write_text("changed")
            with self.assertRaisesRegex(AssertionError, "Frozen export changed"):
                recipe.verify_export(export, "rl-export")


if __name__ == "__main__":
    unittest.main()
