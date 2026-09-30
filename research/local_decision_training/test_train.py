"""Offline checks. Tiny randomly initialized models only; no model/data downloads."""
import ast
import copy
import inspect
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import train as recipe


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages) + "\n"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def save_pretrained(self, path):
        path = Path(path); path.mkdir(parents=True)
        (path / "tokenizer.json").write_text('{}')


class DataTests(unittest.TestCase):
    def test_resume_load_rejects_pickle_execution(self):
        tree = ast.parse(inspect.getsource(recipe.restore))
        loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "load"]
        self.assertEqual(len(loads), 1)
        weights_only = next(keyword.value for keyword in loads[0].keywords
                            if keyword.arg == "weights_only")
        self.assertIsInstance(weights_only, ast.Constant)
        self.assertIs(weights_only.value, True)

    def test_mnli_semantics_and_grouping(self):
        for label, expected in [(0, "entailment"), (1, "__none__"), (2, "contradiction")]:
            row = recipe.convert("mnli", dict(premise="Same Premise", hypothesis="claim", label=label), label)
            self.assertEqual(row["answer_key"], expected)
            self.assertEqual(row["group"], recipe.group_id(" same   premise "))

    def test_multirc_is_binary_not_exclusive_choice(self):
        for label in (0, 1):
            r = recipe.convert("multirc", dict(paragraph="Paragraph", question="Which?", answer="candidate", label=label), label)
            self.assertEqual(r["kind"], "noul")
            self.assertEqual(r["answer_key"], str(bool(label)).lower())

    def test_teacher_criteria_and_soft_targets(self):
        raw = dict(id="one", state="facts", question=dict(type="choice", instructions="choose", criteria={"a": "A", "b": "B", "c": "C"}), expected="a", teacher_probs=dict(a=.6667, b=.1667, c=.1667), domain="test", family="probability", explanation="MUST NOT LEAK")
        r = recipe.convert("plumb", raw, 0)
        self.assertAlmostEqual(sum(r["target"]), 1)
        self.assertGreater(r["target"][1], 0)
        self.assertNotIn("MUST NOT LEAK", json.dumps(r))
        self.assertEqual(recipe.add_omission(r, 1, 17)["answer_key"], "a")
        raw["teacher_probs"]["a"] = 0.9
        with self.assertRaises(AssertionError):
            recipe.convert("plumb", raw, 0)

    def test_omission_is_none_and_preserves_group(self):
        r = recipe.record("banking77", 0, "facts", "question", [("a", "A"), ("b", "B"), ("__none__", "None")], "a")
        omitted = recipe.add_omission(r, 1, 17)
        self.assertEqual(omitted["answer_key"], "__none__")
        self.assertEqual(omitted["group"], r["group"])
        self.assertNotIn("a", [o["key"] for o in omitted["options"]])
        self.assertEqual(r["answer_key"], "a")

    def test_permutation_keeps_target_association(self):
        r = recipe.record("test", 0, "facts", "question", [("a", "A"), ("b", "B")], "b")
        for seed in range(12):
            e = recipe.encode(r, FakeTokenizer(), seed)
            self.assertEqual(e["keys"][e["target"].index(1.0)], "b")
            self.assertEqual(e["code_ids"], [ord("A"), ord("B")])

    def test_counterfactual_groups_and_exact_labels(self):
        rows = list(recipe.synthetic_rows(100, 17))
        groups = {}
        for r in rows:
            groups.setdefault(r["group"], []).append(r)
        self.assertEqual(len(groups), 100)
        self.assertTrue(all(len(v) == 5 for v in groups.values()))
        self.assertTrue(all(len({recipe.split_group(x["group"], 17) for x in v}) == 1 for v in groups.values()))
        self.assertEqual(sum(r["answer_key"] == "__none__" for r in rows), 100)
        self.assertEqual(len({r["state"] for r in rows if r["kind"] == "score"}), 100)

    def test_metrics_and_per_source_retention(self):
        row = dict(id="x", source="one", group="g", kind="choice", keys=["yes", "__none__"], target=[1., 0.], answer_key="yes", label_class="yes", logits=[10., -10.])
        good = recipe.report([row] * 10)
        wrong = copy.deepcopy(row); wrong["logits"] = [-10., 10.]
        bad = recipe.report([wrong] * 10)
        self.assertEqual(good["pooled"]["accuracy"], 1)
        self.assertEqual(bad["pooled"]["false_none_rate"], 1)
        self.assertFalse(recipe.retention(bad, good, recipe.DEFAULT_CONFIG)["passed"])
        self.assertAlmostEqual(sum(recipe.probabilities([1., 2., 3.], 2)), 1)

    def test_identity_locks_refuse_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lock.json"
            recipe.immutable_json(p, {"value": 1})
            recipe.immutable_json(p, {"value": 1})
            with self.assertRaises(AssertionError):
                recipe.immutable_json(p, {"value": 2})


def tiny_model(config):
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    torch.manual_seed(17); torch.set_num_threads(1)
    c = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, rope_parameters={"rope_type": "default", "rope_theta": 10000.,
        "partial_rotary_factor": .5, "mrope_section": [1, 1, 0]})
    c._attn_implementation = "sdpa"
    return recipe.setup_adapters(Qwen3_5ForCausalLM(c), config)


def encoded_rows():
    return [dict(id=str(i), source="tiny", group=str(i), kind="noul", keys=["false", "true"],
                 target=[0., 1.], answer_key="true", label_class="true", supervision="human",
                 input_ids=[1, 2, 3+i], code_ids=[20, 21]) for i in range(4)]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = {**recipe.DEFAULT_CONFIG, "rank": 2, "alpha": 4, "max_steps": 2,
                       "accumulation": 2, "evaluate_every": 1, "checkpoint_every": 1,
                       "learning_rate": 1e-3}

    def test_selected_projection_and_real_hybrid_gradients(self):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        model = tiny_model(self.config)
        row = encoded_rows()[0]
        model.eval()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            full = model.get_base_model()(input_ids=torch.tensor([row["input_ids"]]), use_cache=False).logits[0, -1, row["code_ids"]]
            selected = recipe.logits(model, row)
        torch.testing.assert_close(full, selected, rtol=1e-5, atol=1e-6)
        result = recipe.gradient_preflight(model, row, self.config)
        self.assertTrue(result["finite"])
        self.assertTrue(all(x > 0 for x in result["squared_gradient_norms"].values()))

    def test_train_resume_gate_and_export(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = encoded_rows()
            data = {role: copy.deepcopy(rows) for role in recipe.ROLES}
            model = tiny_model(self.config)
            identity = "offline-tiny-test"
            for name in ("CONFIG.json", "ENVIRONMENT.json"):
                recipe.atomic_json(root / name, {})
            selection = recipe.fit(model, data, self.config, root, identity)
            step2 = root / "checkpoints/step_000002"
            self.assertEqual(recipe.verify_checkpoint(step2, identity)["step"], 2)
            before = recipe.logits(model, rows[0]).detach().clone()
            # Interrupt after a durable step, then execute the remaining update after restart.
            interrupted_root = root / "interrupted"
            interrupted_model = tiny_model(self.config)
            save = recipe.checkpoint
            def interrupt_after_step_one(*args, **kwargs):
                result = save(*args, **kwargs)
                if args[4] == 1:
                    raise RuntimeError("simulated Colab disconnect")
                return result
            with mock.patch.object(recipe, "checkpoint", side_effect=interrupt_after_step_one):
                with self.assertRaisesRegex(RuntimeError, "simulated Colab disconnect"):
                    recipe.fit(interrupted_model, data, self.config, interrupted_root, identity)
            continued = tiny_model(self.config)
            resumed_selection = recipe.fit(continued, data, self.config, interrupted_root, identity)
            self.assertEqual(selection, resumed_selection)
            torch.testing.assert_close(before, recipe.logits(continued, rows[0]).detach())
            # Resume reloads the last optimizer state and reproduces the immutable selection.
            restarted = tiny_model(self.config)
            again = recipe.fit(restarted, data, self.config, root, identity)
            self.assertEqual(selection, again)
            torch.testing.assert_close(before, recipe.logits(restarted, rows[0]).detach())
            gate = recipe.calibrate_and_gate(restarted, data, self.config, root, identity, selection)
            export = recipe.export_bundle(restarted, FakeTokenizer(), self.config,
                {"sources": {"tiny": {"license": "test fixture"}}}, root, identity, selection, gate)
            metadata = json.loads((export / "EXPORT.json").read_text())
            self.assertTrue(metadata["files"])
            for name, sha in metadata["files"].items():
                self.assertEqual(recipe.file_digest(export / name), sha)
            with self.assertRaises(AssertionError):
                recipe.verify_checkpoint(step2, "different-run")
            (step2 / "adapter/adapter_config.json").write_text('{}')
            with self.assertRaises(AssertionError):
                recipe.verify_checkpoint(step2, identity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
