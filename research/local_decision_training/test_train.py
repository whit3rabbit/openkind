"""Offline checks. Tiny randomly initialized models only; no model/data downloads."""
import copy
import json
import os
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest import mock

import train as recipe
import benchmark


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages) + "\n"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def save_pretrained(self, path):
        path = Path(path); path.mkdir(parents=True)
        (path / "tokenizer.json").write_text('{}')


class ReloadableTinyTokenizer(FakeTokenizer):
    """Explicit offline bundle seam with code identities inside the tiny model vocabulary."""
    def encode(self, text, **kwargs):
        return [20 + ord(c) - ord("A") if c in recipe.CODES else 1 + ord(c) % 18 for c in text]

    def save_pretrained(self, path):
        path = Path(path); path.mkdir(parents=True)
        (path / "tokenizer.json").write_text('{"offline_fixture": true}')

    @classmethod
    def from_pretrained(cls, path):
        assert json.loads((Path(path) / "tokenizer.json").read_text()) == {"offline_fixture": True}
        return cls()


class DataTests(unittest.TestCase):
    def test_helpsteer2_rating_proxy_and_request_grouping(self):
        raw = dict(prompt="Explain a cancellation", response="Candidate reply", helpfulness=3.5, correctness=3)
        good = recipe.convert("helpsteer2", raw, 0)
        bad = recipe.convert("helpsteer2", {**raw, "response": "A different reply", "correctness": 1}, 1)
        self.assertEqual(good["answer_key"], "true")
        self.assertEqual(bad["answer_key"], "false")
        self.assertNotEqual(good["state"], bad["state"])
        self.assertEqual(good["group"], bad["group"])
        self.assertEqual(recipe.split_group(good["group"], 17), recipe.split_group(bad["group"], 17))
        self.assertEqual(good["supervision"], "human_rating_proxy")
        self.assertNotIn("helpfulness", good["state"])
        self.assertNotIn("correctness", good["state"])
        self.assertIsNone(recipe.convert("helpsteer2", {**raw, "helpfulness": 2.5}, 0))
        self.assertIsNone(recipe.convert("helpsteer2", {**raw, "prompt": "Conversation <extra_id_1> continuation"}, 0))
        with self.assertRaisesRegex(AssertionError, "ratings"):
            recipe.convert("helpsteer2", {**raw, "correctness": float("nan")}, 0)
        self.assertFalse(recipe.DEFAULT_CONFIG["include_helpsteer2"])

    def test_helpsteer2_preparation_balances_training_and_keeps_request_siblings(self):
        raw = [dict(prompt=f"Request {i}", response=f"Reply {j}", helpfulness=4, correctness=1 if j == 0 else 4)
               for i in range(400) for j in range(3)]
        config = {**recipe.DEFAULT_CONFIG, "include_helpsteer2": True, "train_per_source": 100,
                  "eval_per_source": 32, "synthetic_groups": 400}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(recipe.SOURCES, {"helpsteer2": recipe.SOURCES["helpsteer2"]}, clear=True):
                with mock.patch.object(recipe, "download_source", return_value=(raw, recipe.SOURCES["helpsteer2"])):
                    data, manifest = recipe.prepare_data(config, FakeTokenizer(), tmp)
        training = [r for r in data["train"] if r["source"] == "helpsteer2"]
        self.assertEqual(sum(r["answer_key"] == "true" for r in training), 50)
        self.assertEqual(sum(r["answer_key"] == "false" for r in training), 50)
        roles = {}
        for role, rows in data.items():
            for row in rows:
                if row["source"] == "helpsteer2":
                    self.assertEqual(roles.setdefault(row["group"], role), role)
        self.assertEqual(manifest["audit"]["helpsteer2:train:accepted"], 100)

    def test_typesafe_is_excluded_before_download_and_selection(self):
        with mock.patch.dict(recipe.SOURCES, {"blocked": {"repo": "TypeSafe/forbidden"}}):
            with self.assertRaisesRegex(AssertionError, "benchmark-only"):
                recipe.prepare_data(recipe.DEFAULT_CONFIG, FakeTokenizer(), "/unused")
        with self.assertRaisesRegex(AssertionError, "benchmark-only"):
            recipe.download_source({"repo": "typesafe/evalsafe-onet"})
        row = {**encoded_rows()[0], "benchmark_only": True}
        with self.assertRaisesRegex(AssertionError, "Benchmark rows"):
            recipe.fit(None, {"train": [row], "development": []}, recipe.DEFAULT_CONFIG, "/unused", "blocked")
        with self.assertRaisesRegex(AssertionError, "Benchmark rows"):
            recipe.calibrate_and_gate(None, {"calibration": [row]}, recipe.DEFAULT_CONFIG, "/unused", "blocked", {})

    def test_typesafe_conversion_preserves_criteria_and_consensus(self):
        raw = benchmark_fixture()
        row = benchmark.convert("onet", raw)
        self.assertEqual([o["text"] for o in row["options"]], ["Contradicted", "Supported"])
        self.assertEqual(row["target"], [.25, .75])
        self.assertTrue(row["benchmark_only"])
        self.assertNotIn("REFERENCE_ONLY", row["state"] + row["question"])
        raw["kind"] = "choice"
        raw["question_json"] = json.dumps({"type": "choice", "instructions": "Choose", "criteria": {"a": "Meaning A", "b": "Meaning B"}})
        raw["consensus"]["probabilities"] = [{"option": "b", "probability": .5}, {"option": "a", "probability": .5}]
        raw["consensus"]["answer_json"] = None
        row = benchmark.convert("customer_service", raw)
        self.assertEqual(row["target"], [.5, .5, 0.])
        self.assertEqual(row["options"][-1]["key"], "__none__")
        self.assertTrue(row["semantic_none_added"])
        raw["consensus"]["probabilities"][0]["probability"] = .9
        with self.assertRaisesRegex(AssertionError, "mass"):
            benchmark.convert("customer_service", raw)

    def test_typesafe_metric_and_failure_denominators(self):
        self.assertEqual(benchmark.agreement("noul", ["false", "true"], [1., 0.], [1., 0.]), 1.)
        self.assertEqual(benchmark.agreement("noul", ["false", "true"], [1., 0.], [0., 1.]), 0.)
        self.assertAlmostEqual(benchmark.agreement("choice", ["a", "b"], [1., 0.], [.5, .5]), .6887218755408672)
        self.assertEqual(benchmark.agreement("score", ["0", "1", "2"], [0., 1., 0.], [.5, 0., .5]), 1.)
        self.assertEqual(benchmark.agreement("score", ["0", "1", "2"], [1., 0., 0.], [0., 0., 1.]), 0.)
        raw = [benchmark_fixture("one"), benchmark_fixture("two")]
        raw[1]["state_json"] = json.dumps("X" * 5000)
        ledger, sampling = benchmark.prepare_source("onet", raw, FakeTokenizer(), 17, 2048, 1)
        self.assertEqual(sampling["selected_questions"], 2)  # Whole case, before token admission.
        with mock.patch.object(recipe, "logits", return_value=__import__("torch").tensor([0., 0.])):
            result = benchmark.evaluate(mock.Mock(), ledger, 1.)
        self.assertEqual(result["by_kind"]["noul"]["n"], 2)
        self.assertEqual(result["by_kind"]["noul"]["answered"], 1)
        self.assertEqual(result["failures"], {"overlength": 1})
        self.assertLess(result["equal_type_agreement"], .5)

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
        self.assertTrue(all(len(v) == 9 for v in groups.values()))
        self.assertTrue(all(len({recipe.split_group(x["group"], 17) for x in v}) == 1 for v in groups.values()))
        self.assertEqual(sum(r["answer_key"] == "__none__" for r in rows), 300)
        self.assertEqual(len({r["state"] for r in rows if r["kind"] == "score"}), 100)
        for siblings in groups.values():
            conjunction = [r for r in siblings if r["family"] == "partial_evidence_conjunction"]
            self.assertEqual([r["answer_key"] for r in conjunction], ["deny", "__none__"])
            disjunction = [r for r in siblings if r["family"] == "partial_evidence_disjunction"]
            self.assertEqual([r["answer_key"] for r in disjunction], ["approve", "__none__"])

    def test_schema_variants_preserve_soft_targets_and_semantic_none(self):
        row = recipe.record("plumb", 0, "facts", "question",
                            [("a", "A"), ("b", "B"), ("__none__", "No valid answer")], "a",
                            target=[.7, .2, .1])
        encoded = recipe.encode(row, FakeTokenizer(), 17)
        for name, variant in recipe.schema_variants(encoded, FakeTokenizer()):
            expected = dict(zip(encoded["keys"], encoded["target"]))
            self.assertEqual(dict(zip(variant["canonical_keys"], variant["target"])), expected)
            self.assertIn("__none__", variant["keys"])
            self.assertEqual(variant["group"], row["group"])
            self.assertEqual(variant["keys"][variant["target"].index(.7)], variant["answer_key"])

    def test_metrics_and_per_source_retention(self):
        row = dict(id="x", source="one", group="g", kind="choice", keys=["yes", "__none__"], target=[1., 0.], answer_key="yes", label_class="yes", logits=[10., -10.])
        good = recipe.report([row] * 10)
        wrong = copy.deepcopy(row); wrong["logits"] = [-10., 10.]
        bad = recipe.report([wrong] * 10)
        self.assertEqual(good["pooled"]["accuracy"], 1)
        self.assertEqual(bad["pooled"]["false_none_rate"], 1)
        self.assertFalse(recipe.retention(bad, good, recipe.DEFAULT_CONFIG)["passed"])
        brier_only = copy.deepcopy(good)
        brier_only["by_source"]["one"]["brier"] += .02
        self.assertIn("one: Brier regression", recipe.retention(brier_only, good, recipe.DEFAULT_CONFIG)["reasons"])
        self.assertAlmostEqual(sum(recipe.probabilities([1., 2., 3.], 2)), 1)
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(AssertionError):
                recipe.probabilities([0., 1.], temperature)

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


def benchmark_fixture(instance="one"):
    return dict(case_id="case", question_instance_id=instance, kind="noul", state_json=json.dumps("Supplied facts"),
                question_json=json.dumps({"instruction": "Assess the claim", "levels": [
                    {"value": 0, "description": "Contradicted"}, {"value": 1, "description": "Supported"}]}),
                consensus=dict(status="ok", answer_json="true", probabilities=[
                    {"option": "false", "probability": .25}, {"option": "true", "probability": .75}]),
                metadata="REFERENCE_ONLY")


def encoded_rows():
    return [dict(id=str(i), source="tiny", group=str(i), kind="noul", keys=["false", "true"],
                 target=[0., 1.], answer_key="true", label_class="true", supervision="human",
                 input_ids=[1, 2, 3+i], code_ids=[20, 21]) for i in range(4)]


class PickleMarker:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return os.mkdir, (str(self.path),)


class ResumeStateTests(unittest.TestCase):
    def test_rejects_pickle_side_effect(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.pt"
            marker = Path(tmp) / "unpickled"
            torch.save(PickleMarker(marker), path)
            with self.assertRaises(pickle.UnpicklingError):
                recipe.load_resume_state(path)
            self.assertFalse(marker.exists())

    def test_optimizer_scheduler_and_rng_round_trip(self):
        import torch
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9 ** step)
        parameter.square().sum().backward()
        optimizer.step(); scheduler.step()
        state = dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     torch_rng=torch.get_rng_state(), cuda_rng=[torch.get_rng_state()],
                     python_rng=random.getstate())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.pt"
            torch.save(state, path)
            loaded = recipe.load_resume_state(path)
        other = torch.nn.Parameter(parameter.detach().clone())
        resumed_optimizer = torch.optim.AdamW([other], lr=0.01)
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda step: 0.9 ** step)
        resumed_optimizer.load_state_dict(loaded["optimizer"])
        resumed_scheduler.load_state_dict(loaded["scheduler"])
        self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
        for key, value in optimizer.state[parameter].items():
            torch.testing.assert_close(resumed_optimizer.state[other][key], value)
        torch.testing.assert_close(loaded["torch_rng"], state["torch_rng"])
        torch.testing.assert_close(loaded["cuda_rng"][0], state["cuda_rng"][0])
        self.assertEqual(loaded["python_rng"], state["python_rng"])

    def test_rejects_malformed_state(self):
        import torch
        valid = dict(optimizer={}, scheduler={}, torch_rng=torch.get_rng_state(),
                     cuda_rng=[], python_rng=random.getstate())
        malformed = [[], {**valid, "extra": 1},
                     {k: v for k, v in valid.items() if k != "scheduler"}]
        malformed += [{**valid, key: value} for key, value in [
            ("optimizer", []), ("scheduler", None), ("torch_rng", []),
            ("torch_rng", torch.tensor([1.0])), ("cuda_rng", ()),
            ("cuda_rng", [1]), ("python_rng", []), ("python_rng", (3, (), None))]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.pt"
            for state in malformed:
                with self.subTest(state=list(state) if isinstance(state, dict) else state):
                    torch.save(state, path)
                    with self.assertRaises(ValueError):
                        recipe.load_resume_state(path)


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
        combined = {**self.config, "label_smoothing": .05, "brier_weight": .1}
        result = recipe.gradient_preflight(model, row, combined)
        self.assertTrue(all(x > 0 for x in result["squared_gradient_norms"].values()))

    def test_loss_ablation_math_soft_targets_and_permutation(self):
        import torch
        logits = torch.tensor([1.0, -.5, .25], requires_grad=True)
        target = torch.tensor([.7, .2, .1])
        ce = -(target * logits.log_softmax(-1)).sum()
        ce_config = {**self.config, "label_smoothing": 0.0, "brier_weight": 0.0}
        torch.testing.assert_close(recipe.decision_loss(logits, target, ce_config), ce)
        config = {**self.config, "label_smoothing": .05, "brier_weight": .1}
        expected = -((.95 * target + .05 / 3) * logits.log_softmax(-1)).sum()
        expected += .1 * (logits.softmax(-1) - target).square().sum()
        actual = recipe.decision_loss(logits, target, config)
        torch.testing.assert_close(actual, expected)
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(actual, recipe.decision_loss(logits[order], target[order], config))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        for field, value in (("label_smoothing", -1), ("label_smoothing", 1),
                             ("brier_weight", -1), ("brier_weight", float("nan"))):
            with self.assertRaises(AssertionError):
                recipe.decision_loss(logits, target, {**config, field: value})

    def test_reference_inference_admission_isolation_and_confidence(self):
        import torch
        tokenizer = FakeTokenizer()
        contract = dict(schema=recipe.VERSION, prompt_id=recipe.PROMPT_ID,
                        model_id=recipe.MODEL_ID, model_revision=recipe.MODEL_REVISION,
                        answer_codes=recipe.CODES, code_token_ids=[ord(c) for c in recipe.CODES],
                        truncation=False, generation=False, max_length=4096, max_questions=8, temperature=1.)
        question = dict(id="first", kind="choice", instructions="Choose using the facts.",
                        options=[dict(key="yes", text="Yes"), dict(key="__none__", text="No valid option")])
        model = mock.Mock()
        seen = []
        def readout(model, row):
            seen.append(row)
            return torch.zeros(len(row["keys"]))
        with mock.patch.object(recipe, "logits", side_effect=readout) as forward:
            first = recipe.decide(model, tokenizer, "facts", [question], contract)
            renamed = {**question, "id": "second"}
            second = recipe.decide(model, tokenizer, "facts", [renamed], contract)
            self.assertEqual(seen[0]["input_ids"], seen[1]["input_ids"])
            self.assertEqual(first["first"], second["second"])
            self.assertAlmostEqual(first["first"]["confidence"], 0)
            forward.reset_mock()
            with self.assertRaisesRegex(AssertionError, "Overlength"):
                recipe.decide(model, tokenizer, "facts", [question, {**renamed, "instructions": "X" * 5000}], contract)
            forward.assert_not_called()  # Validate the entire request before any model work.
            with self.assertRaises(AssertionError):
                recipe.decide(model, tokenizer, "facts", [question, question], contract)
            with self.assertRaisesRegex(AssertionError, "Too many questions"):
                recipe.decide(model, tokenizer, "facts", [question], {**contract, "max_questions": 0})
            with self.assertRaisesRegex(AssertionError, "semantic-none"):
                recipe.decide(model, tokenizer, "facts", [{**question, "options": [dict(key="yes", text="Yes"), dict(key="no", text="No")]}], contract)
            with self.assertRaisesRegex(AssertionError, "Missing instructions"):
                recipe.decide(model, tokenizer, "facts", [{**question, "instructions": ""}], contract)
            noul = dict(id="binary", kind="noul", instructions="Assess", options=[dict(key=k, text=k) for k in ("false", "true")])
            score = dict(id="ordinal", kind="score", instructions="Rate", options=[dict(key=str(i), text=str(i)) for i in range(3)])
            result = recipe.decide(model, tokenizer, "facts", [noul, score], contract)
            self.assertNotIn("confidence", result["binary"])
            self.assertAlmostEqual(result["ordinal"]["expected_level"], 1.)
            self.assertAlmostEqual(result["ordinal"]["confidence"], 0.)

    def test_schema_diagnostics_maps_probabilities_to_canonical_keys(self):
        import torch
        tokenizer = FakeTokenizer()
        row = recipe.record("plumb", 0, "facts", "question",
                            [("a", "A"), ("b", "B"), ("__none__", "No valid answer")], "a", target=[.7, .2, .1])
        encoded = recipe.encode(row, tokenizer, 17)
        def semantic_readout(model, row):
            return torch.tensor(row["target"]).log()
        with mock.patch.object(recipe, "logits", side_effect=semantic_readout):
            result = recipe.schema_diagnostics(mock.Mock(), tokenizer, [encoded], 4096)
        self.assertEqual({r["transform"] for r in result["rows"]},
                         {"reverse_option_order", "option_order_only", "code_assignment_only", "opaque_option_keys"})
        self.assertTrue(all(r["total_variation"] < 1e-7 and not r["selected_key_changed"] for r in result["rows"]))
        self.assertFalse(result["used_for_selection"])
        self.assertFalse(result["parity_established"])

    def test_resolve_precision_matches_device_capabilities(self):
        pick = recipe.resolve_precision
        self.assertEqual(pick("auto", 40.0, True), "bf16")
        self.assertEqual(pick("auto", 40.0, False), "nf4_fp16")
        self.assertEqual(pick("auto", 24.0, True), "nf4")
        self.assertEqual(pick("auto", 24.0, False), "nf4_fp16")
        self.assertEqual(pick("auto", 15.0, False), "nf4_fp16")
        self.assertEqual(pick("nf4_fp16", 15.0, True), "nf4_fp16")
        for requested, gib, supported in (("bf16", 24.0, True), ("bf16", 40.0, False), ("nf4", 24.0, False),
                                          ("nf4_fp16", 10.0, False), ("fp8", 40.0, True)):
            with self.assertRaises(AssertionError):
                pick(requested, gib, supported)

        import torch
        from transformers import Qwen3_5ForCausalLM
        # BF16 tensor emulation is insufficient for the native-compute paths.
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_properties", return_value=mock.Mock(total_memory=15 * 1024**3)), \
             mock.patch.object(torch.cuda, "is_bf16_supported", side_effect=lambda including_emulation=True: including_emulation) as capability, \
             mock.patch.object(recipe, "resolve_precision", side_effect=RuntimeError("precision boundary")) as resolution, \
             mock.patch.object(Qwen3_5ForCausalLM, "from_pretrained") as loader:
            with self.assertRaisesRegex(RuntimeError, "precision boundary"):
                recipe.load_model({**self.config, "precision": "auto"})
            capability.assert_called_once_with(including_emulation=False)
            resolution.assert_called_once_with("auto", 15.0, False)
            loader.assert_not_called()

    def test_parameter_sweep_configs_vary_one_dimension_and_require_control(self):
        arms = [dict(name="lr_5e_04", learning_rate=5e-4), dict(name="lr_1e_03", learning_rate=1e-3),
                dict(name="lr_2e_03", learning_rate=2e-3)]
        configs = recipe.parameter_sweep_configs(self.config, arms, "learning_rate")
        self.assertEqual([arm["name"] for arm in configs], ["lr_5e_04", "lr_1e_03", "lr_2e_03"])
        for arm in configs:
            fixed = {k: v for k, v in arm["config"].items() if k != "learning_rate"}
            self.assertEqual(fixed, {k: v for k, v in self.config.items() if k != "learning_rate"})
        ranked = recipe.parameter_sweep_configs(self.config, [dict(name="rank_1", rank=1, alpha=2),
                                                             dict(name="rank_2", rank=2, alpha=4)], "rank")
        self.assertEqual([(arm["config"]["rank"], arm["config"]["alpha"]) for arm in ranked], [(1, 2), (2, 4)])
        self.assertTrue(all(arm["config"]["max_steps"] == self.config["max_steps"] for arm in ranked))
        with self.assertRaises(AssertionError):
            recipe.parameter_sweep_configs({**self.config, "alpha": 8},
                                           [dict(name="rank_1", rank=1, alpha=2), dict(name="rank_2", rank=2, alpha=4)], "rank")
        invalid_learning_rate = ([], arms[:1], arms[:1] + arms[:1],
                                 [dict(name="x", learning_rate=5e-4), dict(name="y", learning_rate=5e-4)],
                                 [dict(name="x", learning_rate=5e-4), dict(name="y", learning_rate=1e-3, rank=64)],
                                 [dict(name="x", learning_rate=1e-1), dict(name="y", learning_rate=1e-3)],
                                 [dict(name="x", learning_rate=1e-7), dict(name="y", learning_rate=1e-3)],
                                 [dict(name="x", learning_rate=5e-4), dict(name="y", learning_rate=2e-3)])
        for invalid in invalid_learning_rate:
            with self.assertRaises(AssertionError):
                recipe.parameter_sweep_configs(self.config, invalid, "learning_rate")
        invalid_rank = ([dict(name="rank_1", rank=1), dict(name="rank_2", rank=2, alpha=4)],
                        [dict(name="rank_1", rank=1, alpha=4), dict(name="rank_2", rank=2, alpha=4)],
                        [dict(name="rank_1", rank=1.0, alpha=2), dict(name="rank_2", rank=2, alpha=4)],
                        [dict(name="rank_1", rank=1, alpha=2), dict(name="rank_2", rank=3, alpha=6)],
                        [dict(name="lr_1e_03", learning_rate=1e-3), dict(name="rank_2", rank=2, alpha=4)])
        for invalid in invalid_rank:
            with self.assertRaises(AssertionError):
                recipe.parameter_sweep_configs(self.config, invalid, "rank")

    def test_parameter_sweep_fresh_matched_fits_resume_and_selection(self):
        import torch
        class ProtectedRoles(dict):
            def __getitem__(self, role):
                if role not in ("train", "development"):
                    raise AssertionError("protected panel opened: " + role)
                return super().__getitem__(role)
        data = ProtectedRoles(train=encoded_rows(), development=encoded_rows())
        initial_states = []
        def factory(config):
            model = tiny_model(config)
            initial_states.append({n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
            return model, dict(precision="fp32-cpu", compute_capability=[])
        lr_arms = [dict(name="lr_5e_04", learning_rate=5e-4), dict(name="lr_1e_03", learning_rate=1e-3)]
        with tempfile.TemporaryDirectory() as tmp:
            result = recipe.run_parameter_sweep(factory, data, self.config, tmp, {"source_sha": "tiny-test"},
                                                "learning_rate", lr_arms)
            self.assertEqual(len(result["results"]), 2)
            expected = min(result["results"], key=lambda r: (r["metrics"]["macro_nll"], r["selected_step"], r["index"]))
            self.assertEqual(result["selected_arm"], expected["name"])
            self.assertEqual(result["selected_config"]["learning_rate"], expected["config"]["learning_rate"])
            self.assertTrue(all(not result[k] for k in ("calibration_used", "gate_used", "test_opened", "model_promoted")))
            for arm in result["results"]:
                path = Path(arm["run"])
                self.assertEqual(recipe.verify_checkpoint(path / "checkpoints/step_000002", arm["identity"])["step"], 2)
                self.assertFalse((path / "GATE_RESULT.json").exists())
            resumed = recipe.run_parameter_sweep(factory, data, self.config, tmp, {"source_sha": "tiny-test"},
                                                 "learning_rate", lr_arms)
            self.assertEqual(resumed, result)
            for state in initial_states[1:]:
                for name, value in initial_states[0].items():
                    torch.testing.assert_close(state[name], value, rtol=0, atol=0)
        # Rank arms rebuild adapters at a different capacity; step-zero reports still match.
        rank_arms = [dict(name="rank_1", rank=1, alpha=2), dict(name="rank_2", rank=2, alpha=4)]
        with tempfile.TemporaryDirectory() as tmp:
            ranked = recipe.run_parameter_sweep(factory, data, self.config, tmp, {"source_sha": "tiny-test"},
                                                "rank", rank_arms)
            self.assertEqual(len(ranked["results"]), 2)
            winner = min(ranked["results"], key=lambda r: (r["metrics"]["macro_nll"], r["selected_step"], r["index"]))
            self.assertEqual(ranked["selected_arm"], winner["name"])
            self.assertEqual(ranked["selected_config"]["rank"], winner["config"]["rank"])
            plan = json.loads((Path(ranked["selected_run"]).parent / "SWEEP_PLAN.json").read_text())
            self.assertEqual(plan["dimension"], "rank")
            self.assertEqual(plan["kind"], "parameter")

    def test_sweep_configs_only_vary_losses_and_require_controls(self):
        configs = recipe.loss_sweep_configs(self.config)
        self.assertEqual(len(configs), 6)
        self.assertEqual(recipe.DEFAULT_CONFIG["label_smoothing"], 0.0)
        self.assertEqual(recipe.DEFAULT_CONFIG["brier_weight"], 0.0)
        for arm in configs:
            fixed = {k: v for k, v in arm["config"].items() if k not in ("label_smoothing", "brier_weight")}
            self.assertEqual(fixed, {k: v for k, v in self.config.items() if k not in ("label_smoothing", "brier_weight")})
        for invalid in (configs[:1], [], [dict(name="x", label_smoothing=.05, brier_weight=.1),
                                          dict(name="y", label_smoothing=.02, brier_weight=.1)],
                        [dict(name="ce", label_smoothing=0., brier_weight=0.),
                         dict(name="x", label_smoothing=.05, brier_weight=.1, rank=64)]):
            with self.assertRaises(AssertionError):
                recipe.loss_sweep_configs(self.config, invalid)

    def test_sweep_fresh_matched_fits_selection_and_resume_without_protected_roles(self):
        import torch
        arms = [recipe.LOSS_SWEEP_ARMS[i] for i in (0, 1, 4)]
        class ProtectedRoles(dict):
            def __getitem__(self, role):
                if role not in ("train", "development"):
                    raise AssertionError("protected panel opened: " + role)
                return super().__getitem__(role)
        data = ProtectedRoles(train=encoded_rows(), development=encoded_rows())
        initial_states = []
        def factory(config):
            model = tiny_model(config)
            initial_states.append({n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
            return model, dict(precision="fp32-cpu", compute_capability=[])
        with tempfile.TemporaryDirectory() as tmp:
            result = recipe.run_loss_sweep(factory, data, self.config, tmp, {"source_sha": "tiny-test"}, arms)
            self.assertEqual(len(result["results"]), 3)
            expected = min(result["results"], key=lambda r: (r["metrics"]["macro_nll"], r["selected_step"], r["index"]))
            self.assertEqual(result["selected_arm"], expected["name"])
            self.assertTrue(all(not result[k] for k in ("calibration_used", "gate_used", "test_opened", "model_promoted")))
            for arm in result["results"]:
                path = Path(arm["run"])
                self.assertEqual(recipe.verify_checkpoint(path / "checkpoints/step_000002", arm["identity"])["step"], 2)
                self.assertFalse((path / "GATE_RESULT.json").exists())
            resumed = recipe.run_loss_sweep(factory, data, self.config, tmp, {"source_sha": "tiny-test"}, arms)
            self.assertEqual(resumed, result)
            for state in initial_states[1:]:
                for name, value in initial_states[0].items():
                    torch.testing.assert_close(state[name], value, rtol=0, atol=0)
            winner_model = tiny_model(result["selected_config"])
            run = Path(result["selected_run"])
            identity = result["selected_identity"]
            selection = json.loads((run / "SELECTION.json").read_text())
            full_data = {role: encoded_rows() for role in recipe.ROLES}
            gate = recipe.calibrate_and_gate(winner_model, full_data, result["selected_config"], run, identity, selection)
            export = recipe.export_bundle(winner_model, ReloadableTinyTokenizer(), result["selected_config"],
                                         {"sources": {}}, run, identity, selection, gate,
                                         tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)
            metadata = json.loads((export / "EXPORT.json").read_text())
            self.assertIn("SWEEP_PLAN.json", metadata["files"])
            self.assertIn("SWEEP_RESULT.json", metadata["files"])
            self.assertEqual(json.loads((export / "SWEEP_RESULT.json").read_text()), result)
            self.assertEqual(sum((Path(r["run"]) / "GATE_RESULT.json").exists() for r in result["results"]), 1)

    def test_sweep_rejects_mismatched_parent_before_training_later_arm(self):
        import torch
        arms = [recipe.LOSS_SWEEP_ARMS[i] for i in (0, 4)]
        data = {"train": encoded_rows(), "development": encoded_rows()}
        calls = []
        def factory(config):
            model = tiny_model(config)
            calls.append(config)
            if len(calls) > 1:
                with torch.no_grad():
                    model.get_base_model().get_output_embeddings().weight[20].add_(1.)
            return model, dict(precision="fp32-cpu", compute_capability=[])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(AssertionError, "Step-zero"):
                recipe.run_loss_sweep(factory, data, self.config, tmp, {"source_sha": "tiny-test"}, arms)
            second = next(Path(tmp).glob("sweep_*/combined_default"))
            self.assertFalse((second / "checkpoints/step_000001").exists())

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
            export = recipe.export_bundle(restarted, ReloadableTinyTokenizer(), self.config,
                {"sources": {"tiny": {"license": "test fixture"}}}, root, identity, selection, gate,
                tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)
            metadata = json.loads((export / "EXPORT.json").read_text())
            self.assertTrue(metadata["files"])
            for name, sha in metadata["files"].items():
                self.assertEqual(recipe.file_digest(export / name), sha)
            self.assertIn("benchmark.py", metadata["files"])
            source_loader = mock.Mock(return_value=([benchmark_fixture()], {"upstream_split": "test"}))
            kwargs = dict(max_length=4096, tokenizer_loader=ReloadableTinyTokenizer.from_pretrained,
                          source_loader=source_loader)
            result = benchmark.run_benchmark(restarted, None, self.config, root, identity, **kwargs)
            self.assertEqual(source_loader.call_count, 5)
            self.assertEqual(set(result["candidate"]), set(benchmark.TYPESAFE_SOURCES))
            self.assertFalse(result["plan"]["used_for_training_or_selection"])
            self.assertFalse(result["direct_leaderboard_comparison"])
            self.assertTrue(result["plan"]["extended_context_unqualified"])
            self.assertEqual(result["plan"]["export_max_length"], self.config["max_length"])
            self.assertEqual(benchmark.run_benchmark(restarted, None, self.config, root, identity, **kwargs), result)
            self.assertEqual(source_loader.call_count, 5)
            for name, sha in metadata["files"].items():
                self.assertEqual(recipe.file_digest(export / name), sha)
            with self.assertRaises(AssertionError):
                recipe.verify_checkpoint(step2, "different-run")
            (step2 / "adapter/adapter_config.json").write_text('{}')
            with self.assertRaises(AssertionError):
                recipe.verify_checkpoint(step2, identity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
