"""Offline v4 admission, exact-label, and occurrence-presentation checks."""
import collections
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import train as recipe


class SoftTargetTests(unittest.TestCase):
    def test_omission_never_discards_small_teacher_probability_mass(self):
        options = [("a", "A"), ("b", "B"), ("__none__", "Neither")]
        for target in ([.9995, .0005, 0.], [0., 1., 0.]):
            row = recipe.record("plumb", 0, "Facts", "Choose", options, "a", target=target)
            self.assertEqual(recipe.add_omission(row, 1., 17), row)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages) + "\n"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def row(index, group="document", unit=None, size=1):
    result = recipe.record("fixture", index, f"Facts {index}", "Select the supported outcome.",
                           [("a", "Outcome A"), ("b", "Outcome B")], "a", group=group,
                           atomic_unit=unit, atomic_size=size)
    result["role"] = "train"
    return result


def sample(rows, cap, **overrides):
    audit = collections.Counter()
    selected = recipe.admit_and_sample(rows, cap, {**recipe.DEFAULT_CONFIG, **overrides},
                                       Tokenizer(), audit, {}, {}, {})
    return selected, audit


class V4DataTests(unittest.TestCase):
    def test_plain_ce_and_seed_streams(self):
        self.assertEqual(recipe.VERSION, "local-decision-mix-v4")
        self.assertEqual(recipe.DEFAULT_CONFIG["label_smoothing"], 0)
        self.assertEqual(recipe.DEFAULT_CONFIG["brier_weight"], 0)
        config = {**recipe.DEFAULT_CONFIG, "seed": 41, "split_seed": 42, "sampling_seed": 43}
        self.assertEqual([recipe.experiment_seed(config, kind) for kind in
                          ("split", "initialization", "sampling", "augmentation")], [42, 41, 43, 41])

    def test_atomic_units_skip_capacity_without_fragmenting_documents(self):
        rows = [row(i, unit="pair", size=2) for i in range(2)]
        picked, audit = sample(rows, 1)
        self.assertEqual(picked, [])
        self.assertEqual(audit["fixture:train:unused_capacity"], 1)
        self.assertEqual(audit["fixture:train:unit_exceeds_remaining_cap"], 1)
        picked, _ = sample(rows, 2)
        self.assertEqual(len(picked), 2)
        ordinary = [row(i) for i in range(50)]
        picked, _ = sample(ordinary, 3)
        self.assertEqual(len(picked), 3)
        self.assertEqual({r["group"] for r in picked}, {"document"})

    def test_admission_dedup_and_missing_members_reject_whole_unit(self):
        rows = [row(i, unit="pair", size=2) for i in range(2)]
        rows[1]["state"] = "X" * 5000
        picked, audit = sample(rows, 2)
        self.assertEqual(picked, [])
        self.assertEqual(audit["fixture:train:rejected_unit_rows"], 2)
        rows[1]["state"] = rows[0]["state"]
        picked, audit = sample(rows, 2)
        self.assertEqual(picked, [])
        self.assertEqual(audit["fixture:train:duplicate"], 1)
        picked, audit = sample(rows[:1], 2)
        self.assertEqual(picked, [])
        self.assertEqual(audit["fixture:train:incomplete_unit_rows"], 1)

    def test_legacy_rule_pairs_and_triplets_are_declared(self):
        rows = list(recipe.synthetic_rows(10, 17))
        units = collections.defaultdict(list)
        for item in rows:
            if item["atomic_unit"]:
                units[item["atomic_unit"]].append(item)
        self.assertEqual(len(units), 30)
        self.assertEqual(collections.Counter(len(members) for members in units.values()), {2: 20, 3: 10})
        for members in units.values():
            self.assertEqual({r["atomic_size"] for r in members}, {len(members)})

    def test_exact_reasoning_labels_and_complete_requests(self):
        requests, units = collections.defaultdict(list), collections.defaultdict(list)
        fixture_answers = collections.defaultdict(set)
        rows = list(recipe.reasoning_rows(100, 17))
        self.assertEqual(len(rows), 1400)
        for item in rows:
            facts, family = item["generator_facts"], item["family"]
            if family == "numeric_cap":
                expected = str(min(facts["requested"], facts["cap"]))
            elif family == "chronology":
                expected = "alpha" if facts["event_alpha"] < facts["event_beta"] else "beta"
            elif family == "instruction_flip":
                values = [facts["first_value"], facts["second_value"]]
                expected = str(min(values) if "smaller" in item["question"] else max(values))
            else:
                values = [facts[key] for key in ("condition_a", "condition_b")]
                if " AND " in item["question"]:
                    expected = "deny" if "false" in values else "approve" if all(v == "true" for v in values) else "__none__"
                else:
                    expected = "approve" if "true" in values else "deny" if all(v == "false" for v in values) else "__none__"
            self.assertEqual(item["answer_key"], expected)
            self.assertEqual(item["supervision"], "exact")
            self.assertEqual(item["none_reason"], "missing_evidence" if expected == "__none__" else None)
            units[item["atomic_unit"]].append(item)
            if item["request_size"] > 1:
                requests[item["request_id"]].append(item)
                fixture_answers[item["question_id"]].add(item["answer_key"])
        self.assertEqual(len(requests), 100)
        for members in requests.values():
            self.assertEqual(len(members), 3)
            self.assertEqual(len({r["state"] for r in members}), 1)
            self.assertEqual({r["question_id"] for r in members}, {"capped", "earliest", "eligible"})
        for members in units.values():
            self.assertEqual({r["atomic_size"] for r in members}, {len(members)})
            if members[0]["family"] == "chronology" and len(members) == 2:
                self.assertEqual(members[0]["generator_facts"], members[1]["generator_facts"])
                self.assertEqual(members[0]["answer_key"], members[1]["answer_key"])
                self.assertNotEqual(members[0]["state"], members[1]["state"])
                self.assertEqual({r["chronology_display"] for r in members}, {"oldest_first", "newest_first"})
        self.assertEqual(fixture_answers["eligible"], {"approve", "deny", "__none__"})
        self.assertEqual(fixture_answers["earliest"], {"alpha", "beta"})
        capped = [item for item in rows if item.get("question_id") == "capped"]
        self.assertTrue(any(item["generator_facts"]["requested"] < item["generator_facts"]["cap"] for item in capped))
        self.assertTrue(any(item["generator_facts"]["requested"] > item["generator_facts"]["cap"] for item in capped))

    def test_templates_are_disjoint_by_role(self):
        templates, groups = collections.defaultdict(set), {}
        for item in recipe.reasoning_rows(400, 17):
            role = recipe.split_group(item["group"], 17)
            templates[role].add(item["template"])
            self.assertEqual(groups.setdefault(item["group"], role), role)
        self.assertEqual(set(templates), set(recipe.ROLES))
        self.assertEqual(len(set.union(*templates.values())), len(recipe.ROLES))

    def test_optional_data_intervention_fixes_eval_and_other_sources(self):
        config = {**recipe.DEFAULT_CONFIG, "train_per_source": 40, "eval_per_source": 16, "synthetic_groups": 400}
        raw = [dict(passage=f"Independent passage {i}", question="Is this supplied?", answer=i % 2 == 0)
               for i in range(1000)]
        with mock.patch.dict(recipe.SOURCES, {"boolq": recipe.SOURCES["boolq"]}, clear=True):
            with mock.patch.object(recipe, "download_source", return_value=(raw, {"fixture": True})):
                with tempfile.TemporaryDirectory() as tmp:
                    control, control_manifest = recipe.prepare_data(config, Tokenizer(), tmp + "/control")
                    changed, changed_manifest = recipe.prepare_data({**config, "data_intervention": "reasoning"}, Tokenizer(), tmp + "/reasoning")
        for role in recipe.ROLES[1:]:
            self.assertEqual(control[role], changed[role])
            self.assertEqual(control_manifest["hashes"][role], changed_manifest["hashes"][role])
            self.assertTrue(any(r.get("diagnostic") for r in control[role]))
            self.assertEqual(len([r for r in control[role] if r["source"] == "rules" and not r.get("diagnostic")]), 16)
        self.assertFalse(any(r["intervention"] == "reasoning" for r in control["train"]))
        reasoning_count = sum(r["intervention"] == "reasoning" for r in changed["train"])
        self.assertIn(reasoning_count, (19, 20))
        self.assertEqual(changed_manifest["audit"]["rules:train:reasoning:unused_capacity"], 20 - reasoning_count)
        self.assertLessEqual(len(changed["train"]), len(control["train"]))
        self.assertEqual([r for r in control["train"] if r["source"] == "boolq"],
                         [r for r in changed["train"] if r["source"] == "boolq"])
        for data in (control, changed):
            groups = {}
            for role, rows in data.items():
                units = collections.defaultdict(list)
                for item in rows:
                    self.assertEqual(groups.setdefault(item["group"], role), role)
                    if item["atomic_unit"]:
                        units[item["atomic_unit"]].append(item)
                for members in units.values():
                    self.assertEqual({r["atomic_size"] for r in members}, {len(members)})

    def test_presentation_transforms_preserve_soft_targets_independently(self):
        raw = recipe.record("plumb", "soft", "Facts", "Choose by meaning.",
                            [("a", "Meaning A"), ("b", "Meaning B"), ("__none__", "No supported option")],
                            "a", target=[0.6, 0.3, 0.1])
        encoded = recipe.encode(raw, Tokenizer(), 17)
        original = copy.deepcopy(encoded)
        expected = dict(zip(encoded["keys"], encoded["target"]))
        for mode in ("option_order", "code_assignment", "opaque_keys", "all"):
            config = {**recipe.DEFAULT_CONFIG, "presentation_augmentation": mode}
            presentations = [recipe.training_presentation(encoded, Tokenizer(), config, occurrence) for occurrence in range(8)]
            self.assertEqual(presentations, [recipe.training_presentation(encoded, Tokenizer(), config, occurrence) for occurrence in range(8)])
            self.assertGreater(len({r["rendered_prompt"] for r in presentations}), 1)
            for presented in presentations:
                self.assertEqual(dict(zip(presented["canonical_keys"], presented["target"])), expected)
                self.assertEqual(presented["canonical_target"], raw["target"])
                self.assertIn("__none__", presented["keys"])
                if mode == "option_order":
                    self.assertEqual(presented["permutation"], encoded["permutation"])
                    self.assertEqual(presented["keys"], encoded["keys"])
                if mode == "code_assignment":
                    self.assertEqual([presented["permutation"][j] for j in presented["display_order"]], encoded["permutation"])
                if mode == "opaque_keys":
                    self.assertEqual(presented["permutation"], encoded["permutation"])
                    self.assertEqual(presented["display_order"], encoded["display_order"])
        self.assertEqual(encoded, original)
        self.assertIs(recipe.training_presentation(encoded, None, recipe.DEFAULT_CONFIG, 7), encoded)

    def test_opaque_keys_preserve_boolean_and_ordinal_contracts(self):
        config = {**recipe.DEFAULT_CONFIG, "presentation_augmentation": "opaque_keys"}
        for kind, keys in (("noul", ["false", "true"]), ("score", ["0", "1", "2"])):
            raw = recipe.record("fixture", kind, "facts", "Decide", [(key, f"Meaning {key}") for key in keys], keys[0], kind)
            encoded = recipe.encode(raw, Tokenizer(), 17)
            presented = recipe.training_presentation(encoded, Tokenizer(), config, 3)
            self.assertEqual(presented["keys"], encoded["keys"])
            self.assertEqual(presented["answer_key"], encoded["answer_key"])

    def test_augmented_tiny_training_resumes_exact_weights_and_schedule(self):
        import torch
        from safetensors.torch import load_file
        from test_train import tiny_model

        class TinyPresentationTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return messages[-1]["content"] + "\n"

            def encode(self, text, **kwargs):
                return [20 + ord(c) - ord("A") if c in recipe.CODES else 1 + ord(c) % 18 for c in text]

        tokenizer = TinyPresentationTokenizer()
        config = {**recipe.DEFAULT_CONFIG, "rank": 2, "alpha": 4, "max_steps": 2,
                  "accumulation": 2, "evaluate_every": 1, "checkpoint_every": 1,
                  "learning_rate": 1e-3, "presentation_augmentation": "all", "train_per_source": 4}
        raw = [recipe.record("tiny", i, f"Fact {i}", "Choose.",
                             [("a", "A"), ("b", "B"), ("__none__", "None")], "a", target=[0.6, 0.3, 0.1])
               for i in range(4)]
        rows = [recipe.encode(item, tokenizer, 17) for item in raw]
        data = {"train": rows, "development": rows}
        with tempfile.TemporaryDirectory() as tmp:
            uninterrupted, interrupted = Path(tmp) / "full", Path(tmp) / "resumed"
            selection = recipe.fit(tiny_model(config), data, config, uninterrupted, "augmented-test", tokenizer=tokenizer)
            checkpoint = recipe.checkpoint

            def disconnect(*args, **kwargs):
                result = checkpoint(*args, **kwargs)
                if args[4] == 1:
                    raise RuntimeError("intentional interruption after durable checkpoint")
                return result

            with mock.patch.object(recipe, "checkpoint", side_effect=disconnect):
                with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                    recipe.fit(tiny_model(config), data, config, interrupted, "augmented-test", tokenizer=tokenizer)
            resumed = recipe.fit(tiny_model(config), data, config, interrupted, "augmented-test", tokenizer=tokenizer)
            self.assertEqual(selection, resumed)
            self.assertEqual(json.loads((uninterrupted / "TRAINING_SCHEDULE.json").read_text()),
                             json.loads((interrupted / "TRAINING_SCHEDULE.json").read_text()))
            checkpoint_path = "checkpoints/step_000002/adapter/adapter_model.safetensors"
            complete_weights = load_file(str(uninterrupted / checkpoint_path))
            resumed_weights = load_file(str(interrupted / checkpoint_path))
            self.assertEqual(complete_weights.keys(), resumed_weights.keys())
            for key, value in complete_weights.items():
                torch.testing.assert_close(value, resumed_weights[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
