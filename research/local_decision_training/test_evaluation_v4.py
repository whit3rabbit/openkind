"""Offline v4 evaluation and presentation-schedule regression checks."""
import copy
import math
import tempfile
import unittest
from unittest import mock

import train as recipe


def prediction(index, *, hit=True, group=None, **metadata):
    return dict(id=str(index), source="rules", group=str(index) if group is None else group,
                kind="choice", keys=["yes", "__none__"], target=[1., 0.], answer_key="yes",
                label_class="yes", logits=[math.log(.9), math.log(.1)] if hit else [0., 1.],
                family="logic", template="held_out", **metadata)


class EvaluationTests(unittest.TestCase):
    def test_confidence_coverage_matches_export_and_excludes_noul(self):
        rows = [prediction(0), {**prediction(1), "kind": "noul"}]
        result = recipe.metrics(rows)
        maximum = result["risk_coverage"]["max_probability"]
        entropy = result["risk_coverage"]["exported_entropy_confidence"]
        self.assertEqual(maximum["thresholds"]["0.8"]["n"], 2)
        self.assertEqual(entropy["thresholds"]["0.8"]["n"], 0)
        self.assertEqual(entropy["n_eligible"], 1)
        self.assertFalse(result["action_policy"])

    def test_complete_pairs_and_requests_do_not_count_missing_siblings(self):
        rows = [prediction(0, group="a", atomic_unit="pair", atomic_size=2,
                           request_id="request", request_size=3),
                prediction(1, group="a", atomic_unit="pair", atomic_size=2,
                           request_id="request", request_size=3),
                prediction(2, group="a", hit=False, request_id="request", request_size=3)]
        full = recipe.report(rows)
        self.assertEqual(full["paired_correctness"]["all_correct"], 1)
        self.assertEqual(full["complete_request_correctness"]["all_correct"], 0)
        incomplete = recipe.report(rows[:1])
        self.assertEqual(incomplete["paired_correctness"]["n_incomplete"], 1)
        self.assertIsNone(incomplete["paired_correctness"]["all_correct"])

    def test_diagnostics_cannot_change_source_macro_selection(self):
        rows = [prediction(i) for i in range(10)]
        control = recipe.report(rows)
        diagnostic = prediction("extra", hit=False, diagnostic=True)
        result = recipe.report(rows + [diagnostic])
        self.assertEqual(result["by_source"], control["by_source"])
        self.assertEqual(result["macro_nll"], control["macro_nll"])
        self.assertEqual(result["diagnostic_by_source"]["rules"]["accuracy"], 0)
        self.assertIn("rules/logic", result["by_family"])
        self.assertEqual(result["by_source"]["rules"]["predicted_class_counts"], {"yes": 10})

    def test_cluster_bootstrap_preserves_pairing_and_reports_sparse_groups(self):
        baseline = [prediction(i, hit=False, group=str(i//2)) for i in range(20)]
        candidate = [{**r, "logits": [1., 0.]} for r in baseline]
        result = recipe.paired_comparison(candidate, baseline, samples=100)
        self.assertEqual(result, recipe.paired_comparison(candidate, baseline, samples=100))
        self.assertEqual(result["slices"]["primary"]["n_groups"], 10)
        self.assertEqual(result["slices"]["primary"]["estimates"]["accuracy"]["interval_95"], [1., 1.])
        sparse = recipe.paired_comparison(candidate[:4], baseline[:4], samples=100)
        self.assertIsNone(sparse["slices"]["primary"]["estimates"]["accuracy"]["interval_95"])
        changed = copy.deepcopy(baseline)
        changed[0]["group"] = "wrong"
        with self.assertRaisesRegex(AssertionError, "identity"):
            recipe.paired_comparison(candidate, changed)
        with self.assertRaisesRegex(AssertionError, "Unpaired"):
            recipe.paired_comparison(candidate, baseline[:-1])

    def test_sparse_class_evidence_never_claims_confirmed_retention(self):
        result = recipe.report([prediction(i, group="same") for i in range(20)])
        screen = recipe.retention(result, result, recipe.DEFAULT_CONFIG)
        self.assertTrue(screen["passed"])
        self.assertIn("rules: yes", screen["sparse_class_slices"])
        self.assertFalse(screen["confirmed_retention"])

    def test_source_presentation_slots_stay_fixed_across_data_interventions(self):
        config = {**recipe.DEFAULT_CONFIG, "max_steps": 7, "accumulation": 4, "train_per_source": 10}
        common = [dict(id="a"+str(i), source="human") for i in range(5)]
        left = common + [dict(id="old"+str(i), source="rules") for i in range(3)]
        right = common + [dict(id="new"+str(i), source="rules") for i in range(11)]
        a, b = recipe.build_training_schedule(left, config), recipe.build_training_schedule(right, config)
        self.assertEqual([left[i]["source"] for i in a], [right[i]["source"] for i in b])
        self.assertEqual([left[i]["id"] for i in a if left[i]["source"] == "human"],
                         [right[i]["id"] for i in b if right[i]["source"] == "human"])
        self.assertEqual(a, recipe.build_training_schedule(left, config))
        with self.assertRaisesRegex(AssertionError, "Diagnostic"):
            recipe.build_training_schedule([{**left[0], "diagnostic": True}], config)

    def test_gate_reuses_one_decision_and_refuses_changed_panels(self):
        data = {"calibration": [prediction(1)], "gate": [prediction(2)]}
        with tempfile.TemporaryDirectory() as directory:
            saved = dict(identity="run", selected_step=1,
                         panel_hashes={k: recipe.digest(v) for k, v in data.items()})
            recipe.atomic_json(recipe.Path(directory)/"GATE_RESULT.json", saved)
            with mock.patch.object(recipe, "predict", side_effect=AssertionError("gate reopened")):
                self.assertEqual(recipe.calibrate_and_gate(None, data, {}, directory, "run", {"selected_step": 1}), saved)
                with self.assertRaisesRegex(AssertionError, "panels changed"):
                    recipe.calibrate_and_gate(None, {**data, "gate": []}, {}, directory, "run", {"selected_step": 1})


if __name__ == "__main__":
    unittest.main()
