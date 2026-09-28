"""Offline checks for the non-final Qwen qualification tools."""

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "qwen-qualification.py"
SPEC = importlib.util.spec_from_file_location("qwen_qualification", SCRIPT)
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


class ChoiceQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def files(self, gold_rows, prediction_rows):
        workload = [
            {
                "id": ident,
                "state": {"source": source},
                "primitive": "choice",
                "text": "Choose",
                "options": [{"id": "yes"}, {"id": "no"}, {"id": "__none__"}],
            }
            for ident, source in (("a", "doc-a"), ("b", "doc-b"))
        ]
        paths = {}
        for name, rows in (("workload", workload), ("gold", gold_rows), ("predictions", prediction_rows)):
            path = self.root / f"{name}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths[name] = path
        digest = hashlib.sha256(paths["workload"].read_bytes()).hexdigest()
        prediction_digest = hashlib.sha256(paths["predictions"].read_bytes()).hexdigest()
        paths["summary"] = self.root / "summary.json"
        paths["summary"].write_text(json.dumps({
            "schema": "openkind-bench/v1",
            "fixture": {"sha256": digest, "rows": 2},
            "prediction_sha256": {"single": prediction_digest},
            "engine": "test", "profile_id": "frozen", "model_revision": "pinned",
        }))
        return argparse.Namespace(**paths, policy_threshold=0.8,
                                  reference_summary=None, reference_predictions=None)

    def test_metrics_include_false_none_proper_scores_and_policy(self):
        gold = [
            {"id": "a", "source_id": "doc-a", "split": "policy_audit", "family": "contract", "gold": "yes", "options": ["yes", "no", "__none__"]},
            {"id": "b", "source_id": "doc-b", "split": "policy_audit", "family": "contract", "gold": "__none__", "options": ["yes", "no", "__none__"]},
        ]
        predictions = [
            {"id": "a", "strategy": "single", "answer": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9, "no": 0.05, "__none__": 0.05}}},
            {"id": "b", "strategy": "single", "answer": {"type": "choice", "choice": "no", "probabilities": {"yes": 0.05, "no": 0.9, "__none__": 0.05}}},
        ]
        report = qualification.quality_report(self.files(gold, predictions))
        metrics = report["metrics"]["policy_audit/contract"]
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["false_none_rate"], 0)
        self.assertEqual(metrics["none_recall"], 0)
        self.assertGreater(metrics["nll"], 0)
        self.assertGreater(metrics["brier_sum_classes"], 0)
        self.assertGreater(metrics["none_brier"], 0)
        self.assertAlmostEqual(metrics["top_label_ece_10"], 0.4)
        self.assertEqual(sum(bucket["count"] for bucket in metrics["top_label_calibration_bins"]), 2)
        self.assertEqual(metrics["policy"]["wrong_accepted_questions"], 1)
        self.assertEqual(metrics["policy"]["accepted_sources"], 2)

    def test_rejects_source_leakage_and_final_split(self):
        gold = [
            {"id": "a", "source_id": "same", "split": "policy_selection", "family": "contract", "gold": "yes", "options": ["yes", "no", "__none__"]},
            {"id": "b", "source_id": "same", "split": "policy_audit", "family": "contract", "gold": "no", "options": ["yes", "no", "__none__"]},
        ]
        predictions = [
            {"id": ident, "strategy": "single", "answer": {"type": "choice", "choice": "yes", "probabilities": {"yes": 1, "no": 0, "__none__": 0}}}
            for ident in ("a", "b")
        ]
        with self.assertRaisesRegex(ValueError, "crosses evaluation splits"):
            qualification.quality_report(self.files(gold, predictions))
        gold[1]["split"] = "final-confirmation"
        with self.assertRaisesRegex(ValueError, "final split remains closed"):
            qualification.quality_report(self.files(gold, predictions))

    def test_rejects_incomplete_distribution_and_wrong_workload_hash(self):
        gold = [
            {"id": ident, "source_id": ident, "split": "policy_audit", "family": "contract", "gold": "yes", "options": ["yes", "no", "__none__"]}
            for ident in ("a", "b")
        ]
        predictions = [
            {"id": ident, "strategy": "single", "answer": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}}}
            for ident in ("a", "b")
        ]
        args = self.files(gold, predictions)
        with self.assertRaisesRegex(ValueError, "probability keys"):
            qualification.quality_report(args)
        predictions[0]["answer"]["probabilities"] = {"yes": 0.9, "no": 0.05, "__none__": 0.05}
        predictions[1]["answer"]["probabilities"] = {"yes": 0.9, "no": 0.05, "__none__": 0.05}
        args.predictions.write_text("".join(json.dumps(row) + "\n" for row in predictions))
        with self.assertRaisesRegex(ValueError, "prediction digest"):
            qualification.quality_report(args)
        summary = json.loads(args.summary.read_text())
        summary["fixture"]["sha256"] = "bad"
        args.summary.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "digest"):
            qualification.quality_report(args)

    def test_paired_reference_reports_supported_answer_loss(self):
        gold = [
            {"id": "a", "source_id": "doc-a", "split": "policy_audit", "family": "contract", "gold": "yes", "options": ["yes", "no", "__none__"]},
            {"id": "b", "source_id": "doc-b", "split": "policy_audit", "family": "contract", "gold": "__none__", "options": ["yes", "no", "__none__"]},
        ]
        primary = [
            {"id": ident, "strategy": "single", "answer": {"type": "choice", "choice": "__none__", "probabilities": {"yes": 0.05, "no": 0.05, "__none__": 0.9}}}
            for ident in ("a", "b")
        ]
        args = self.files(gold, primary)
        args.reference_predictions = self.root / "reference.jsonl"
        reference = [
            {"id": "a", "strategy": "single", "answer": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9, "no": 0.05, "__none__": 0.05}}},
            primary[1],
        ]
        args.reference_predictions.write_text("".join(json.dumps(row) + "\n" for row in reference))
        args.reference_summary = self.root / "reference-summary.json"
        ref_summary = json.loads(args.summary.read_text())
        ref_summary["prediction_sha256"]["single"] = hashlib.sha256(args.reference_predictions.read_bytes()).hexdigest()
        args.reference_summary.write_text(json.dumps(ref_summary))
        report = qualification.quality_report(args)
        retention = report["metrics"]["policy_audit/contract"]["retention_vs_reference"]
        self.assertEqual(retention["reference_correct"], 2)
        self.assertEqual(retention["retained_correct"], 1)
        self.assertEqual(retention["lost_correct"], 1)
        self.assertEqual(retention["supported_lost_to_none"], 1)

    def test_zero_gold_probability_reports_infinite_nll_without_json_infinity(self):
        report = qualification.aggregate([
            ({"gold": "yes", "source_id": "doc-a"},
             {"choice": "no", "probabilities": {"yes": 0.0, "no": 1.0, "__none__": 0.0}})
        ], None)
        self.assertIsNone(report["nll"])
        self.assertEqual(report["nll_infinite_zero_gold_probability"], 1)


class HistoryComparisonTests(unittest.TestCase):
    def test_detects_answer_or_policy_flip_below_probability_tolerance(self):
        left = {"choice": "yes", "probabilities": {"yes": 0.5001, "no": 0.4999}}
        right = {"choice": "no", "probabilities": {"yes": 0.4999, "no": 0.5001}}
        result = qualification.compare_answers(left, right, 0.5, 0.005)
        self.assertLess(result["max_probability_delta"], 0.005)
        self.assertTrue(result["answer_changed"])
        self.assertTrue(result["policy_changed"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
