"""Offline loader regressions using actual Transformers checkpoint diagnostics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import train as recipe
from test_train import tiny_model


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.config = {**recipe.DEFAULT_CONFIG, "rank": 2, "alpha": 4}

    def cuda_boundaries(self):
        import contextlib
        import torch
        stack = contextlib.ExitStack()
        for name, value in (("is_available", True), ("is_bf16_supported", True),
                            ("get_device_properties", mock.Mock(total_memory=40 * 1024**3)),
                            ("get_device_name", "offline CUDA boundary"), ("get_device_capability", (8, 0))):
            stack.enter_context(mock.patch.object(torch.cuda, name, return_value=value))
        return stack

    def test_loader_report_serializes_actual_transformers_sets(self):
        import torch
        from transformers import Qwen3_5ForCausalLM
        base = tiny_model(self.config).unload()
        original = Qwen3_5ForCausalLM.from_pretrained
        with tempfile.TemporaryDirectory() as root:
            base.save_pretrained(root)
            raw_diagnostics = []
            def load_local(name, **kwargs):
                self.assertEqual(name, recipe.MODEL_ID)
                self.assertEqual(kwargs["revision"], recipe.MODEL_REVISION)
                self.assertTrue(kwargs["output_loading_info"])
                self.assertIsNone(kwargs["quantization_config"])
                model, info = original(root, dtype=torch.float32, output_loading_info=True, local_files_only=True)
                raw_diagnostics.append(info)
                return model, info
            with self.cuda_boundaries(), mock.patch.object(Qwen3_5ForCausalLM, "from_pretrained", side_effect=load_local):
                model, report = recipe.load_model(self.config)
            self.assertIsInstance(raw_diagnostics[0]["missing_keys"], set)
            self.assertEqual(report["loading_info"]["missing_keys"], [])
            self.assertEqual(json.loads(json.dumps(report)), report)
            path = Path(root) / "ENVIRONMENT.json"
            recipe.atomic_json(path, dict(model=report))
            self.assertEqual(json.loads(path.read_text()), dict(model=report))
            self.assertTrue(any(p.requires_grad for p in model.parameters()))

    def test_diagnostics_preserve_nested_details_and_deterministic_order(self):
        raw = dict(unexpected_keys={"visual.z.weight", "mtp.a.weight"}, missing_keys=set(),
                   mismatched_keys={("layer.weight", (2, 3), (3, 4))},
                   error_msgs=[], nested=[dict(keys=frozenset(("z", "a")))])
        result = recipe._loading_info_json(raw)
        self.assertEqual(result["unexpected_keys"], ["mtp.a.weight", "visual.z.weight"])
        self.assertEqual(result["mismatched_keys"], [["layer.weight", [2, 3], [3, 4]]])
        self.assertEqual(result["nested"], [dict(keys=["a", "z"])])
        self.assertIsInstance(raw["missing_keys"], set)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_loader_still_rejects_missing_mismatched_and_unexpected_text_weights(self):
        from transformers import Qwen3_5ForCausalLM
        for details in (dict(missing_keys={"text.weight"}),
                        dict(mismatched_keys={("text.weight", (2,), (3,))}),
                        dict(unexpected_keys={"unreviewed_text.weight"})):
            info = {**dict(missing_keys=set(), mismatched_keys=set(), unexpected_keys=set()), **details}
            with self.subTest(details=details), self.cuda_boundaries(), \
                 mock.patch.object(Qwen3_5ForCausalLM, "from_pretrained", return_value=(mock.Mock(), info)), \
                 mock.patch.object(recipe, "setup_adapters") as setup:
                with self.assertRaises(AssertionError):
                    recipe.load_model(self.config)
                setup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
