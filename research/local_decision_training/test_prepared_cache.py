"""Prepared-data reuse binds code, tokenizer, configuration and every admitted role."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import train as recipe
from test_train import FakeTokenizer


class CacheTokenizer(FakeTokenizer):
    backend_tokenizer = mock.Mock(to_str=mock.Mock(return_value="offline tokenizer"))
    chat_template = "offline template"
    special_tokens_map = {}


class CacheTests(unittest.TestCase):
    def test_cache_reuses_data_without_reconverting_or_reencoding(self):
        config = {**recipe.DEFAULT_CONFIG, "train_per_source": 16, "eval_per_source": 16,
                  "synthetic_groups": 1000, "max_length": 8192}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(recipe.SOURCES, {}, clear=True):
            data, manifest = recipe.prepare_data(config, CacheTokenizer(), root)
            with mock.patch.object(recipe, "synthetic_rows", side_effect=AssertionError("regenerated")), \
                 mock.patch.object(recipe, "encode", side_effect=AssertionError("reencoded")):
                cached, cached_manifest = recipe.prepare_data(config, CacheTokenizer(), root)
            self.assertEqual(cached, data)
            self.assertEqual(cached_manifest, manifest)
            path = Path(root) / "development.json"
            path.write_text("[]")
            with self.assertRaisesRegex(AssertionError, "Prepared data changed"):
                recipe.prepare_data(config, CacheTokenizer(), root)

    def test_cache_rejects_changed_config_tokenizer_and_code(self):
        config = {**recipe.DEFAULT_CONFIG, "train_per_source": 16, "eval_per_source": 16,
                  "synthetic_groups": 1000, "max_length": 8192}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(recipe.SOURCES, {}, clear=True):
            recipe.prepare_data(config, CacheTokenizer(), root)
            with self.assertRaisesRegex(AssertionError, "Prepared data identity changed"):
                recipe.prepare_data({**config, "learning_rate": 1e-5}, CacheTokenizer(), root)
            tokenizer = CacheTokenizer()
            tokenizer.chat_template = "changed template"
            with self.assertRaisesRegex(AssertionError, "Prepared data identity changed"):
                recipe.prepare_data(config, tokenizer, root)
            with mock.patch.object(recipe, "file_digest", return_value="changed code"):
                with self.assertRaisesRegex(AssertionError, "Prepared data identity changed"):
                    recipe.prepare_data(config, CacheTokenizer(), root)

    def test_cache_rejects_changed_provenance(self):
        config = {**recipe.DEFAULT_CONFIG, "train_per_source": 16, "eval_per_source": 16,
                  "synthetic_groups": 1000, "max_length": 8192}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(recipe.SOURCES, {}, clear=True):
            recipe.prepare_data(config, CacheTokenizer(), root)
            path = Path(root) / "DATA_MANIFEST.json"
            manifest = json.loads(path.read_text())
            manifest["audit"] = {"invented": 1}
            recipe.atomic_json(path, manifest)
            with self.assertRaisesRegex(AssertionError, "Prepared data manifest changed"):
                recipe.prepare_data(config, CacheTokenizer(), root)
