"""Offline release packaging tests; no Hub repository, token or model downloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import release
import train as recipe
from test_train import ReloadableTinyTokenizer, encoded_rows, tiny_model


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.manual_seed(17)
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name) / "run"
        cls.identity = "offline-release-fixture"
        cls.config = {**recipe.DEFAULT_CONFIG, "rank": 2, "alpha": 4, "max_steps": 4,
                      "accumulation": 1, "evaluate_every": 1, "checkpoint_every": 1, "learning_rate": 1e-3}
        data = {role: encoded_rows() for role in recipe.ROLES}
        recipe.atomic_json(cls.root / "CONFIG.json", dict(identity=cls.identity, config=cls.config))
        recipe.atomic_json(cls.root / "ENVIRONMENT.json", dict(model=dict(precision="fp32-cpu", compute_capability=[])))
        model = tiny_model(cls.config)
        selected = recipe.fit(model, data, cls.config, cls.root, cls.identity)
        gate = recipe.calibrate_and_gate(model, data, cls.config, cls.root, cls.identity, selected)
        recipe.export_bundle(model, ReloadableTinyTokenizer(), cls.config, dict(sources={}), cls.root,
                             cls.identity, selected, gate, tokenizer_loader=ReloadableTinyTokenizer.from_pretrained)
        cls.archive = recipe.archive_export(cls.root, cls.identity)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name) / "stage"

    def prepare(self):
        return release.prepare_release(self.root / "export", self.stage)

    def test_package_pins_base_and_keeps_locked_bundle(self):
        self.prepare()
        manifest = release.verify_release(self.stage)
        self.assertGreater(manifest["exported_step"], 0)
        self.assertEqual(json.loads((self.stage / "adapter_config.json").read_text())["revision"], recipe.MODEL_REVISION)
        self.assertTrue((self.stage / "bundle/EXPORT_LOCK.json").is_file())
        self.assertFalse(any(name.endswith("resume.pt") for name in manifest["files"]))
        self.assertIn("not declared", (self.stage / "README.md").read_text())
        self.assertEqual(self.prepare(), self.stage)
        with self.assertRaisesRegex(AssertionError, "Release request changed"):
            release.prepare_release(self.root / "export", self.stage, "Different name")

    def test_corruption_and_extra_assets_are_rejected(self):
        self.prepare()
        path = self.stage / "adapter_model.safetensors"
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(AssertionError, "Release asset changed"):
            release.verify_release(self.stage)

    def test_parent_is_not_branded_as_trained_model(self):
        metadata, contract, config = recipe.verify_export(self.root / "export", self.identity)
        with mock.patch.object(recipe, "verify_export", return_value=(metadata, {**contract, "exported_step": 0}, config)):
            with self.assertRaisesRegex(AssertionError, "frozen parent"):
                self.prepare()
        self.assertFalse(self.stage.exists())

    def test_archive_reuse_checks_source_and_blocks_traversal(self):
        extracted = Path(self.temp.name) / "input"
        export = release.unpack_export(self.archive, extracted)
        self.assertEqual(release.unpack_export(self.archive, extracted), export)
        other = Path(self.temp.name) / "bad.zip"
        with zipfile.ZipFile(other, "w") as archive:
            archive.writestr("export/../../outside.txt", "bad")
        with self.assertRaisesRegex(AssertionError, "Invalid archive path"):
            release.unpack_export(other, Path(self.temp.name) / "new-input")
        with self.assertRaisesRegex(AssertionError, "Extraction source changed"):
            release.unpack_export(other, extracted)

    def test_fixture_upload_is_rejected_before_remote_mutation(self):
        self.prepare()
        api = mock.Mock()
        with self.assertRaisesRegex(AssertionError, "Offline test fixtures"):
            release.upload_private_release(self.stage, "owner/model", "test-token", api=api)
        api.create_repo.assert_not_called()

    def test_upload_uses_new_private_repo_and_checks_commit_inventory(self):
        self.prepare()
        manifest = release.verify_release(self.stage)
        api = mock.Mock()
        api.repo_info.return_value.private = True
        api.upload_folder.return_value.oid = "fixed-commit"
        api.list_repo_files.return_value = list(manifest["files"]) + ["RELEASE_MANIFEST.json", ".gitattributes"]
        with mock.patch.object(release, "verify_release", return_value={**manifest, "offline_fixture": False}):
            commit = release.upload_private_release(self.stage, "owner/model", "test-token", api=api)
        self.assertEqual(commit.oid, "fixed-commit")
        api.create_repo.assert_called_once_with(repo_id="owner/model", repo_type="model", private=True, exist_ok=False)
        self.assertEqual(api.list_repo_files.call_args.kwargs["revision"], "fixed-commit")
