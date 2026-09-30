import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "sync-model-registry.py"
SPEC = importlib.util.spec_from_file_location("sync_model_registry", SCRIPT)
sync_model_registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync_model_registry)


class FilesBelowTests(unittest.TestCase):
    def test_rejects_file_symlink(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            root = temporary_root / "registry"
            root.mkdir()
            outside = temporary_root / "outside-metadata.json"
            outside.write_text("outside")
            (root / "catalog.json").symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                sync_model_registry.files_below(root)

    def test_rejects_directory_symlink(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            root = temporary_root / "registry"
            root.mkdir()
            outside = temporary_root / "outside-metadata"
            outside.mkdir()
            (root / "models").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                sync_model_registry.files_below(root)


if __name__ == "__main__":
    unittest.main()
