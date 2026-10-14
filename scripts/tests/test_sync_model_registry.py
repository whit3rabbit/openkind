import importlib.util
import json
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


class DigestCatalogTests(unittest.TestCase):
    def test_source_catalog_publishes_under_its_digest(self):
        digest = "a" * 64
        self.assertEqual(
            sync_model_registry.mirror_path(Path("catalog.json"), digest),
            Path("catalogs") / f"{digest}.json",
        )
        manifest = Path("manifests/test.json")
        self.assertEqual(sync_model_registry.mirror_path(manifest, digest), manifest)

    def test_retains_only_released_catalogs(self):
        data = b'{"models": []}'
        digest = sync_model_registry.sha256(data)
        sync_model_registry.check_retained(Path(f"catalogs/{digest}.json"), data)
        with self.assertRaisesRegex(ValueError, "does not match its digest name"):
            sync_model_registry.check_retained(Path(f"catalogs/{'0' * 64}.json"), data)
        with self.assertRaisesRegex(ValueError, "frozen v0.2.1 bytes"):
            sync_model_registry.check_retained(Path("catalog.json"), data)
        with self.assertRaisesRegex(ValueError, "extra metadata"):
            sync_model_registry.check_retained(Path("stray.json"), data)


class MlxAlternativesTests(unittest.TestCase):
    def test_checked_in_index_covers_every_catalog_entry(self):
        catalog = json.loads((sync_model_registry.SOURCE / "catalog.json").read_text())
        alternatives = json.loads(
            (sync_model_registry.SOURCE / "mlx-alternatives.json").read_text()
        )
        sync_model_registry.validate_mlx_alternatives(
            alternatives, {entry["name"] for entry in catalog["models"]}
        )
        self.assertEqual(len(catalog["models"]), 33)
        unqualified = sum(
            model["openkind_mlx_status"] != "qualified"
            for model in alternatives["models"]
        )
        self.assertEqual(unqualified, 25)

    def test_rejects_unpinned_conversion_revision(self):
        alternatives = {
            "schema": "openkind-mlx-alternatives/v1",
            "checked_at": "2026-10-02",
            "installable": False,
            "scope": "Test fixture.",
            "models": [
                {
                    "catalog_name": "test:abc",
                    "openkind_mlx_status": "none",
                    "assessment": "No path.",
                    "leads": [
                        {
                            "repository": "mlx-community/test",
                            "revision": "main",
                            "license": "apache-2.0",
                            "base_model": "author/test",
                            "relationship": "same-base-conversion",
                            "notes": "Test lead.",
                        }
                    ],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "pinned commit"):
            sync_model_registry.validate_mlx_alternatives(alternatives, {"test:abc"})


if __name__ == "__main__":
    unittest.main()
