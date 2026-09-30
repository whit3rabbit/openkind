"""Offline checks for comparator artifact integrity and invocation."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("jeeves_helper", Path(__file__).parents[1] / "jeeves.py")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class JeevesTests(unittest.TestCase):
    def test_source_rejects_revision_drift_and_local_changes(self):
        lock = {"source_revision": "pinned"}
        with patch.object(HELPER.subprocess, "check_output", return_value="other\n"):
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                HELPER.verify_source(Path("source"), lock)
        with patch.object(HELPER.subprocess, "check_output", side_effect=["pinned\n", " M inference/engine.py\n"]):
            with self.assertRaisesRegex(ValueError, "local changes"):
                HELPER.verify_source(Path("source"), lock)

    def test_verification_rejects_corruption_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "head.pt"
            original = b"pinned artifact"
            lock = {"files": [{"path": path.name, "size": len(original),
                               "sha256": hashlib.sha256(original).hexdigest()}]}
            path.write_bytes(original)
            HELPER.verify_weights(root, lock)
            path.write_bytes(b"X" * len(original))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                HELPER.verify_weights(root, lock)
            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                HELPER.verify_weights(root, lock)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                HELPER.verify_weights(root, lock)

    def test_server_uses_local_weights_and_loopback(self):
        root = Path("/persistent model files")
        args = SimpleNamespace(port=8009, max_think=768, fp8=False)
        command = HELPER.serve_command(root / "source", root / "weights", args)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--model") + 1], str(root / "weights"))
        self.assertIn("--no-fp8", command)


if __name__ == "__main__":
    unittest.main()
