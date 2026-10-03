"""Exercise release verification policy without invoking compilers or models."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "verify-commit.sh"


class VerificationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="openkind-verify-policy-")
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.repo = base / "repository"
        self.repo.mkdir()
        self.bin = base / "bin"
        self.bin.mkdir()
        self.calls = base / "cargo-calls"
        self.env = {**os.environ, "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                    "TEST_CARGO_CALLS": str(self.calls), "TEST_OS": "Linux", "TEST_ARCH": "x86_64"}
        # The fake SDK must not change how macOS locates the real Git executable.
        self.command("git", f'exec env -u SDKROOT {shlex.quote(shutil.which("git"))} "$@"')
        self.command("uname", 'if [ "$1" = -s ]; then echo "$TEST_OS"; else echo "$TEST_ARCH"; fi')
        self.command("rustc", 'echo "fixture Rust compiler"')
        self.command("cargo", 'printf "%s\\n" "$*" >> "$TEST_CARGO_CALLS"\n'
                     'if [ "${TEST_FAIL_CLIPPY:-0}" = 1 ] && [ "$1" = clippy ]; then exit 9; fi\n'
                     'case "$*" in *--list*) echo "fixture_test: test";; esac')
        for directory in ("research/14_phase3b_backbone_parity_results",
                          "crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8",
                          "crates/openkind-core/schemas"):
            folder = self.repo / directory
            folder.mkdir(parents=True)
            (folder / "fixture").write_text("offline fixture\n")
        (self.repo / ".gitignore").write_text("target/\n")
        shutil.copy2(SCRIPT, self.repo / "verify.sh")
        for arguments in (("init", "-q"), ("add", "."),
                          ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                           "commit", "-qm", "initial fixture")):
            subprocess.run(["git", *arguments], cwd=self.repo, env=self.env, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (self.repo / "second").write_text("second clean commit\n")
        subprocess.run(["git", "add", "second"], cwd=self.repo, env=self.env, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                        "commit", "-qm", "second fixture"], cwd=self.repo, env=self.env, check=True)

    def command(self, name, body):
        path = self.bin / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)

    def run_verification(self, *extra):
        return subprocess.run(["bash", "verify.sh", "--offline-only", "--output-dir", "target/report", *extra],
                              cwd=self.repo, env=self.env, text=True, capture_output=True)

    def test_linux_default_checks_onnx_without_native_cuda(self):
        result = self.run_verification()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text()
        self.assertNotIn("--all-features", calls)
        self.assertIn("--features onnx,onnx-cuda,onnx-rocm", calls)
        self.assertNotIn("--features onnx,onnx-cuda,onnx-rocm,cuda", calls)
        manifest = (self.repo / "target/report/manifest.env").read_text()
        self.assertIn("exit_status=0", manifest)

    def test_apple_silicon_uses_mlx_without_native_cuda(self):
        self.env.update(TEST_OS="Darwin", TEST_ARCH="arm64")
        self.command("xcrun", "echo /fixture/sdk")
        self.command("sw_vers", "echo fixture-macos")
        self.command("sysctl", "echo fixture-hardware")
        result = self.run_verification()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--features mlx,onnx,onnx-cuda,onnx-rocm", self.calls.read_text())
        rejected = self.run_verification("--cuda")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("--cuda requires", rejected.stderr)

    def test_offline_verification_does_not_require_checkpoint_evidence(self):
        result = self.run_verification("--reference-root", "missing-reference",
                                       "--head-bundle-root", "missing-head")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("qwen35-full-parity", self.calls.read_text())

    def test_model_verification_requires_checkpoint_evidence_before_compiling(self):
        for arguments, message in (
            (("--reference-root", "missing-reference"), "reference root not found"),
            (("--head-bundle-root", "missing-head"), "head bundle root not found"),
        ):
            with self.subTest(message=message):
                result = subprocess.run(
                    ["bash", "verify.sh", "--checkpoint-root", str(self.repo), *arguments],
                    cwd=self.repo, env=self.env, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.calls.exists())

    def test_failure_retains_the_failing_gate_and_exit_status(self):
        self.env["TEST_FAIL_CLIPPY"] = "1"
        result = self.run_verification()
        self.assertEqual(result.returncode, 9)
        manifest = (self.repo / "target/report/manifest.env").read_text()
        self.assertIn("failed_gate=cargo-clippy-default", manifest)
        self.assertIn("exit_status=9", manifest)
        self.assertTrue((self.repo / "target/report/cargo-clippy-default.log").is_file())

    def test_dirty_checkout_fails_before_running_compilers(self):
        (self.repo / "uncommitted").write_text("neighboring work\n")
        result = self.run_verification()
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires a clean working tree", result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
