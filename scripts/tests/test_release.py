"""Offline release guards: no checkpoint or runtime downloads."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location("release", Path(__file__).parents[1] / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_pinned_manifest(self):
        manifest = json.loads(release.MANIFEST.read_text())
        self.assertEqual(manifest["version"], "1.23.2")
        self.assertEqual(len(manifest["assets"]), 4)
        for target, asset in manifest["assets"].items():
            self.assertIn("/v1.23.2/", asset["url"])
            self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(asset["size_bytes"], 0)
            self.assertEqual("-gpu-" in asset["url"], "linux" in target or "windows" in target)
        self.assertNotIn("musl", " ".join(manifest["assets"]))

    def test_corrupt_archive_size_and_digest_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runtime"
            source.write_bytes(b"verified library")
            asset = {"size_bytes": source.stat().st_size, "sha256": release.digest(source)}
            release.verify(source, asset)
            source.write_bytes(b"corrupt! library!")
            with self.assertRaisesRegex(ValueError, "integrity"):
                release.verify(source, asset)

    def test_zip_and_tar_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "runtime.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escaped", b"invalid")
            with self.assertRaises(ValueError):
                release.extract(archive, root / "stage")
            archive = root / "runtime.tgz"
            with tarfile.open(archive, "w:gz") as output:
                entry = tarfile.TarInfo("../escaped")
                entry.size = 7
                output.addfile(entry, io.BytesIO(b"invalid"))
            with self.assertRaises(tarfile.FilterError):
                release.extract(archive, root / "stage")
            self.assertFalse((root.parent / "escaped").exists())

    def test_candle_override_changes_only_manifest_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text('[dependencies.cudarc]\nfeatures = [\n    "std",\n    "dynamic-linking",\n    "driver",\n]\n')
            (root / "architecture.rs").write_bytes(b"pinned architecture")
            source_digest = release.digest(root / "architecture.rs")
            (root / ".cargo-checksum.json").write_text(json.dumps({"package": "pinned", "files": {"architecture.rs": source_digest}}))
            result = release.patch_cuda_manifest(root)
            self.assertEqual(release.digest(root / "architecture.rs"), source_digest)
            self.assertEqual(result["package_sha256"], "pinned")
            self.assertNotIn("dynamic-linking", (root / "Cargo.toml").read_text())
            self.assertIn('"driver"', (root / "Cargo.toml").read_text())
            checksums = json.loads((root / ".cargo-checksum.json").read_text())
            self.assertEqual(checksums["files"]["Cargo.toml"], release.digest(root / "Cargo.toml"))
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                release.patch_cuda_manifest(root)

    def test_bundle_preserves_providers_symlinks_notices_and_pins(self):
        with tempfile.TemporaryDirectory(prefix="runtime test ") as directory:
            root = Path(directory)
            source = root / "onnxruntime/lib"
            source.mkdir(parents=True)
            (source / "libonnxruntime.so.1.23.2").write_bytes(b"cpu library")
            (source / "libonnxruntime.so").symlink_to("libonnxruntime.so.1.23.2")
            (source / "libonnxruntime_providers_cuda.so").write_bytes(b"provider")
            for name in ("LICENSE", "ThirdPartyNotices.txt"):
                (source.parent / name).write_text("license notice")
            cache = root / "cache"
            cache.mkdir()
            archive = cache / "runtime.tgz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(source.parent, arcname="onnxruntime")
            manifest = root / "manifest.json"
            asset = {"url": "https://invalid.test/runtime.tgz", "size_bytes": archive.stat().st_size, "sha256": release.digest(archive)}
            manifest.write_text(json.dumps({"version": "1.23.2", "assets": {"linux": asset}}))
            stage = root / "stage"
            stage.mkdir()
            with patch.object(release, "MANIFEST", manifest), patch.object(release.urllib.request, "urlopen", side_effect=AssertionError("must stay offline")):
                release.bundle_runtime("linux", stage, cache)
            self.assertTrue((stage / "lib/onnxruntime/libonnxruntime.so").is_symlink())
            self.assertEqual((stage / "lib/onnxruntime/libonnxruntime.so").read_bytes(), b"cpu library")
            self.assertTrue((stage / "lib/onnxruntime/libonnxruntime_providers_cuda.so").is_file())
            self.assertTrue((stage / "licenses/onnxruntime/ThirdPartyNotices.txt").is_file())
            self.assertEqual(json.loads((stage / "onnxruntime.json").read_text())["sha256"], asset["sha256"])

    def test_portable_archive_names_binaries_and_checksum(self):
        with tempfile.TemporaryDirectory(prefix="archive source ") as directory:
            root = Path(directory)
            binaries = root / "binaries"
            binaries.mkdir()
            for name in ("openkind", "openkindd"):
                (binaries / name).write_bytes(b"test executable")
            archive = release.package(binaries, "x86_64-unknown-linux-musl", "v0.1.0", False, root / "dist")
            self.assertEqual(archive.name, "openkind-0.1.0-x86_64-unknown-linux-musl.tar.gz")
            with tarfile.open(archive) as output:
                self.assertEqual(set(output.getnames()), {"openkind", "openkindd", "LICENSE"})
            checksum = archive.with_suffix(archive.suffix + ".sha256")
            self.assertEqual(checksum.read_text().split(), [release.digest(archive), archive.name])
            checksum.write_text("invalid checksum")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                release.smoke(archive, set())

    def test_archive_version_accepts_build_metadata_and_rejects_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            for name in ("openkind", "openkindd"):
                (binaries / name).write_bytes(b"test executable")
            archive = release.package(binaries, "x86_64-apple-darwin", "v0.1.0+build.7",
                                      False, root / "dist")
            self.assertEqual(archive.name, "openkind-0.1.0+build.7-x86_64-apple-darwin.tar.gz")
            for version in ("../outside", "0.1.0/child", "0.1.0\ninvalid", ""):
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "invalid release version"):
                    release.package(binaries, "x86_64-apple-darwin", version, False, root / "dist")

    def test_dependency_audit_rejects_mandatory_cuda(self):
        self.assertTrue(release.CUDA_IMPORT.search("libcublas.so.12"))
        self.assertTrue(release.CUDA_IMPORT.search("cudart64_12.dll"))
        self.assertTrue(release.CUDA_IMPORT.search("nvcuda.dll"))
        self.assertFalse(release.CUDA_IMPORT.search("libc.so.6"))


if __name__ == "__main__":
    unittest.main()
