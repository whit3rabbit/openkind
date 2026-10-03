"""Release guards use fake clocks and registries, never credentials or uploads."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("publish_crates", Path(__file__).parents[1] / "publish-crates.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class Clock:
    def __init__(self):
        self.now = 1_000
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Registry:
    def __init__(self, archives, clock):
        self.archives = archives
        self.clock = clock
        self.present = set()
        self.existing = set()
        self.visible_at = {}

    def api(self, name, version):
        if name in self.present:
            return {"checksum": self.archives[name]["sha256"], "yanked": False}
        return None

    def index(self, name, version):
        if name in self.present and self.clock.time() >= self.visible_at.get(name, 0):
            return {"cksum": self.archives[name]["sha256"], "yanked": False}
        return None

    def exists(self, name):
        return name in self.existing


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Tests must not inherit a user's actual publishing credentials.
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.clock = Clock()
        self.report = self.root / "publish.json"
        self.plan = {"tag": "v0.1.0", "version": "0.1.0", "commit": "c" * 40,
                     "publish_order": ["one", "two"]}
        self.archives = {name: {"sha256": "a" * 64, "size_bytes": 123} for name in self.plan["publish_order"]}
        self.registry = Registry(self.archives, self.clock)
        self.uploads = []

    def upload(self, name):
        self.uploads.append((name, self.clock.time()))
        self.registry.present.add(name)
        return "accepted", None

    def publisher(self, upload=None, **kwargs):
        return release.Publisher(self.plan, self.archives, self.report, self.registry,
                                 upload or self.upload, self.clock, **kwargs)

    def read(self):
        return json.loads(self.report.read_text())

    def test_new_crates_are_serial_and_spaced_by_ten_minutes(self):
        self.publisher().execute()
        self.assertEqual(self.uploads, [("one", 1_000), ("two", 1_600)])
        self.assertEqual(self.read()["status"], "complete")
        self.assertTrue(all(c["status"] == "confirmed" for c in self.read()["crates"].values()))
        self.assertLessEqual(max(self.clock.sleeps), 30)

    def test_existing_crate_updates_use_one_minute_interval(self):
        self.registry.existing.update(self.plan["publish_order"])
        self.publisher().execute()
        self.assertEqual(self.uploads, [("one", 1_000), ("two", 1_060)])

    def test_success_waits_for_index_before_publishing_dependent(self):
        self.registry.visible_at["one"] = 1_030
        self.publisher().execute()
        self.assertIn(10, self.clock.sleeps)
        self.assertEqual(self.read()["crates"]["one"]["status"], "confirmed")

    def test_rate_limit_honors_server_delay_and_persists_attempts(self):
        calls = []
        def upload(name):
            calls.append(self.clock.time())
            if len(calls) == 1:
                return "rate_limited", "1800"
            return self.upload(name)
        self.publisher(upload).execute()
        self.assertEqual(calls, [1_000, 2_800, 3_400])
        self.assertEqual(self.read()["crates"]["one"]["attempts"], 2)

    def test_rate_limit_without_header_uses_new_crate_interval(self):
        count = 0
        def upload(name):
            nonlocal count
            count += 1
            return ("rate_limited", None) if count == 1 else self.upload(name)
        self.publisher(upload).execute()
        self.assertEqual(self.uploads, [("one", 1_600), ("two", 2_200)])

    def test_rate_limit_retry_budget_is_bounded(self):
        with self.assertRaisesRegex(release.ReleaseError, "six-attempt"):
            self.publisher(lambda _: ("rate_limited", None)).execute()
        record = self.read()["crates"]["one"]
        self.assertEqual(record["attempts"], 6)
        self.assertEqual(record["status"], "rate_limited")
        self.assertGreater(record["next_attempt_at"], self.clock.time())

    def test_metadata_backoff_has_six_total_attempts(self):
        publisher = self.publisher()
        calls = []
        def failing():
            calls.append(self.clock.time())
            raise release.TransientError("temporary")
        with self.assertRaises(release.TransientError):
            publisher.metadata(failing)
        self.assertEqual(calls, [1_000, 1_030, 1_090, 1_210, 1_450, 1_930])

    def test_metadata_cooldown_survives_restart(self):
        publisher = self.publisher()
        publisher.state["metadata_retry_at"] = 1_500
        publisher.save()
        resumed = self.publisher()
        calls = []
        resumed.metadata(lambda: calls.append(self.clock.time()))
        self.assertEqual(calls, [1_500])

    def test_metadata_header_delay_is_not_capped(self):
        calls = []
        def operation():
            calls.append(self.clock.time())
            if len(calls) == 1:
                raise release.TransientError("rate limited", 4_000)
            return True
        self.assertTrue(self.publisher().metadata(operation))
        self.assertEqual(calls, [1_000, 4_000])

    def test_confirmation_deadline_never_bypasses_persisted_server_cooldown(self):
        publisher = self.publisher()
        publisher.state["metadata_retry_at"] = 10_000
        calls = []
        with self.assertRaisesRegex(release.TransientError, "confirmation deadline"):
            publisher.metadata(lambda: calls.append(self.clock.time()), deadline=1_900)
        self.assertEqual(calls, [])
        self.assertEqual(self.clock.time(), 1_900)

    def test_unknown_upload_is_reconciled_without_upload_retry(self):
        def unknown(name):
            self.upload(name)
            self.registry.visible_at[name] = self.clock.time() + 20
            return "unknown", None
        self.publisher(unknown).execute()
        self.assertEqual(len(self.uploads), 2)
        self.assertEqual(self.read()["status"], "complete")

    def test_unknown_absent_upload_stops_after_fifteen_minutes(self):
        calls = []
        def unknown(name):
            calls.append(name)
            return "unknown", None
        with self.assertRaisesRegex(release.ReleaseError, "do not re-upload"):
            self.publisher(unknown).execute()
        self.assertEqual(calls, ["one"])
        self.assertEqual(self.clock.time(), 1_900)
        self.assertEqual(self.read()["crates"]["one"]["status"], "unknown")
        with self.assertRaisesRegex(release.ReleaseError, "do not re-upload"):
            self.publisher(unknown).execute()
        self.assertEqual(calls, ["one"])

    def test_pending_metadata_retries_do_not_extend_deadline(self):
        def absent(name):
            def unavailable(*_):
                raise release.TransientError("unavailable", 10_000)
            self.registry.api = unavailable
            return "unknown", None
        with self.assertRaisesRegex(release.ReleaseError, "15 minutes"):
            self.publisher(absent).execute()
        self.assertEqual(self.clock.time(), 1_900)

    def test_resume_confirms_existing_checksums_and_retains_cooldown(self):
        publisher = self.publisher()
        publisher.publish_one("one")
        resumed = self.publisher()
        resumed.execute()
        self.assertEqual(self.uploads, [("one", 1_000), ("two", 1_600)])

    def test_resume_rejects_commit_or_checksum_changes(self):
        self.publisher().save()
        self.plan["commit"] = "d" * 40
        with self.assertRaisesRegex(release.ReleaseError, "commit"):
            self.publisher()
        self.plan["commit"] = "c" * 40
        self.archives["one"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(release.ReleaseError, "checksum"):
            self.publisher()

    def test_registry_conflict_yank_and_disagreement_stop_without_upload(self):
        for kind in ("checksum", "yanked", "disagreement"):
            with self.subTest(kind=kind):
                self.report.unlink(missing_ok=True)
                self.registry.present.add("one")
                if kind == "checksum":
                    self.registry.api = lambda *_: {"checksum": "b" * 64, "yanked": False}
                elif kind == "yanked":
                    self.registry.api = lambda *_: {"checksum": "a" * 64, "yanked": True}
                else:
                    self.registry.api = lambda *_: None
                with self.assertRaises(release.ReleaseError):
                    self.publisher().execute()
                self.assertEqual(self.uploads, [])
                self.assertEqual(self.read()["status"], "failed")

    def test_signal_during_upload_preserves_unknown_outcome(self):
        def interrupted(_):
            raise KeyboardInterrupt("terminated")
        with self.assertRaises(KeyboardInterrupt):
            self.publisher(interrupted).execute()
        self.assertEqual(self.read()["status"], "interrupted")
        self.assertEqual(self.read()["crates"]["one"]["status"], "uploading")

    def test_source_change_stops_before_upload(self):
        def changed():
            raise release.ReleaseError("source changed")
        with self.assertRaisesRegex(release.ReleaseError, "source changed"):
            self.publisher(check_source=changed).execute()
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.read()["crates"]["one"]["attempts"], 0)

    def test_permanent_rejection_does_not_become_unknown(self):
        def denied(_):
            raise release.ReleaseError("unauthorized")
        with self.assertRaisesRegex(release.ReleaseError, "unauthorized"):
            self.publisher(denied).execute()
        self.assertEqual(self.read()["crates"]["one"]["status"], "rejected")
        self.assertEqual(self.clock.time(), 1_000)

    def test_final_registry_verification_detects_lost_confirmation(self):
        checks = []
        original = self.registry.index
        def index(name, version):
            checks.append(name)
            return None if len(checks) >= 5 else original(name, version)
        self.registry.index = index
        with self.assertRaisesRegex(release.ReleaseError, "final registry"):
            self.publisher().execute()
        self.assertEqual(self.read()["status"], "failed")


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "Cargo.toml").write_text('[workspace.package]\nversion="0.1.0"\nlicense="MIT"\nlicense-file="LICENSE"\n')
        (self.root / "LICENSE").write_text("MIT License\nCopyright example\n")

    def package(self, name, dependencies=(), publish=None):
        return {"id": name, "name": name, "version": "0.1.0", "publish": publish,
                "manifest_path": str(self.root / name / "Cargo.toml"), "license": "MIT",
                "license_file": "../LICENSE", "rust_version": "1.88", "dependencies": list(dependencies)}

    def dependency(self, name, kind=None, req="^0.1.0"):
        return {"name": name, "kind": kind, "req": req}

    def plan(self, packages, tag="v0.1.0"):
        metadata = {"packages": packages, "workspace_members": [p["id"] for p in packages]}
        return release.build_plan(self.root, tag, metadata, "c" * 40, True)

    def test_order_includes_retained_dev_dependencies_and_excludes_private_tool(self):
        packages = [self.package("client", [self.dependency("api", "dev")]),
                    self.package("api", [self.dependency("core")]), self.package("core"),
                    self.package("generator", [self.dependency("core")], [])]
        plan = self.plan(packages)
        self.assertEqual(plan["publish_order"], ["core", "api", "client"])
        self.assertEqual(len(plan["packages"]), 4)
        self.assertTrue(all(p["registry_status"] == "unchecked" for p in plan["packages"]))
        self.assertEqual(plan["timing"]["new_crate_interval_seconds"], 600)

    def test_unversioned_dev_dependencies_are_not_retained(self):
        self.assertEqual(self.plan([self.package("client", [self.dependency("private", "dev", "*")]),
                                    self.package("private", publish=[])])["publish_order"], ["client"])

    def test_all_package_versions_and_internal_versions_are_checked(self):
        package = self.package("generator", publish=[])
        package["version"] = "0.2.0"
        with self.assertRaisesRegex(release.ReleaseError, "generator version"):
            self.plan([self.package("core"), package])
        with self.assertRaisesRegex(release.ReleaseError, "must track"):
            self.plan([self.package("core"), self.package("api", [self.dependency("core", req="^0.2.0")])])

    def test_cycles_unpublished_deps_and_missing_license_fail(self):
        with self.assertRaisesRegex(release.ReleaseError, "cycle"):
            self.plan([self.package("a", [self.dependency("b")]), self.package("b", [self.dependency("a")])])
        with self.assertRaisesRegex(release.ReleaseError, "unpublished"):
            self.plan([self.package("a", [self.dependency("b")]), self.package("b", publish=[])])
        package = self.package("a")
        package["license_file"] = None
        with self.assertRaisesRegex(release.ReleaseError, "license"):
            self.plan([package])

    def test_strict_semver_and_workspace_equality(self):
        for tag in ("0.1.0", "v01.1.0", "v1.2", "v0.1.0-01", "v0.1.0-alpha..1", "v0.1.0/"):
            with self.subTest(tag=tag), self.assertRaises(release.ReleaseError):
                release.strict_version(tag)
        self.assertEqual(release.strict_version("v1.2.3-alpha.1+001.sha"), "1.2.3-alpha.1+001.sha")
        with self.assertRaisesRegex(release.ReleaseError, "workspace.package.version"):
            self.plan([self.package("core")], "v0.2.0")

    def test_archive_tamper_and_tag_change_are_detected(self):
        target = self.root / "target"
        archive = target / "package/one-0.1.0.crate"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"canonical package")
        plan = {"tag": "v0.1.0", "version": "0.1.0", "commit": "c" * 40}
        metadata = {"target_directory": str(target)}
        archives = {"one": {"sha256": release.digest(archive)}}
        with patch.object(release, "run", side_effect=["c" * 40, "", "c" * 40]):
            release.assert_source(self.root, plan, metadata, archives)
        archive.write_bytes(b"tampered")
        with patch.object(release, "run", side_effect=["c" * 40, "", "c" * 40]), self.assertRaisesRegex(release.ReleaseError, "archive changed"):
            release.assert_source(self.root, plan, metadata, archives)
        with patch.object(release, "run", side_effect=["c" * 40, "", "d" * 40]), self.assertRaisesRegex(release.ReleaseError, "tag changed"):
            release.assert_source(self.root, plan, metadata, archives)


class CargoTests(unittest.TestCase):
    def test_cargo_wrapped_auth_metadata_build_and_conflict_fail_permanently(self):
        messages = ["failed to publish to registry\nHTTP/2 401\nunauthorized",
                    "failed to publish to registry\nHTTP/2 403\nforbidden",
                    "failed to publish to registry\nHTTP/2 400\ninvalid metadata",
                    "failed to publish to registry\nHTTP/2 409\nconflict",
                    "failed to publish to registry\nHTTP/2 413\ncrate too large",
                    "failed to publish to registry\ncrate version already uploaded",
                    "failed to prepare local package for uploading\nfailed to compile"]
        for message in messages:
            with self.subTest(message=message), patch.object(release.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout=message)):
                with self.assertRaises(release.ReleaseError):
                    release.CargoPublisher(Path("."))("one")

    def test_unknown_transport_and_rate_limit_are_distinguished(self):
        for message, expected in [("failed to publish to registry\nHTTP/2 503", "unknown"),
                                  ("failed to publish to registry\nrequest timed out", "unknown"),
                                  ("HTTP/2 429\nretry-after: 120", "rate_limited")]:
            with self.subTest(message=message), patch.object(release.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout=message)) as run:
                outcome, after = release.CargoPublisher(Path("."))("one")
                self.assertEqual(outcome, expected)
                self.assertEqual(run.call_args.args[0][:2], ["cargo", "+1.98.1"])
                if expected == "rate_limited":
                    self.assertEqual(after, "120")

    def test_retry_after_supports_seconds_http_dates_and_rfc3339(self):
        self.assertEqual(release.retry_after("120", 1_000), 1_120)
        self.assertEqual(release.retry_after("Thu, 01 Jan 1970 00:30:00 GMT", 1_000), 1_800)
        self.assertEqual(release.retry_after("1970-01-01T00:30:00Z", 1_000), 1_800)
        self.assertIsNone(release.retry_after("nonsense", 1_000))

    def test_tokens_are_redacted_in_fatal_diagnostics(self):
        with patch.dict(os.environ, {"CARGO_REGISTRY_TOKEN": "synthetic-test-secret"}, clear=True), patch.object(
                release.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="401 synthetic-test-secret")):
            with self.assertRaises(release.ReleaseError) as raised:
                release.CargoPublisher(Path("."))("one")
            self.assertNotIn("synthetic-test-secret", str(raised.exception))

    def test_registry_resolved_archive_must_match_before_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            archive = target / "package/one-0.1.0.crate"
            archive.parent.mkdir()
            archive.write_bytes(b"staged archive")
            expected = {"one": {"sha256": release.digest(archive)}}
            publisher = release.CargoPublisher(target, expected, target, "0.1.0")
            def replace_archive(*_):
                candidate = target / "release-registry-check/package/one-0.1.0.crate"
                candidate.parent.mkdir(parents=True)
                candidate.write_bytes(b"registry-resolved drift")
                return ""
            with patch.object(release, "run", side_effect=replace_archive), patch.object(release.subprocess, "run") as upload:
                with self.assertRaisesRegex(release.ReleaseError, "refusing upload"):
                    publisher("one")
                upload.assert_not_called()
                self.assertEqual(archive.read_bytes(), b"staged archive")

    def test_registry_resolved_equal_archive_can_reach_mock_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            archive = target / "package/one-0.1.0.crate"
            archive.parent.mkdir()
            archive.write_bytes(b"canonical archive")
            publisher = release.CargoPublisher(target, {"one": {"sha256": release.digest(archive)}}, target, "0.1.0")
            candidate = target / "release-registry-check/package/one-0.1.0.crate"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(archive.read_bytes())
            with patch.object(release, "run", return_value="") as package, patch.object(
                    release.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="mock success")) as upload:
                self.assertEqual(publisher("one"), ("accepted", None))
                self.assertIn("package", package.call_args.args[1])
                self.assertIn("publish", upload.call_args.args[0])

    def test_source_recheck_between_packaging_and_upload(self):
        def changed():
            raise release.ReleaseError("source changed during package verification")
        with patch.object(release.subprocess, "run") as upload, self.assertRaisesRegex(release.ReleaseError, "source changed"):
            release.CargoPublisher(Path("."), check_source=changed)("one")
        upload.assert_not_called()

    def test_dirty_publish_preflight_writes_failure_report_without_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "publish.json"
            plan = {"dirty": True}
            argv = ["publish-crates.py", "publish", "--tag", "v0.1.0", "--report", str(report)]
            with patch.object(release.sys, "argv", argv), patch.object(release, "inspect", return_value=(plan, {})), patch.object(
                    release, "verify_packages") as verify, patch.object(release, "CargoPublisher") as publisher:
                self.assertEqual(release.main(), 1)
                verify.assert_not_called()
                publisher.assert_not_called()
            self.assertEqual(json.loads(report.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
