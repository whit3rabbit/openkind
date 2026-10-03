#!/usr/bin/env python3
"""Validate lockstep crate releases and publish serially with a resumable ledger."""

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "OpenKind release tooling (https://github.com/whit3rabbit/openkind)"
SEMVER = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
                    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
                    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?")
METADATA_DELAYS = (30, 60, 120, 240, 480)
PENDING_TIMEOUT = 15 * 60
POLL_INTERVAL = 10
CARGO = ["cargo", "+1.98.1"]
NEW_CRATE_INTERVAL = 600
UPDATE_INTERVAL = 60


class ReleaseError(Exception):
    pass


class TransientError(ReleaseError):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def strict_version(tag):
    match = SEMVER.fullmatch(tag.removeprefix("v")) if tag.startswith("v") else None
    if not match or any(part.isdigit() and len(part) > 1 and part.startswith("0")
                        for part in (match.group(4) or "").split(".")):
        raise ReleaseError("release tag must be v followed by a valid SemVer version")
    return tag[1:]


def run(root, command):
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode:
        raise ReleaseError(f"{' '.join(command)} failed:\n{redact(result.stderr or result.stdout)}")
    return result.stdout.strip()


def redact(value):
    token = os.environ.get("CARGO_REGISTRY_TOKEN")
    return value.replace(token, "[redacted]") if token else value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rename prevents interruptions from leaving a ledger that looks completed.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     prefix=path.name + ".", suffix=".tmp") as output:
        pending = Path(output.name)
        try:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def build_plan(root, tag, metadata, commit, dirty):
    workspace = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]
    version = strict_version(tag)
    if workspace["version"] != version:
        raise ReleaseError("release tag does not match workspace.package.version")
    license_path = (root / workspace.get("license-file", "")).resolve()
    if not workspace.get("license-file") or not license_path.is_file():
        raise ReleaseError("workspace license-file must reference a shipped license text")
    text = license_path.read_text()
    for identifier, marker in (("MIT", "MIT License"), ("Apache-2.0", "Apache License")):
        if identifier in workspace["license"] and marker not in text:
            raise ReleaseError(f"workspace license text does not contain {identifier}")
    members = {p["name"]: p for p in metadata["packages"]
               if p["id"] in metadata["workspace_members"]}
    published = {name for name, p in members.items() if p.get("publish") != []}
    dependencies = {name: set() for name in published}
    packages = []
    for name, package in sorted(members.items()):
        if package["version"] != version:
            raise ReleaseError(f"{name} version differs from the workspace release version")
        own_license = package.get("license_file")
        own_path = (Path(package["manifest_path"]).parent / own_license).resolve() if own_license else None
        if package.get("license") != workspace["license"] or own_path != license_path:
            raise ReleaseError(f"{name} must inherit the workspace license and license-file")
        internal = []
        for dep in package["dependencies"]:
            if dep["name"] not in members:
                continue
            # Cargo omits unversioned development dependencies from published metadata.
            if dep["kind"] == "dev" and dep["req"] == "*":
                continue
            required = dep["req"].split("+", 1)[0]
            release = version.split("+", 1)[0]
            if required not in (release, "^" + release, "=" + release):
                raise ReleaseError(f"{name} dependency {dep['name']} must track version {version}")
            internal.append(dep["name"])
            if name in published:
                if dep["name"] not in published:
                    raise ReleaseError(f"{name} depends on unpublished package {dep['name']}")
                dependencies[name].add(dep["name"])
        packages.append({"name": name, "version": version, "publish": name in published,
                         "status": "planned" if name in published else "excluded",
                         "registry_status": "unchecked",
                         "license": package["license"], "license_file": workspace["license-file"],
                         "rust_version": package.get("rust_version"),
                         "internal_dependencies": sorted(set(internal))})
    order = []
    remaining = {name: set(deps) for name, deps in dependencies.items()}
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps)
        if not ready:
            raise ReleaseError("cycle in publishable crate dependencies")
        order.extend(ready)
        for name in ready:
            del remaining[name]
        for deps in remaining.values():
            deps.difference_update(ready)
    if not order:
        raise ReleaseError("workspace has no publishable packages")
    return {"schema_version": 1, "tag": tag, "version": version, "commit": commit,
            "dirty": dirty, "packages": packages, "publish_order": order,
            "timing": {"new_crate_interval_seconds": NEW_CRATE_INTERVAL,
                       "existing_crate_interval_seconds": UPDATE_INTERVAL,
                       "metadata_max_attempts": 6, "metadata_retry_seconds": list(METADATA_DELAYS),
                       "pending_timeout_seconds": PENDING_TIMEOUT}}


def inspect(root, tag, expected_commit=None):
    commit = run(root, ["git", "rev-parse", "HEAD"])
    if expected_commit and expected_commit != commit:
        raise ReleaseError("HEAD differs from the pinned release commit")
    dirty = bool(run(root, ["git", "status", "--porcelain", "--untracked-files=all"]))
    metadata = json.loads(run(root, ["cargo", "metadata", "--no-deps", "--format-version", "1",
                                    "--offline", "--locked"]))
    return build_plan(root, tag, metadata, commit, dirty), metadata


def verify_packages(root, plan, metadata, allow_dirty=False):
    if plan["dirty"] and not allow_dirty:
        raise ReleaseError("package verification requires a clean checkout; audit with --allow-dirty")
    command = [*CARGO, "package", "--workspace", "--registry", "crates-io", "--locked"]
    for package in plan["packages"]:
        if not package["publish"]:
            command.extend(["--exclude", package["name"]])
    if allow_dirty:
        command.append("--allow-dirty")
    # Cargo 1.98.1's offline local-registry staging cannot verify unpublished dependencies.
    run(root, command)
    archives = {}
    target = Path(metadata["target_directory"]) / "package"
    for name in plan["publish_order"]:
        archive = target / f"{name}-{plan['version']}.crate"
        with tarfile.open(archive) as source:
            members = source.getnames()
            manifest = tomllib.loads(source.extractfile(f"{name}-{plan['version']}/Cargo.toml").read().decode())
            license_file = manifest["package"].get("license-file")
            if not license_file or f"{name}-{plan['version']}/{license_file}" not in members:
                raise ReleaseError(f"{name} package does not contain its license-file")
        archives[name] = {"sha256": digest(archive), "size_bytes": archive.stat().st_size}
    return archives


def retry_after(value, now):
    if not value:
        return None
    try:
        return now + max(0, int(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(now, parsed.timestamp())
        except (ValueError, TypeError, OverflowError):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    return None
                return max(now, parsed.timestamp())
            except (ValueError, TypeError, OverflowError):
                return None


def assert_source(root, plan, metadata, archives):
    if run(root, ["git", "rev-parse", "HEAD"]) != plan["commit"] or run(
            root, ["git", "status", "--porcelain", "--untracked-files=all"]):
        raise ReleaseError("source changed after release preparation")
    tag_commit = run(root, ["git", "rev-parse", "--verify", f"refs/tags/{plan['tag']}^{{commit}}"])
    if tag_commit != plan["commit"]:
        raise ReleaseError("release tag changed after preparation")
    for name, archive in archives.items():
        path = Path(metadata["target_directory"]) / "package" / f"{name}-{plan['version']}.crate"
        if digest(path) != archive["sha256"]:
            raise ReleaseError(f"prepared archive changed for {name}")


class Registry:
    def __init__(self, clock=time):
        self.clock = clock

    def get(self, url):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code == 429 or error.code >= 500:
                after = retry_after(error.headers.get("Retry-After"), self.clock.time())
                if error.code == 429 and after is None:
                    after = self.clock.time() + NEW_CRATE_INTERVAL
                raise TransientError(f"registry metadata HTTP {error.code}", after) from error
            raise ReleaseError(f"registry metadata HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TransientError("registry metadata request failed") from error

    def api(self, name, version):
        url = f"https://crates.io/api/v1/crates/{name}/{urllib.parse.quote(version, safe='')}"
        body = self.get(url)
        if body is None:
            return None
        entry = json.loads(body)["version"]
        if entry.get("crate") != name or entry.get("num") != version:
            raise ReleaseError("registry version metadata identifies a different release")
        return entry

    def index(self, name, version):
        lower = name.lower()
        prefix = "1" if len(lower) == 1 else "2" if len(lower) == 2 else f"3/{lower[0]}" if len(lower) == 3 else f"{lower[:2]}/{lower[2:4]}"
        body = self.get(f"https://index.crates.io/{prefix}/{lower}")
        if body is not None:
            for line in body.splitlines():
                entry = json.loads(line)
                if entry["vers"] == version:
                    if entry.get("name") != name:
                        raise ReleaseError("sparse index metadata identifies a different crate")
                    return entry
        return None

    def exists(self, name):
        return self.get(f"https://crates.io/api/v1/crates/{name}") is not None


class CargoPublisher:
    def __init__(self, root, archives=None, target_directory=None, version=None,
                 check_source=lambda: None):
        self.root = root
        self.archives, self.target_directory, self.version = archives, target_directory, version
        self.check_source = check_source

    def __call__(self, name):
        if self.archives is not None:
            # Registry resolution must reproduce the staged archive before any immutable upload.
            candidate_target = Path(self.target_directory) / "release-registry-check"
            run(self.root, [*CARGO, "package", "--registry", "crates-io", "--locked", "-p", name,
                            "--target-dir", str(candidate_target)])
            archive = candidate_target / "package" / f"{name}-{self.version}.crate"
            if digest(archive) != self.archives[name]["sha256"]:
                raise ReleaseError(f"registry-resolved package checksum differs for {name}; refusing upload")
        self.check_source()
        result = subprocess.run([*CARGO, "publish", "--registry", "crates-io", "--locked", "-p", name],
                                cwd=self.root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = redact(result.stdout)
        if result.returncode == 0:
            return "accepted", None
        limited = re.search(r"Please try again after (.+?) and see", output)
        header = re.search(r"retry-after:\s*([^\r\n]+)", output, re.I)
        if limited or re.search(r"\b429\b|too many (?:new crates|updates|requests)", output, re.I):
            return "rate_limited", limited.group(1) if limited else header.group(1) if header else None
        if re.search(r"\b4[0-9]{2}\b|unauthorized|forbidden|authentication|"
                     r"no token|invalid token|already (?:uploaded|exists)|failed to (?:compile|verify|prepare)|"
                     r"license|description", output, re.I):
            raise ReleaseError(f"cargo publish failed for {name}:\n{output}")
        if re.search(r"\b50[0-9]\b|timed? out|timeout|connection (?:reset|closed)|broken pipe|"
                     r"failed to (?:upload|publish|download)|spurious network", output, re.I):
            # A transport error can occur after an accepted upload. Reconcile, never guess.
            return "unknown", None
        raise ReleaseError(f"cargo publish failed for {name}:\n{output}")


class Publisher:
    def __init__(self, plan, archives, report, registry, upload, clock=time, check_source=lambda: None):
        self.plan, self.archives, self.report = plan, archives, report
        self.registry, self.upload, self.clock = registry, upload, clock
        self.check_source = check_source
        self.state = {"schema_version": 1, "tag": plan["tag"], "version": plan["version"],
                      "commit": plan["commit"], "publish_order": plan["publish_order"],
                      "status": "running", "next_upload_at": 0, "crates": {}}
        if report.exists():
            previous = json.loads(report.read_text())
            if not isinstance(previous, dict):
                raise ReleaseError("resume ledger must be a JSON object")
            if previous.get("phase") != "preflight" or previous.get("crates"):
                for key in ("schema_version", "tag", "version", "commit", "publish_order"):
                    if previous.get(key) != self.state[key]:
                        raise ReleaseError(f"resume ledger differs in {key}")
                self.state = previous
                if not isinstance(previous.get("crates"), dict) or set(previous["crates"]) != set(plan["publish_order"]):
                    raise ReleaseError("resume ledger crate set differs from the release plan")
                if not isinstance(previous.get("next_upload_at"), (int, float)) or not math.isfinite(previous["next_upload_at"]):
                    raise ReleaseError("invalid resume ledger upload cooldown")
        for name in plan["publish_order"]:
            record = self.state["crates"].setdefault(name, {"status": "planned", "attempts": 0,
                                                          "sha256": archives[name]["sha256"],
                                                          "next_attempt_at": 0})
            if record["sha256"] != archives[name]["sha256"]:
                raise ReleaseError(f"resume checksum differs for {name}")
            if record["status"] not in ("planned", "uploading", "pending", "unknown", "confirmed", "rate_limited", "rejected"):
                raise ReleaseError(f"invalid resume status for {name}")
            if not isinstance(record["attempts"], int) or record["attempts"] < 0:
                raise ReleaseError(f"invalid resume attempt count for {name}")
            for key in ("next_attempt_at", "pending_deadline"):
                value = record.get(key, 0)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ReleaseError(f"invalid resume {key} for {name}")

    def save(self):
        self.state["updated_at"] = datetime.fromtimestamp(self.clock.time(), timezone.utc).isoformat()
        atomic_json(self.report, self.state)

    def wait_until(self, when):
        while self.clock.time() < when:
            self.clock.sleep(min(30, when - self.clock.time()))

    def metadata(self, operation, deadline=None):
        cooldown = self.state.get("metadata_retry_at", 0)
        self.wait_until(min(deadline, cooldown) if deadline else cooldown)
        if self.clock.time() < cooldown:
            raise TransientError("registry cooldown exceeds the confirmation deadline", cooldown)
        for attempt in range(6):
            try:
                return operation()
            except TransientError as error:
                if attempt == 5:
                    raise
                when = max(self.clock.time() + METADATA_DELAYS[attempt], error.retry_after or 0)
                self.state["metadata_retry_at"] = when
                self.save()
                self.wait_until(min(when, deadline) if deadline else when)
                if deadline and self.clock.time() >= deadline:
                    raise
        raise AssertionError("unreachable")

    def observed(self, name, deadline=None):
        version = self.plan["version"]
        api = self.metadata(lambda: self.registry.api(name, version), deadline)
        index = self.metadata(lambda: self.registry.index(name, version), deadline)
        expected = self.archives[name]["sha256"]
        for entry, checksum in ((api, "checksum"), (index, "cksum")):
            if entry is not None:
                if entry.get("yanked"):
                    raise ReleaseError(f"{name} {version} is yanked; refusing to continue")
                if entry.get(checksum) != expected:
                    raise ReleaseError(f"registry checksum conflict for {name} {version}")
        if index is not None and api is None:
            raise ReleaseError(f"registry API/index disagree for {name} {version}")
        return "confirmed" if index is not None else "pending" if api is not None else "absent"

    def confirm(self, name):
        record = self.state["crates"][name]
        deadline = record.setdefault("pending_deadline", self.clock.time() + PENDING_TIMEOUT)
        self.save()
        while True:
            try:
                confirmed = self.observed(name, deadline) == "confirmed"
            except TransientError:
                if self.clock.time() < deadline:
                    raise
                confirmed = False
            if confirmed:
                record["status"] = "confirmed"
                self.save()
                return
            if self.clock.time() >= deadline:
                record["status"] = "unknown"
                self.save()
                raise ReleaseError(f"{name} publication remains unconfirmed after 15 minutes; do not re-upload")
            self.wait_until(min(deadline, self.clock.time() + POLL_INTERVAL))

    def publish_one(self, name):
        record = self.state["crates"][name]
        observed = self.observed(name)
        if observed == "confirmed":
            record["status"] = "confirmed"
            self.save()
            return
        if observed == "pending" or record["status"] in ("uploading", "pending", "unknown", "confirmed"):
            record["status"] = "pending"
            self.save()
            self.confirm(name)
            return
        interval = UPDATE_INTERVAL if self.metadata(lambda: self.registry.exists(name)) else NEW_CRATE_INTERVAL
        record["interval_seconds"] = interval
        for attempt in range(6):
            self.wait_until(max(self.state["next_upload_at"], record["next_attempt_at"]))
            self.check_source()
            record["status"] = "uploading"
            record["attempts"] += 1
            # Persist before invoking Cargo so an interrupted process resumes by observing.
            self.state["next_upload_at"] = self.clock.time() + interval
            self.save()
            try:
                outcome, after = self.upload(name)
            except ReleaseError:
                record["status"] = "rejected"
                self.save()
                raise
            if outcome == "rate_limited":
                record["status"] = "rate_limited"
                record["next_attempt_at"] = max(self.clock.time() + interval,
                                                 retry_after(after, self.clock.time()) or 0)
                self.state["next_upload_at"] = record["next_attempt_at"]
                self.save()
                if attempt == 5:
                    raise ReleaseError(f"{name} reached the six-attempt rate-limit retry budget")
                # A competing publisher may have completed this version during the wait.
                self.wait_until(record["next_attempt_at"])
                if self.observed(name) != "absent":
                    self.confirm(name)
                    return
            elif outcome in ("accepted", "unknown"):
                record["status"] = "pending"
                self.save()
                self.confirm(name)
                return
            else:
                raise ReleaseError(f"unsupported upload outcome: {outcome}")

    def execute(self):
        self.state["status"] = "running"
        self.state.pop("error", None)
        self.save()
        try:
            for name in self.plan["publish_order"]:
                self.publish_one(name)
            self.check_source()
            for name in self.plan["publish_order"]:
                if self.observed(name) != "confirmed":
                    raise ReleaseError(f"{name} is not confirmed at final registry verification")
            self.state["status"] = "complete"
        except BaseException as error:
            self.state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            self.state["error"] = redact(str(error))
            raise
        finally:
            self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="validate crate versions without uploading")
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--verify", action="store_true", help="compile packaged archives (registry metadata access)")
    plan_parser.add_argument("--allow-dirty", action="store_true", help="audit dirty source with --verify only")
    publish_parser = commands.add_parser("publish", help="publish the pinned clean release and resume its report")
    publish_parser.add_argument("--report", type=Path, required=True)
    for command in (plan_parser, publish_parser):
        command.add_argument("--tag", required=True)
        command.add_argument("--commit", help="expected immutable HEAD commit")
    args = parser.parse_args()
    if args.command == "plan" and args.allow_dirty and not args.verify:
        parser.error("--allow-dirty requires --verify")
    def terminate(_signum, _frame):
        raise KeyboardInterrupt("terminated")
    signal.signal(signal.SIGTERM, terminate)
    try:
        plan, metadata = inspect(ROOT, args.tag, args.commit)
        if args.command == "plan":
            atomic_json(args.output, plan)
            if args.verify:
                try:
                    plan["archives"] = verify_packages(ROOT, plan, metadata, args.allow_dirty)
                    plan["package_verification"] = "passed"
                except BaseException as error:
                    plan["package_verification"] = "failed"
                    plan["error"] = redact(str(error))
                    raise
                finally:
                    atomic_json(args.output, plan)
            print(args.output)
            return 0
        if plan["dirty"]:
            raise ReleaseError("publication requires a clean checkout, including untracked source files")
        tagged = run(ROOT, ["git", "rev-parse", "--verify", f"refs/tags/{args.tag}^{{commit}}"])
        if tagged != plan["commit"]:
            raise ReleaseError("release tag must point to HEAD")
        archives = verify_packages(ROOT, plan, metadata)
        check_source = lambda: assert_source(ROOT, plan, metadata, archives)
        Publisher(plan, archives, args.report, Registry(),
                  CargoPublisher(ROOT, archives, metadata["target_directory"], plan["version"], check_source),
                  check_source=check_source).execute()
        print(args.report)
        return 0
    except (ReleaseError, OSError, ValueError, KeyError, tarfile.TarError, KeyboardInterrupt) as error:
        destination = args.output if args.command == "plan" else args.report
        failure = {"schema_version": 1, "phase": "preflight", "tag": args.tag, "crates": {}}
        if destination.exists():
            try:
                failure = json.loads(destination.read_text())
                if not isinstance(failure, dict):
                    raise ValueError("report is not a JSON object")
            except (OSError, ValueError):
                # Keep an unreadable ledger for inspection instead of replacing its history.
                print(f"release stopped: {redact(str(error))}", file=sys.stderr)
                return 1
        failure["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        failure["error"] = redact(str(error))
        atomic_json(destination, failure)
        print(f"release stopped: {redact(str(error))}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
