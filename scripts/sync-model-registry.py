#!/usr/bin/env python3
"""Check or copy the checked-in catalog to its public, asset-pinned mirror."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "registry" / "v1"
PUBLIC_REPOSITORY = "whit3rabbit/openkind-model-registry"
PUBLIC_BASE = f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/main/registry/v1/"


def files_below(root: Path) -> dict[Path, Path]:
    return {path.relative_to(root): path for path in root.rglob("*") if path.is_file()}


def git(mirror: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(mirror), *args], capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def check_assets(mirror: Path, manifest: dict, remote: bool) -> int:
    checked = 0
    for artifact in manifest["artifacts"]:
        source = artifact["source"]
        if source["kind"] != "github":
            continue
        if source["repository"] != PUBLIC_REPOSITORY:
            raise ValueError(f"unexpected asset repository: {source['repository']}")
        identity = f"{source['revision']}:{source['path']}"
        data = git(mirror, "show", identity)
        if len(data) != artifact["size"]:
            raise ValueError(f"asset size differs: {identity}")
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise ValueError(f"asset digest differs: {identity}")
        if remote:
            request = urllib.request.Request(
                f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/{source['revision']}/{source['path']}",
                headers={"User-Agent": "openkind-registry-sync"},
            )
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
            if size != artifact["size"] or digest.hexdigest() != artifact["sha256"]:
                raise ValueError(f"public HTTPS asset differs: {identity}")
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mirror",
        type=Path,
        default=Path.home() / "Documents/GitHub/openkind-model-registry",
        help="local checkout of the public registry",
    )
    parser.add_argument(
        "--write", action="store_true", help="copy metadata into a clean mirror checkout"
    )
    parser.add_argument(
        "--remote", action="store_true", help="also compare public HTTPS metadata"
    )
    args = parser.parse_args()
    if args.write and args.remote:
        parser.error("run --remote after committing and pushing the mirror")

    mirror = args.mirror.expanduser().resolve()
    target = mirror / "registry/v1"
    if not (mirror / ".git").exists():
        raise ValueError(f"not a registry checkout: {mirror}")
    if args.write and git(mirror, "status", "--porcelain", "--", "registry/v1"):
        raise ValueError("mirror registry/v1 has uncommitted changes")

    source_files = files_below(SOURCE)
    target_files = files_below(target)
    extras = set(target_files) - set(source_files)
    if extras:
        raise ValueError(f"mirror has extra metadata: {sorted(map(str, extras))}")
    if args.write:
        for relative, source in source_files.items():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        target_files = files_below(target)
    if set(source_files) != set(target_files):
        raise ValueError("mirror metadata file set differs")
    for relative, source in source_files.items():
        if source.read_bytes() != target_files[relative].read_bytes():
            raise ValueError(f"mirror metadata differs: {relative}")

    catalog = json.loads((SOURCE / "catalog.json").read_bytes())
    checked_assets = 0
    for entry in catalog["models"]:
        relative = Path(entry["manifest_path"])
        if relative not in source_files:
            raise ValueError(f"missing manifest: {relative}")
        data = source_files[relative].read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["manifest_sha256"]:
            raise ValueError(f"catalog digest differs: {relative}")
        manifest = json.loads(data)
        if manifest["name"] != entry["name"]:
            raise ValueError(f"catalog identity differs: {relative}")
        checked_assets += check_assets(mirror, manifest, args.remote)

    if args.remote:
        for relative, source in source_files.items():
            request = urllib.request.Request(
                PUBLIC_BASE + relative.as_posix(),
                headers={"User-Agent": "openkind-registry-sync"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.read() != source.read_bytes():
                    raise ValueError(f"public HTTPS metadata differs: {relative}")

    print(
        f"registry mirror matches: {len(source_files)} metadata files, "
        f"{checked_assets} pinned assets"
        + ("; public HTTPS matches" if args.remote else "")
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"registry sync failed: {error}", file=sys.stderr)
        sys.exit(1)
