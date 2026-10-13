#!/usr/bin/env python3
"""Check or copy checked-in registry metadata to its public, asset-pinned mirror."""

import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "registry" / "v1"
PUBLIC_REPOSITORY = "whit3rabbit/openkind-model-registry"
PUBLIC_BASE = f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/main/registry/v1/"
SOURCE_CATALOG = Path("catalog.json")
# v0.2.1 and earlier fetch the mirror's mutable registry/v1/catalog.json and
# pin this digest, so that file stays frozen. Later releases fetch the
# immutable catalogs/<sha256>.json named by their pinned digest instead.
LEGACY_CATALOG = Path("catalog.json")
LEGACY_CATALOG_SHA256 = "c26a2e0aa097ff28c44a8a70395e403e87356470eeec862b471b9658e6c99d34"
DIGEST_CATALOG = re.compile(r"catalogs/([0-9a-f]{64})\.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mirror_path(relative: Path, catalog_sha256: str) -> Path:
    """Map a source metadata path to its mirror path."""
    if relative == SOURCE_CATALOG:
        return Path("catalogs") / f"{catalog_sha256}.json"
    return relative


def check_retained(relative: Path, data: bytes) -> None:
    """Accept only mirror files that released clients still fetch."""
    if relative == LEGACY_CATALOG:
        if sha256(data) != LEGACY_CATALOG_SHA256:
            raise ValueError("mirror legacy catalog.json differs from the frozen v0.2.1 bytes")
        return
    match = DIGEST_CATALOG.fullmatch(relative.as_posix())
    if not match:
        raise ValueError(f"mirror has extra metadata: {relative}")
    if sha256(data) != match.group(1):
        raise ValueError(f"mirror catalog does not match its digest name: {relative}")


def check_released_catalogs(target_files: dict[Path, Path]) -> int:
    """Every catalog the mirror serves must still resolve to its manifests."""
    catalogs = [
        relative
        for relative in target_files
        if relative == LEGACY_CATALOG or DIGEST_CATALOG.fullmatch(relative.as_posix())
    ]
    for relative in catalogs:
        catalog = json.loads(target_files[relative].read_bytes())
        for entry in catalog["models"]:
            manifest = Path(entry["manifest_path"])
            if manifest not in target_files:
                raise ValueError(f"{relative} references a missing manifest: {manifest}")
            if sha256(target_files[manifest].read_bytes()) != entry["manifest_sha256"]:
                raise ValueError(f"{relative} manifest digest differs: {manifest}")
    return len(catalogs)


def files_below(root: Path) -> dict[Path, Path]:
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"registry metadata must not contain symlinks: {path}")
        if path.is_file():
            files[path.relative_to(root)] = path
    return files


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


def validate_mlx_alternatives(document: dict, catalog_names: set[str]) -> None:
    """Keep the supplemental MLX survey pinned and aligned with the catalog."""
    if not isinstance(document, dict):
        raise ValueError("MLX alternatives metadata must be an object")
    required = {"schema", "checked_at", "installable", "scope", "models"}
    if set(document) != required:
        raise ValueError("MLX alternatives metadata has unexpected fields")
    if document["schema"] != "openkind-mlx-alternatives/v1":
        raise ValueError("unsupported MLX alternatives schema")
    try:
        if datetime.date.fromisoformat(document["checked_at"]).isoformat() != document["checked_at"]:
            raise ValueError("MLX alternatives checked_at must be an ISO date")
    except (TypeError, ValueError) as error:
        raise ValueError("MLX alternatives checked_at must be an ISO date") from error
    if document["installable"] is not False:
        raise ValueError("MLX alternatives must remain non-installable")
    if not isinstance(document["scope"], str) or not document["scope"].strip():
        raise ValueError("MLX alternatives scope must be non-empty")
    if not isinstance(document["models"], list):
        raise ValueError("MLX alternatives models must be an array")

    observed_names = set()
    statuses = {"qualified", "unqualified", "none"}
    lead_fields = {
        "repository",
        "revision",
        "license",
        "base_model",
        "relationship",
        "notes",
    }
    for model in document["models"]:
        if not isinstance(model, dict) or set(model) != {
            "catalog_name",
            "openkind_mlx_status",
            "assessment",
            "leads",
        }:
            raise ValueError("MLX alternatives model has unexpected fields")
        name = model["catalog_name"]
        if not isinstance(name, str) or name not in catalog_names:
            raise ValueError(f"MLX alternatives references unknown catalog model: {name}")
        if name in observed_names:
            raise ValueError(f"duplicate MLX alternatives entry: {name}")
        observed_names.add(name)
        status = model["openkind_mlx_status"]
        if not isinstance(status, str) or status not in statuses:
            raise ValueError(f"invalid OpenKind MLX status for {name}")
        if not isinstance(model["assessment"], str) or not model["assessment"].strip():
            raise ValueError(f"missing MLX assessment for {name}")
        if not isinstance(model["leads"], list):
            raise ValueError(f"MLX leads must be an array for {name}")
        for lead in model["leads"]:
            if not isinstance(lead, dict) or set(lead) != lead_fields:
                raise ValueError(f"MLX lead has unexpected fields for {name}")
            repository = lead["repository"]
            if not isinstance(repository, str) or repository.count("/") != 1:
                raise ValueError(f"invalid Hugging Face repository for {name}")
            if not isinstance(lead["revision"], str) or not re.fullmatch(
                r"[0-9a-f]{40}", lead["revision"]
            ):
                raise ValueError(f"MLX lead revision must be a pinned commit for {name}")
            if lead["license"] is not None and not isinstance(lead["license"], str):
                raise ValueError(f"invalid MLX lead license for {name}")
            if lead["base_model"] is not None and not isinstance(lead["base_model"], str):
                raise ValueError(f"invalid MLX lead base model for {name}")
            for field in ("relationship", "notes"):
                if not isinstance(lead[field], str) or not lead[field].strip():
                    raise ValueError(f"missing MLX lead {field} for {name}")

    missing = catalog_names - observed_names
    if missing:
        raise ValueError(f"MLX alternatives omit catalog models: {sorted(missing)}")


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
    if any(DIGEST_CATALOG.fullmatch(path.as_posix()) for path in source_files):
        raise ValueError("source registry must not contain digest-named catalogs")
    catalog_bytes = (SOURCE / SOURCE_CATALOG).read_bytes()
    catalog_sha256 = sha256(catalog_bytes)
    # The source catalog publishes under its digest; the mirror's legacy
    # catalog.json is never overwritten, and older digest files stay.
    expected = {
        mirror_path(relative, catalog_sha256): source
        for relative, source in source_files.items()
    }
    target_files = files_below(target)
    if LEGACY_CATALOG not in target_files:
        raise ValueError("mirror lacks the frozen legacy catalog.json that v0.2.1 fetches")
    for relative in set(target_files) - set(expected):
        check_retained(relative, target_files[relative].read_bytes())
    if args.write:
        for relative, source in expected.items():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        target_files = files_below(target)
    if not set(expected) <= set(target_files):
        missing = sorted(map(str, set(expected) - set(target_files)))
        raise ValueError(f"mirror metadata is missing: {missing}")
    for relative, source in expected.items():
        if source.read_bytes() != target_files[relative].read_bytes():
            raise ValueError(f"mirror metadata differs: {relative}")
    released_catalogs = check_released_catalogs(target_files)

    catalog = json.loads(catalog_bytes)
    mlx_alternatives = json.loads((SOURCE / "mlx-alternatives.json").read_bytes())
    validate_mlx_alternatives(
        mlx_alternatives, {entry["name"] for entry in catalog["models"]}
    )
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
        for relative, source in [*expected.items(), (LEGACY_CATALOG, None)]:
            request = urllib.request.Request(
                PUBLIC_BASE + relative.as_posix(),
                headers={"User-Agent": "openkind-registry-sync"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            if source is None:
                check_retained(relative, data)
            elif data != source.read_bytes():
                raise ValueError(f"public HTTPS metadata differs: {relative}")

    print(
        f"registry mirror matches: {len(source_files)} metadata files "
        f"(catalog at catalogs/{catalog_sha256}.json), {released_catalogs} released "
        f"catalogs resolve, {checked_assets} pinned assets"
        + ("; public HTTPS matches" if args.remote else "")
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"registry sync failed: {error}", file=sys.stderr)
        sys.exit(1)
