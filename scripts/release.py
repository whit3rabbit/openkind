#!/usr/bin/env python3
"""Pinned runtime packaging and clean-host release checks. Never fetches models."""

import argparse
import hashlib
import functools
import http.server
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import threading
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "packaging/onnxruntime.json"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify(path, asset):
    if path.stat().st_size != asset["size_bytes"] or digest(path) != asset["sha256"]:
        raise ValueError(f"runtime archive integrity failure: {path.name}")


def extract(archive, destination):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                name = PurePosixPath(entry.filename.replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts or ":" in entry.filename:
                    raise ValueError("archive path escapes extraction directory")
            source.extractall(destination)
    else:
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")


def patch_cuda_manifest(directory):
    """Only remove redundant link declarations. CUDA APIs already use libloading."""
    manifest = directory / "Cargo.toml"
    original_bytes = manifest.read_bytes()
    original = original_bytes.decode()
    marker = '    "dynamic-linking",\n'
    if original.count(marker) != 1:
        raise ValueError("pinned candle-core CUDA manifest changed; review the override")
    manifest.write_text(original.replace(marker, ""), newline="\n")
    checksums_path = directory / ".cargo-checksum.json"
    checksums = json.loads(checksums_path.read_text())
    checksums["files"]["Cargo.toml"] = digest(manifest)
    checksums_path.write_text(json.dumps(checksums, sort_keys=True))
    return {"crate": "candle-core", "version": "0.8.0",
            "package_sha256": checksums["package"],
            "original_manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "patched_manifest_sha256": digest(manifest),
            "change": "remove cudarc dynamic-linking feature; no architecture source edits"}


def prepare_cuda(vendor_dir):
    vendor_dir = vendor_dir.resolve()
    config = subprocess.run(["cargo", "vendor", "--locked", "--versioned-dirs", str(vendor_dir)],
                            cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout
    override = patch_cuda_manifest(vendor_dir / "candle-core-0.8.0")
    config_path = ROOT / "target/release-cargo-config.toml"
    config_path.write_text(config)
    (ROOT / "target/cuda-build-override.json").write_text(json.dumps(override, indent=2) + "\n")


def bundle_runtime(target, stage, cache):
    manifest = json.loads(MANIFEST.read_text())
    asset = manifest["assets"][target]
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / asset["url"].rsplit("/", 1)[1]
    if not archive.exists():
        pending = archive.with_suffix(archive.suffix + ".partial")
        with urllib.request.urlopen(asset["url"], timeout=120) as source, pending.open("wb") as output:
            shutil.copyfileobj(source, output)
        verify(pending, asset)
        pending.replace(archive)
    verify(archive, asset)
    with tempfile.TemporaryDirectory(prefix="openkind-ort-") as directory:
        extracted = Path(directory)
        extract(archive, extracted)
        package, = [entry for entry in extracted.iterdir() if entry.is_dir()]
        # Windows DLL dependencies use the executable directory. Unix ORT
        # providers use their runtime module directory and relative rpaths.
        libraries = stage if "windows" in target else stage / "lib/onnxruntime"
        libraries.mkdir(parents=True, exist_ok=True)
        for library in (package / "lib").iterdir():
            if ".so" in library.name or library.suffix in (".dll", ".dylib"):
                if library.is_symlink():
                    (libraries / library.name).symlink_to(os.readlink(library))
                else:
                    shutil.copy2(library, libraries / library.name)
        notices = stage / "licenses/onnxruntime"
        notices.mkdir(parents=True)
        for name in ("LICENSE", "ThirdPartyNotices.txt", "Privacy.md"):
            source = package / name
            if source.is_file():
                shutil.copy2(source, notices / name)
        if not (notices / "LICENSE").is_file() or not (notices / "ThirdPartyNotices.txt").is_file():
            raise ValueError("runtime archive is missing required license notices")
    (stage / "onnxruntime.json").write_text(json.dumps({"schema_version": 1, "version": manifest["version"],
                                                       "target": target, **asset}, indent=2) + "\n")


def package(binary_dir, target, version, onnx, destination):
    version = version.removeprefix("v")
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]*", version):
        raise ValueError("invalid release version")
    destination.mkdir(parents=True, exist_ok=True)
    name = f"openkind-{version}-{target}" + ("-onnx" if onnx else "")
    extension = ".zip" if "windows" in target else ".tar.gz"
    archive = destination / (name + extension)
    with tempfile.TemporaryDirectory(prefix="openkind-release-") as directory:
        stage = Path(directory)
        executable_suffix = ".exe" if "windows" in target else ""
        for binary in ("openkind", "openkindd"):
            shutil.copy2(binary_dir / (binary + executable_suffix), stage)
        shutil.copy2(ROOT / "LICENSE", stage)
        if "linux-gnu" in target or "windows" in target:
            shutil.copy2(ROOT / "target/cuda-build-override.json", stage)
        if "aarch64-apple" in target:
            metallib = binary_dir / "mlx.metallib"
            if not metallib.is_file():
                raise ValueError("MLX release is missing mlx.metallib")
            shutil.copy2(metallib, stage)
        if onnx:
            bundle_runtime(target, stage, ROOT / "target/onnxruntime-downloads")
        if extension == ".zip":
            shutil.make_archive(str(destination / name), "zip", stage)
        else:
            with tarfile.open(archive, "w:gz") as output:
                for entry in sorted(stage.iterdir()):
                    output.add(entry, arcname=entry.name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest(archive)}  {archive.name}\n")
    print(archive)
    return archive


CUDA_IMPORT = re.compile(r"(?:lib)?(?:nvcuda|nvml|nvidia-ml|cuda|cudart|cublas(?:lt)?|curand|nvrtc|cudnn)(?:[._0-9-]|64)", re.I)


def audit(binary):
    if os.name == "nt":
        command = ["dumpbin", "/DEPENDENTS", str(binary)]
    elif os.uname().sysname == "Darwin":
        command = ["otool", "-L", str(binary)]
    else:
        command = ["readelf", "-d", str(binary)]
    output = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE).stdout
    for line in output.splitlines():
        # readelf prints rpaths too; only imported dependencies matter here.
        if command[0] == "readelf" and "NEEDED" not in line:
            continue
        if CUDA_IMPORT.search(line):
            raise ValueError(f"mandatory CUDA startup dependency: {line.strip()}")


def smoke(archive, expected_features):
    checksum = archive.with_suffix(archive.suffix + ".sha256").read_text().split()
    if len(checksum) != 2 or checksum[1] != archive.name or checksum[0] != digest(archive):
        raise ValueError("release archive checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="openkind relocated release ") as directory:
        stage = Path(directory)
        extract(archive, stage)
        suffix = ".exe" if os.name == "nt" else ""
        cli, daemon = (stage / (name + suffix) for name in ("openkind", "openkindd"))
        for binary in (cli, daemon):
            audit(binary)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("OPENKIND_", "OPENDECISION_", "OPENPICK_", "ORT_", "CUDA_", "CUDNN_", "MLX_"))
               and key not in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "RUST_LOG", "TYPESAFE_API_KEY", "OPENKINDD_BINARY")}
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                     if "cuda" not in p.lower() and "cudnn" not in p.lower())
        # The diagnostic environment aliases must not break hidden child probes.
        diagnostic_env = {**env, "OPENKIND_DIAGNOSTICS_JSON": "true"}
        report = json.loads(subprocess.check_output([str(cli), "doctor", "--json"], env=diagnostic_env,
                                                   cwd=tempfile.gettempdir(), timeout=150))
        statuses = {entry["backend"]: entry for entry in report["backends"]}
        for feature in ("cuda", "mlx-fp32", "onnx", "onnx-cuda"):
            if statuses[feature]["compiled"] != (feature in expected_features):
                raise ValueError(f"unexpected compiled support: {feature}")
        if not statuses["native-cpu"]["ready"]:
            raise ValueError("CPU reference runtime unavailable")
        if "onnx" in expected_features and not statuses["onnx"]["ready"]:
            raise ValueError("bundled ONNX Runtime failed required tiny-Gemm execution")
        if "mlx-fp32" in expected_features and not statuses["mlx-fp32"]["ready"]:
            raise ValueError("MLX runtime failed required GPU operation")
        if statuses["cuda"]["ready"] or statuses["onnx-cuda"]["ready"]:
            raise ValueError("clean-host test unexpectedly has an operational NVIDIA GPU")
        if "onnx" in expected_features:
            runtime = report["onnx_runtime"]
            if runtime is None or not Path(runtime).is_relative_to(stage):
                raise ValueError("diagnostics did not use the bundled runtime")
            override_env = {**env, "OPENKIND_ONNX_RUNTIME": str(stage / "missing-runtime")}
            explicit = json.loads(subprocess.check_output([str(cli), "doctor", "--json", "--onnx-runtime", runtime], env=override_env, timeout=150))
            if not next(entry for entry in explicit["backends"] if entry["backend"] == "onnx")["ready"]:
                raise ValueError("explicit runtime setting did not override the environment in probes")
            missing = json.loads(subprocess.check_output([str(cli), "doctor", "--json", "--onnx-runtime", str(stage / "missing-runtime")], env=env, timeout=150))
            if any(entry["ready"] for entry in missing["backends"] if entry["backend"].startswith("onnx")):
                raise ValueError("explicit missing runtime incorrectly fell back to bundle")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        process = subprocess.Popen([str(daemon), "--models", "jev-latest", "--http-addr", f"127.0.0.1:{port}", "--grpc-addr", "0"],
                                   env=env, cwd=tempfile.gettempdir(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise ValueError("mock daemon failed startup")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.1)
            request = json.loads((ROOT / "examples/01_noul.json").read_text())
            request["model"] = "jev-latest"
            call = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=json.dumps(request).encode(),
                                          headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(call, timeout=10) as response:
                result = json.load(response)
                if not result.get("answers"):
                    raise ValueError("mock request returned no answers")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print(f"verified relocated archive: {archive.name}")


def homebrew(archive):
    """Exercise the rendered formula against the local CI archive."""
    formula = (ROOT / "packaging/homebrew/openkind.rb").read_text()
    with tempfile.TemporaryDirectory(prefix="openkind brew ") as directory:
        root = Path(directory)
        shutil.copy2(archive, root / archive.name)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/{archive.name}"
            formula = re.sub(r'url "[^"\n]+"', f'url "{url}"', formula)
            formula = re.sub(r'@@SHA_[A-Z_]+@@', digest(archive), formula).replace("@@VERSION@@", "0.0.0")
            subprocess.run(["brew", "tap-new", "openkind/release-ci"], check=True)
            repository = Path(subprocess.check_output(["brew", "--repository", "openkind/release-ci"], text=True).strip())
            (repository / "Formula/openkind.rb").write_text(formula)
            subprocess.run(["brew", "install", "--formula", "openkind/release-ci/openkind"], check=True)
            subprocess.run(["brew", "test", "openkind/release-ci/openkind"], check=True)
        finally:
            server.shutdown()
            server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cuda = commands.add_parser("prepare-cuda")
    cuda.add_argument("--vendor-dir", type=Path, default=ROOT / "target/release-vendor")
    pack = commands.add_parser("package")
    pack.add_argument("--bin-dir", type=Path, required=True)
    pack.add_argument("--target", required=True)
    pack.add_argument("--version", required=True)
    pack.add_argument("--onnx", action="store_true")
    pack.add_argument("--destination", type=Path, default=ROOT / "dist")
    check = commands.add_parser("smoke")
    check.add_argument("archive", type=Path)
    check.add_argument("--features", default="")
    brew = commands.add_parser("homebrew")
    brew.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-cuda":
        prepare_cuda(args.vendor_dir)
    elif args.command == "package":
        package(args.bin_dir, args.target, args.version, args.onnx, args.destination)
    elif args.command == "homebrew":
        homebrew(args.archive.resolve())
    else:
        smoke(args.archive.resolve(), set(filter(None, args.features.split(","))))


if __name__ == "__main__":
    main()
