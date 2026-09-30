#!/usr/bin/env python3
"""Acquire and run the pinned external Jeeves comparator, outside openkindd."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

LOCK_PATH = Path(__file__).with_name("jeeves-artifacts.json")
SOURCE_URL = "https://github.com/PostHog/jeeves.git"


def verify_weights(root, lock):
    for artifact in lock["files"]:
        path = root / artifact["path"]
        if path.stat().st_size != artifact["size"]:
            raise ValueError(f"size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")


def verify_source(root, lock):
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != lock["source_revision"]:
        raise ValueError(f"source revision mismatch: {root}")
    # Local changes could alter imports even when HEAD is pinned.
    changes = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    if changes:
        raise ValueError(f"source checkout has local changes: {root}")


def serve_command(source, weights, args):
    return [
        sys.executable, "-m", "inference.serve",
        "--model", str(weights),
        "--drafter", str(weights / "drafter_k4.safetensors"),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-think", str(args.max_think),
        "--fp8" if args.fp8 else "--no-fp8",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "verify", "serve", "plan"])
    parser.add_argument("--root", type=Path, required=True,
                        help="persistent directory for source/ and weights/")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--max-think", type=int, default=768)
    parser.add_argument("--fp8", action="store_true", help="opt into upstream FP8 kernels")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 <= args.max_think <= 2560:
        parser.error("port must be 1..65535 and max-think 0..2560")
    lock = json.loads(LOCK_PATH.read_text())
    source, weights = args.root.resolve() / "source", args.root.resolve() / "weights"
    if args.action == "plan":
        print(json.dumps({"source": SOURCE_URL, "source_revision": lock["source_revision"],
                          "weights_revision": lock["revision"],
                          "download_bytes": sum(f["size"] for f in lock["files"]),
                          "cwd": str(source), "serve": serve_command(source, weights, args)}, indent=2))
        return
    if args.action == "download":
        # Import before cloning so a missing dependency causes no partial setup.
        from huggingface_hub import snapshot_download

        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            subprocess.run(["git", "clone", "--no-checkout", SOURCE_URL, str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "checkout", "--detach",
                            lock["source_revision"]], check=True)
        verify_source(source, lock)
        snapshot_download(lock["repository"], revision=lock["revision"],
                          allow_patterns=[f["path"] for f in lock["files"]], local_dir=weights)
    verify_source(source, lock)
    verify_weights(weights, lock)
    if args.action == "serve":
        # This entry point is a CUDA comparator, never a CPU/MLX fallback.
        import torch

        if not torch.cuda.is_available():
            raise ValueError("Jeeves upstream serving requires a CUDA GPU")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        subprocess.run(serve_command(source, weights, args), cwd=source, env=env, check=True)
    else:
        print(f"Verified Jeeves source and {len(lock['files'])} weight files in {args.root}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
