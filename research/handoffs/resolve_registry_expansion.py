#!/usr/bin/env python3
"""Re-resolve every REGISTRY_EXPANSION_2026-10-12.md pin against the Hub API.

Writes one JSON evidence file per repository under
research/handoffs/registry-expansion-2026-10-12/<group>/<safe-name>.json with
the observed sha, lastModified, private/gated flags, license, card data, and
the full sibling file list with sizes and LFS SHA-256 digests.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "registry-expansion-2026-10-12"

TIER1 = [
    "vllm-sr/d3",
    "Mapika/decider-31b",
    "FINAL-Bench/Darwin-27B-ZTC",
    "perplexity-ai/pplx-decider-v1.1-27b",
    "sthanika-ai/Sieve-27B",
    "jaredpalmer/kev-27b",
    "shisa-ai/shisa-de-2",
    "vllm-sr/d3-flash",
    "vllm-sr/d3-mini",
    "vllm-sr/d3-nano",
    "vllm-sr/d3-lite",
    "vllm-sr/d3-edge",
]
TIER2 = [
    "Maincode/matilda-jev-v1.5",
    "shgao/rsi-jev-v6.1-vl-27b",
    "shgao/rsi-jev-v6.1-vl-4b",
    "torchcast-ai/torchcast-decision-27b",
    "torchcast-ai/torchcast-decision-12b",
    "torchcast-ai/torchcast-decision-4b",
    "startlux-models/StartLux-Decision-v1-27B",
    "startlux-models/StartLux-Decision-v1-9B",
    "startlux-models/StartLux-Decision-v1-4B",
    "jsaurabh/qwen-decision-27b-noncausal-lora",
    "MISA-core/linne-fern-1-0",
    "chinhnc/Quyet-1.0-Large",
    "juspay/Xor-26B-A4B",
    "PixilabAI/Blink-v0.3-26B-A4B-NVFP4",
    "plantwaterer/garas",
    "vllm-sr/Decision-2.0-Vega-27B",
    "vllm-sr/Decision-2.0-Lux-9B",
    "vllm-sr/Decision-2.0-Nox-4B",
    "sthanika-ai/Sieve-9B-Plus",
    "sthanika-ai/Sieve-4B",
    "bespokelabs/Bespoke-Nimble-9B-v3",
    "Cloudflare/clef",
    "Cloudflare/clef-flash",
]
BASES = [
    "Qwen/Qwen3.8-27B",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B",
    # Unresolved in the handoff; needed by d3-nano/lite/edge and Sieve 2B Plus.
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-0.8B",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-12B-it",
]
MLX = [
    "mlx-community/pplx-decider-v1.1-27b-4bit",
    "ahmedandaloes/d3-mlx",
    "ahmedandaloes/d3-flash-mlx",
    "thoddnn/d3-flash-q4",
    "ahmedandaloes/d3-mini-mlx",
    "thoddnn/d3-mini-q4",
    "Jakevin/d3-mini-ternary-mlx",
    "ahmedandaloes/d3-nano-mlx",
    "thoddnn/d3-nano-q4",
    "Jakevin/d3-nano-ternary-mlx",
    "Jakevin/d3-nano-ternary-3bit-mlx",
    "ahmedandaloes/d3-lite-mlx",
    "ahmedandaloes/d3-edge-mlx",
    "shgao/rsi-jev-v6.1-vl-4b-mlx-8bit",
    "shgao/rsi-jev-v6.1-vl-4b-mlx-4bit",
    "moritalous/Decision-2.0-Lux-9B-MLX-4bit",
    "moritalous/Decision-2.0-Nox-4B-MLX-4bit",
    "mlx-community/clef-4bit",
    "mlx-community/clef-8bit",
    "mlx-community/clef-flash-4bit",
    "mlx-community/clef-flash-8bit",
]

GROUPS = {"tier1": TIER1, "tier2": TIER2, "bases": BASES, "mlx": MLX}


def get(url: str, binary: bool = False):
    request = urllib.request.Request(url, headers={"User-Agent": "openkind-registry"})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    return data if binary else json.loads(data)


def safe_name(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", repo)


def resolve(repo: str) -> dict:
    meta = get(f"https://huggingface.co/api/models/{repo}?blobs=true")
    return meta


def main() -> int:
    only = set(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    for group, repos in GROUPS.items():
        if only and group not in only:
            continue
        group_dir = OUT / group
        group_dir.mkdir(exist_ok=True)
        for repo in repos:
            target = group_dir / f"{safe_name(repo)}.json"
            if target.exists():
                print(f"skip {repo} (cached)", flush=True)
                continue
            try:
                meta = resolve(repo)
            except urllib.error.HTTPError as error:
                failures.append((repo, f"HTTP {error.code}"))
                print(f"FAIL {repo}: HTTP {error.code}", flush=True)
                continue
            except Exception as error:  # noqa: BLE001
                failures.append((repo, repr(error)))
                print(f"FAIL {repo}: {error!r}", flush=True)
                continue
            record = {
                "repository": repo,
                "resolved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sha": meta.get("sha"),
                "lastModified": meta.get("lastModified"),
                "private": meta.get("private"),
                "gated": meta.get("gated"),
                "license": (meta.get("cardData") or {}).get("license"),
                "pipeline_tag": meta.get("pipeline_tag"),
                "tags": meta.get("tags"),
                "config": meta.get("config"),
                "siblings": [
                    {
                        "path": s.get("rfilename"),
                        "size": s.get("size"),
                        "lfs_sha256": (s.get("lfs") or {}).get("sha256"),
                        "lfs_size": (s.get("lfs") or {}).get("size"),
                    }
                    for s in meta.get("siblings", [])
                ],
            }
            target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(f"ok {repo} @ {meta.get('sha')}", flush=True)
    if failures:
        print("\nfailures:")
        for repo, why in failures:
            print(f"  {repo}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
