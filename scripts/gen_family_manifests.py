#!/usr/bin/env python3
"""Generate registry/v1 manifests for the surveyed-family catalog profiles.

Every digest and size was verified against the pinned source revisions:
LFS objects through the Hugging Face tree API (the LFS OID is the SHA-256),
small files through the loader-pinned digests in
crates/openkind-backends/src/families/*/mod.rs, and the two mirror assets
through the public registry asset commit.
"""

import hashlib
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "v1"
MIRROR = "whit3rabbit/openkind-model-registry"
ASSET_COMMIT = "8ed6c7b6444ecf0ab0c0c37612ea22e33dd9996b"

QWEN25 = ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")
QWEN25_GGUF = ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "9217f5db79a29953eb74d5343926648285ec7e67")
DISTILBERT_MNLI = ("typeform/distilbert-base-uncased-mnli", "cfa538a0fddbbd978fefe8966c1aeff7ad409c90")
GLICLASS = ("knowledgator/gliclass-modern-base-v3.0", "ac369222ca4375ca66ebaf7fb5220f223514c035")
MINILM = ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a")
QWEN3GUARD = ("Qwen/Qwen3Guard-Stream-0.6B", "419364a715de9840d47b1457982f64ff37f90ed4")
KEV = ("jaredpalmer/kev-0.6b", "dece6dba8d43f0f7ded45e9f5b9df12474d90843")
QWEN3_BASE = ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
JEVK5 = ("alibiserikbay/JevK5", "c4f7fdb3aeab5582336406e78d3bef11bf98833d")

TOKENIZER_QWEN25 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    7031645,
)
TOKENIZER_QWEN3 = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    11422654,
)


def hf(repo_rev, path, sha, size):
    return {
        "path": path,
        "size": size,
        "sha256": sha,
        "source": {
            "kind": "huggingface",
            "repository": repo_rev[0],
            "revision": repo_rev[1],
            "path": path.split("/", 1)[1] if "/" in path else path,
        },
    }


def mirror_asset(install_path, source_path, sha, size):
    return {
        "path": install_path,
        "size": size,
        "sha256": sha,
        "source": {
            "kind": "github",
            "repository": MIRROR,
            "revision": ASSET_COMMIT,
            "path": source_path,
        },
    }


def hf_artifact(repo_rev, source_path, install_path, sha, size):
    artifact = hf(repo_rev, install_path, sha, size)
    artifact["source"]["path"] = source_path
    return artifact


MODELS = [
    {
        "name": "decoder-logit-letter:5492c97dfcdaf3fe9439",
        "profile_id": "5492c97dfcdaf3fe9439",
        "loader_id": "decoder-logit-letter",
        "description": "Pinned Qwen2.5-0.5B-Instruct letter-logit decision decoder (Apache-2.0); prototype readout, research status",
        "release_date": "2026-09-26",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(QWEN25, "config.json", "checkpoint/config.json",
                        "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45", 659),
            hf_artifact(QWEN25, "tokenizer.json", "checkpoint/tokenizer.json", *TOKENIZER_QWEN25),
            hf_artifact(QWEN25, "model.safetensors", "checkpoint/model.safetensors",
                        "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe", 988097824),
        ],
    },
    {
        "name": "encoder-nli:1041a4c362338a61b820",
        "profile_id": "1041a4c362338a61b820",
        "loader_id": "encoder-nli",
        "description": "Pinned DistilBERT MNLI zero-shot entailment decision encoder (Apache-2.0); prototype readout, research status",
        "release_date": "2026-09-26",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(DISTILBERT_MNLI, "config.json", "checkpoint/config.json",
                        "d6d658b44d7260410d8aa3f6cd585016656dfe57dd57855c070f86ddcb257385", 776),
            hf_artifact(DISTILBERT_MNLI, "vocab.txt", "checkpoint/vocab.txt",
                        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3", 231508),
            hf_artifact(DISTILBERT_MNLI, "model.safetensors", "checkpoint/model.safetensors",
                        "16d47e5948c7076ecfe8b9d343c0d1474cd600f18405c40cfcf183605787af41", 267835640),
        ],
    },
    {
        "name": "encoder-instruct-label:9fd68313a5606eca42f2",
        "profile_id": "9fd68313a5606eca42f2",
        "loader_id": "encoder-instruct-label",
        "description": "Pinned GLiClass uni-encoder over ModernBERT-base with label markers (Apache-2.0); prototype readout, research status",
        "release_date": "2026-09-26",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(GLICLASS, "config.json", "checkpoint/config.json",
                        "f28b93a1ab70736f6b0f38d9da058bdabc045238cd8c50bb6c334197c797d1e8", 3898),
            hf_artifact(GLICLASS, "tokenizer.json", "checkpoint/tokenizer.json",
                        "7c1979be5ac04a6681dbfbb98a2b01b176883a298f2a7240c93048b244118a01", 3583595),
            hf_artifact(GLICLASS, "model.safetensors", "checkpoint/model.safetensors",
                        "b83af831dc664ff552fc52054f9ec0def73b83d81ce554ae1c91b86ade7cd369", 605529372),
        ],
    },
    {
        "name": "decoder-logit-llm:465963d705b6f35d6208",
        "profile_id": "465963d705b6f35d6208",
        "loader_id": "decoder-logit-llm",
        "description": "Pinned Qwen2.5-0.5B-Instruct GGUF q8_0 label-logit decoder (Apache-2.0); prototype readout, research status",
        "release_date": "2026-09-26",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(QWEN25, "tokenizer.json", "checkpoint/tokenizer.json", *TOKENIZER_QWEN25),
            hf_artifact(QWEN25_GGUF, "qwen2.5-0.5b-instruct-q8_0.gguf", "checkpoint/qwen2.5-0.5b-instruct-q8_0.gguf",
                        "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e", 675710816),
        ],
    },
    {
        "name": "schema-scorer:5a7350af556f0ee66566",
        "profile_id": "5a7350af556f0ee66566",
        "loader_id": "schema-scorer",
        "description": "Pinned MS MARCO MiniLM-L-6 cross-encoder with single-logit schema score (Apache-2.0); prototype readout, research status",
        "release_date": "2026-09-26",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(MINILM, "config.json", "checkpoint/config.json",
                        "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc", 794),
            hf_artifact(MINILM, "tokenizer.json", "checkpoint/tokenizer.json",
                        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66", 711396),
            hf_artifact(MINILM, "model.safetensors", "checkpoint/model.safetensors",
                        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae", 90870598),
        ],
    },
    {
        "name": "qwen3guard:0fcf416cab16d94f933d",
        "profile_id": "0fcf416cab16d94f933d",
        "loader_id": "qwen3guard",
        "description": "Pinned Qwen3Guard-Stream-0.6B safety-verdict decision decoder (Apache-2.0); prototype readout, research status",
        "release_date": "2026-09-26",
        "question_types": ["choice", "noul"],
        "artifacts": [
            hf_artifact(QWEN3GUARD, "config.json", "checkpoint/config.json",
                        "b3bcfee86ed04c86e3c92000a2768d3c879564ac45e7e97909bce2e5767503cd", 1509),
            hf_artifact(QWEN3GUARD, "tokenizer.json", "checkpoint/tokenizer.json", *TOKENIZER_QWEN3),
            hf_artifact(QWEN3GUARD, "model.safetensors", "checkpoint/model.safetensors",
                        "e0a2eac6cc79cca5bf35bf8fb356f94c333dd9715f4f0dd58883b01f2fe33419", 1194258680),
        ],
    },
    {
        "name": "kev:39d88c11faeb4ac165fa",
        "profile_id": "39d88c11faeb4ac165fa",
        "loader_id": "kev",
        "description": "Pinned kev-0.6b LoRA decision decoder with pointer head over Qwen3-0.6B-Base (Apache-2.0); prototype readout, research status",
        "release_date": "2026-09-26",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(KEV, "adapter_model.safetensors", "adapter/adapter_model.safetensors",
                        "deaab63b61f95d628503831e8b336e1c3e334b9a42d3530d979a5149b429c492", 40419816),
            hf_artifact(KEV, "tokenizer.json", "adapter/tokenizer.json", *TOKENIZER_QWEN3),
            mirror_asset("adapter/head.safetensors",
                         "assets/kev-0.6b/39d88c11faeb4ac165fa/head.safetensors",
                         "006201e630a8cdef3386e803e38d6374786c557cf7b5d8637a4774b52f075202", 2099504),
            hf_artifact(QWEN3_BASE, "model.safetensors", "base/model.safetensors",
                        "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba", 1192135096),
            hf_artifact(QWEN3_BASE, "config.json", "base/config.json",
                        "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59", 727),
        ],
    },
    {
        "name": "decoder-logit-qwen35:415bcf4a064e6dadcf85",
        "profile_id": "415bcf4a064e6dadcf85",
        "loader_id": "decoder-logit-qwen35",
        "description": "Pinned JevK5 merged Qwen3.5-4B letter-logit decision decoder (Apache-2.0); prototype readout, research status",
        "release_date": "2026-09-27",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(JEVK5, "config.json", "checkpoint/config.json",
                        "63f47812d0f11118e4d252d2b3ad488707eb9287a11589f4fd382a1d31182724", 1979),
            hf_artifact(JEVK5, "tokenizer.json", "checkpoint/tokenizer.json",
                        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523", 19989325),
            hf_artifact(JEVK5, "jevk5_config.json", "checkpoint/jevk5_config.json",
                        "0d689fd13d15dc962265e2ae10b56359706ab5d05ad24e00e6334e4c19cf83d2", 52),
            hf_artifact(JEVK5, "model.safetensors", "checkpoint/model.safetensors",
                        "13824e47f2e40fe052f06943976cf742cb366ba305741a111e75a8ebae907a9c", 8411558400),
        ],
    },
    {
        "name": "winnow:4dff8c5b03cfbf680db6",
        "profile_id": "4dff8c5b03cfbf680db6",
        "loader_id": "winnow",
        "description": "Pinned in-house Winnow script-aware router over Qwen2.5-0.5B-Instruct (Apache-2.0 base); prototype router, research status",
        "release_date": "2026-09-26",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(QWEN25, "model.safetensors", "checkpoint/model.safetensors",
                        "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe", 988097824),
            hf_artifact(QWEN25, "tokenizer.json", "checkpoint/tokenizer.json", *TOKENIZER_QWEN25),
            mirror_asset("adapter.safetensors",
                         "assets/winnow/4dff8c5b03cfbf680db6/adapters.safetensors",
                         "f4dbf4dae0974b0afae79ceb73487d24a994520fd53365fcb20a28e157b49d83", 5877295),
        ],
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    manifests_dir = REGISTRY / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    catalog_path = REGISTRY / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    known = {entry["name"] for entry in catalog["models"]}
    for model in MODELS:
        if model["name"] in known:
            catalog["models"] = [
                entry for entry in catalog["models"] if entry["name"] != model["name"]
            ]
        manifest = {
            "schema": "openkind-model/v1",
            "name": model["name"],
            "profile_id": model["profile_id"],
            "loader_id": model["loader_id"],
            "description": model["description"],
            "release_date": model["release_date"],
            "support_status": "rust-loadable",
            "question_types": model["question_types"],
            "artifacts": model["artifacts"],
        }
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
        manifest_name = model["name"].replace(":", "-") + ".json"
        (manifests_dir / manifest_name).write_bytes(manifest_bytes)
        catalog["models"].append(
            {
                "name": model["name"],
                "profile_id": model["profile_id"],
                "loader_id": model["loader_id"],
                "description": model["description"],
                "support_status": "rust-loadable",
                "manifest_path": f"manifests/{manifest_name}",
                "manifest_sha256": sha256_bytes(manifest_bytes),
            }
        )
        print(f"wrote {manifest_name}")
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"catalog now lists {len(catalog['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
