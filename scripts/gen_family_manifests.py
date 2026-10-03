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
CLEF_FLASH = ("Cloudflare/clef-flash", "17f0b0ad64efb65d273590632833508766b2aae6")
CLEF = ("Cloudflare/clef", "2f3de3dd85f379784083b0814d997ab627200f0c")
CLEF_FLASH_GGUF = ("bartowski/Cloudflare_clef-flash-GGUF", "d7f376ea88c05e7bb1014dd5351a93df9dd8029e")
CLEF_GGUF = ("bartowski/Cloudflare_clef-GGUF", "e306f00c6c85da175dfb8de952ebb872087426a7")

TOKENIZER_CLEF = (
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    19989325,
)

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
        "aliases": ["jevk5:4b"],
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
    },    {
        "name": "clef-flash:dfe12a21a5c9dd5b2fb1",
        "aliases": ["clef:flash"],
        "profile_id": "dfe12a21a5c9dd5b2fb1",
        "loader_id": "clef-flash",
        "description": "Cloudflare Clef-Flash 9B joint-schema decision model, BF16 safetensors oracle (Apache-2.0); reference-parity readout, research status",
        "release_date": "2026-10-01",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(CLEF_FLASH, "config.json", "checkpoint/config.json",
                        "66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238", 2832),
            hf_artifact(CLEF_FLASH, "tokenizer.json", "checkpoint/tokenizer.json", *TOKENIZER_CLEF),
            hf_artifact(CLEF_FLASH, "joint_head.safetensors", "checkpoint/joint_head.safetensors",
                        "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0", 243538016),
            hf_artifact(CLEF_FLASH, "model.safetensors.index.json", "checkpoint/model.safetensors.index.json",
                        "941305ff9f77551e145a6cea976ef456cd5cb208cbece99f168376c752fcf96c", 69253),
            hf_artifact(CLEF_FLASH, "model-00001-of-00004.safetensors", "checkpoint/model-00001-of-00004.safetensors",
                        "8b45a8e968141cdcc58fb71c9adfc258e2c77b5f062bc636c1fd5bc5d916b565", 4942706120),
            hf_artifact(CLEF_FLASH, "model-00002-of-00004.safetensors", "checkpoint/model-00002-of-00004.safetensors",
                        "7590856c713eed844a2dcf48e6c43c4de165b788bc3f80e328311183cdbc7db8", 4987757928),
            hf_artifact(CLEF_FLASH, "model-00003-of-00004.safetensors", "checkpoint/model-00003-of-00004.safetensors",
                        "e6eac2467952c33361ed7dcb3c7959d1086bbe57201cd3749c3d769fdc17fe63", 4954810240),
            hf_artifact(CLEF_FLASH, "model-00004-of-00004.safetensors", "checkpoint/model-00004-of-00004.safetensors",
                        "9fcecc6556b39171238373a465f409794b7f821fb4cd1e6459e3a9c0fe317af7", 3934446832),
        ],
    },
    {
        "name": "clef-flash-gguf:c330d9ee7e9cc658ad45",
        "aliases": ["clef:flash-gguf"],
        "profile_id": "c330d9ee7e9cc658ad45",
        "loader_id": "clef-flash-gguf",
        "description": "Cloudflare Clef-Flash 9B Q4_K_M GGUF backbone with the official BF16 joint schema head (Apache-2.0); reference-parity readout, research status",
        "release_date": "2026-10-01",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(CLEF_FLASH, "config.json", "checkpoint/config.json",
                        "66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238", 2832),
            hf_artifact(CLEF_FLASH, "tokenizer.json", "checkpoint/tokenizer.json", *TOKENIZER_CLEF),
            hf_artifact(CLEF_FLASH, "joint_head.safetensors", "checkpoint/joint_head.safetensors",
                        "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0", 243538016),
            hf_artifact(CLEF_FLASH_GGUF, "Cloudflare_clef-flash-Q4_K_M.gguf", "checkpoint/Cloudflare_clef-flash-Q4_K_M.gguf",
                        "45f803cbcb6144784653bc31cde957e0d184d963a5198d423dc589a79e178d45", 5841052992),
        ],
    },
    {
        "name": "clef-27b-gguf:48cb5634b4a258de5a6b",
        "aliases": ["clef:27b"],
        "profile_id": "48cb5634b4a258de5a6b",
        "loader_id": "clef-27b-gguf",
        "description": "Cloudflare Clef 27B Q4_K_M GGUF backbone with the official BF16 joint schema head (Apache-2.0); reference-parity readout, research status",
        "release_date": "2026-10-01",
        "question_types": ["choice", "score", "noul"],
        "artifacts": [
            hf_artifact(CLEF, "config.json", "checkpoint/config.json",
                        "c42e88892bd3fd84e8276b2ad90df58c1c3b797676ea161035006a72ad468c58", 3688),
            hf_artifact(CLEF, "tokenizer.json", "checkpoint/tokenizer.json", *TOKENIZER_CLEF),
            hf_artifact(CLEF, "joint_head.safetensors", "checkpoint/joint_head.safetensors",
                        "a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953", 256125024),
            hf_artifact(CLEF_GGUF, "Cloudflare_clef-Q4_K_M.gguf", "checkpoint/Cloudflare_clef-Q4_K_M.gguf",
                        "6a03997c1fe1b22580d15b540535f61b3da79e261766cac7104febc8e4651849", 17203416256),
        ],
    },
]

# Maximum total input sequence length in tokens each pinned checkpoint's own
# configuration declares (`max_position_embeddings` in Hugging Face configs,
# `context_length` in GGUF metadata); emitted as the catalog entry's
# `context_limit` and mirrored by docs/MODELS.md.
CONTEXT_LIMITS = {
    "decoder-logit-letter:5492c97dfcdaf3fe9439": 32768,
    "encoder-nli:1041a4c362338a61b820": 512,
    "encoder-instruct-label:9fd68313a5606eca42f2": 8192,
    "decoder-logit-llm:465963d705b6f35d6208": 32768,
    "schema-scorer:5a7350af556f0ee66566": 512,
    "qwen3guard:0fcf416cab16d94f933d": 8192,
    "kev:39d88c11faeb4ac165fa": 32768,
    "decoder-logit-qwen35:415bcf4a064e6dadcf85": 262144,
    "winnow:4dff8c5b03cfbf680db6": 32768,
    "clef-flash:dfe12a21a5c9dd5b2fb1": 262144,
    "clef-flash-gguf:c330d9ee7e9cc658ad45": 262144,
    "clef-27b-gguf:48cb5634b4a258de5a6b": 262144,
}



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
        entry = {
            "name": model["name"],
            "profile_id": model["profile_id"],
            "loader_id": model["loader_id"],
            "description": model["description"],
            "context_limit": CONTEXT_LIMITS[model["name"]],
            "support_status": "rust-loadable",
            "manifest_path": f"manifests/{manifest_name}",
            "manifest_sha256": sha256_bytes(manifest_bytes),
        }
        if model.get("aliases"):
            entry["aliases"] = model["aliases"]
        catalog["models"].append(entry)
        print(f"wrote {manifest_name}")
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"catalog now lists {len(catalog['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
