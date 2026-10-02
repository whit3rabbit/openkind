#!/usr/bin/env python3
"""Finish profiles added after the main local benchmark campaign started."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
from pathlib import Path

import run_dataset_suite as suite


SUPPLEMENTAL_LOADERS = (
    "von",
    "winnow-e4b",
    "strands-decider-2b",
    "clef-flash",
    "clef-flash-gguf",
    "clef-27b-gguf",
)
SUPPLEMENTAL_COMMIT = "4f49048714c056cf97757de1f5fb2ea126fc87f6"
LOCAL_CHECKPOINTS = {
    "winnow-e4b": Path.home() / ".cache/openkind/winnow-e4b-1b257e8fa80b270a62338362a8b35e37f7890273",
    "strands-decider-2b": Path.home() / ".cache/openkind/strands-decider-2b",
    "clef-flash": Path.home() / ".cache/openkind/models/clef-flash",
    "clef-flash-gguf": Path.home() / ".cache/openkind/models/clef-flash-gguf-q4",
    "clef-27b-gguf": Path.home() / ".cache/openkind/models/clef-27b-gguf-q4",
}


def append_unique(items: list[dict[str, str | int]], item: dict[str, str | int]) -> None:
    if item not in items:
        items.append(item)


def persist(
    completed: list[dict[str, str | int]],
    performance: list[dict[str, str | int]],
    skipped: list[dict[str, str]],
    failures: list[dict[str, str]],
    profile: str | None,
    phase: str,
    installed: str | None,
) -> None:
    suite.write_status(
        completed=completed,
        performance=performance,
        skipped=skipped,
        failures=failures,
        current_profile=profile,
        current_phase=phase,
        installed_profile=installed,
        commit=SUPPLEMENTAL_COMMIT,
    )


def classify_failure(log_path: Path) -> str | None:
    excerpt = "\n".join(log_path.read_text().splitlines()[-20:]).lower()
    if "candidate" in excerpt and any(
        token in excerpt
        for token in ("max", "maximum", "at most", "supports", "covers only", "capacity")
    ):
        return "input exceeds engine candidate-count capacity"
    if any(
        token in excerpt
        for token in ("rendered prompt length", "sequence length", "token length")
    ) and any(token in excerpt for token in ("exceeds", "maximum", "max_length")):
        return "input exceeds engine rendered-token length capacity"
    return None


def artifact_cache_path(loader: str, root: Path, artifact: dict) -> Path:
    relative = Path(artifact["path"])
    if loader.startswith("clef-"):
        relative = Path(*relative.parts[1:])
    elif loader == "winnow-e4b":
        relative = Path(relative.name)
    return root / relative


def verified_local_checkpoint(loader: str, manifest: dict) -> Path | None:
    root = LOCAL_CHECKPOINTS.get(loader)
    if root is None:
        return None
    if not root.is_dir():
        return None

    for artifact in manifest.get("artifacts", []):
        path = artifact_cache_path(loader, root, artifact)
        if not path.is_file() or path.stat().st_size != artifact["size"]:
            raise RuntimeError(f"pre-existing checkpoint cache failed size check: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact["sha256"]:
            raise RuntimeError(f"pre-existing checkpoint cache failed SHA-256: {path}")
    return root


def main() -> int:
    status_path = suite.STATUS_PATH
    status = json.loads(status_path.read_text())
    completed = list(status.get("completed_evaluations", []))
    performance = list(status.get("completed_performance_runs", []))
    skipped = list(status.get("skipped_combinations", []))
    failures = list(status.get("failures", []))

    catalog_root = suite.REPO / "registry/v1"
    catalog = json.loads((catalog_root / "catalog.json").read_text())
    entries = {
        entry["loader_id"]: entry
        for entry in catalog["models"]
        if entry["loader_id"] in SUPPLEMENTAL_LOADERS
    }
    datasets = json.loads(
        (suite.REPO / "crates/openkind-datasets/registry/v1/datasets.json").read_text()
    )["datasets"]
    env = os.environ.copy()
    env["OPENKIND_MODELS_DIR"] = str(suite.MODEL_STORE)
    suite.DATASET_STORE.mkdir(parents=True, exist_ok=True)

    for loader in SUPPLEMENTAL_LOADERS:
        entry = entries.get(loader)
        if entry is None:
            failures.append({"profile": loader, "phase": "catalog", "reason": "profile missing from current catalog"})
            continue

        name = entry["name"]
        manifest = json.loads((catalog_root / entry["manifest_path"]).read_text())
        install_root = suite.MODEL_STORE / "models" / name
        root = install_root
        temporary_install = True
        engine = suite.ENGINE_BY_LOADER[loader]
        log_path = suite.CAMPAIGN / "logs" / f"dataset-{loader}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        model_env = env.copy()

        with log_path.open("a") as log:
            log.write(f"\nsupplemental profile={name}\nloader={loader}\ncommit={SUPPLEMENTAL_COMMIT}\nhost={suite.HOST}\n\n")
            try:
                if loader in LOCAL_CHECKPOINTS:
                    print(f"[verify cached checkpoint] {loader}", flush=True)
                cached_root = verified_local_checkpoint(loader, manifest)
                if cached_root is not None:
                    root = cached_root
                    temporary_install = False
                    log.write(f"using pre-existing local checkpoint cache after manifest size and SHA-256 verification: {root}\n")
                source_method = "pre-existing-cache-sha256-verified" if not temporary_install else "campaign-pull"
                if not (root / "manifest.json").is_file():
                    if temporary_install:
                        persist(completed, performance, skipped, failures, name, "pull", name)
                        if suite.run([str(suite.OPENKIND), "pull", name], model_env, log, f"pull {name}"):
                            failures.append({"profile": name, "phase": "pull", "reason": "model pull failed"})
                            continue

                types = set(manifest.get("question_types", []))
                for dataset in datasets:
                    primitive = dataset.get("primitives", [""])[0]
                    if primitive not in types:
                        append_unique(skipped, {"profile": name, "dataset": dataset["name"], "reason": f"profile does not declare {primitive}"})
                        continue

                    output = suite.CAMPAIGN / "quality" / loader / dataset["name"]
                    tune = primitive == "noul"
                    if suite.evaluation_complete(output, engine, 50, tune):
                        if not any(x.get("profile") == name and x.get("dataset") == dataset["name"] for x in completed):
                            completed.append({"profile": name, "dataset": dataset["name"], "rows": 50, "commit": SUPPLEMENTAL_COMMIT, "source": source_method})
                        continue

                    command = [
                        str(suite.BENCH), "dataset", "eval", dataset["name"],
                        "--datasets-dir", str(suite.DATASET_STORE),
                        "--limit", "50", "--engine", engine,
                        "--host", suite.HOST, "--commit", SUPPLEMENTAL_COMMIT,
                        "--output-dir", str(output), "--pretty",
                        *(
                            ["--model-root", str(root)]
                            if loader in {"winnow-e4b", "clef-flash", "clef-flash-gguf", "clef-27b-gguf"} and not temporary_install
                            else suite.engine_args(loader, root)
                        ),
                    ]
                    if tune:
                        command.append("--tune-threshold")
                    persist(completed, performance, skipped, failures, name, f"eval:{dataset['name']}", name)
                    result = suite.run(command, model_env, log, f"eval {loader}/{dataset['name']} ({engine})")
                    if result:
                        reason = classify_failure(log_path)
                        if reason:
                            append_unique(skipped, {"profile": name, "dataset": dataset["name"], "reason": reason})
                        else:
                            append_unique(failures, {"profile": name, "dataset": dataset["name"], "phase": "eval", "reason": f"exit {result}; see {log_path.relative_to(suite.REPO)}"})
                    else:
                        append_unique(completed, {"profile": name, "dataset": dataset["name"], "rows": 50, "commit": SUPPLEMENTAL_COMMIT, "source": source_method})
                        failures = [
                            item for item in failures
                            if not (item.get("profile") == name and item.get("dataset") == dataset["name"])
                        ]
                    persist(completed, performance, skipped, failures, name, f"eval-complete:{dataset['name']}", name)

                score_dir = suite.CAMPAIGN / "performance" / loader / engine
                score_rows = 12
                if not suite.score_complete(score_dir, engine, score_rows):
                    command = [
                        str(suite.BENCH), "score", str(suite.CAMPAIGN / "shape-smoke-12.jsonl"),
                        "--engine", engine, "--output-dir", str(score_dir), "--reps", "1",
                        "--host", suite.HOST, "--commit", SUPPLEMENTAL_COMMIT,
                        *(
                            ["--model-root", str(root)]
                            if loader in {"winnow-e4b", "clef-flash", "clef-flash-gguf", "clef-27b-gguf"} and not temporary_install
                            else suite.engine_args(loader, root)
                        ),
                    ]
                    persist(completed, performance, skipped, failures, name, "score:12-rows", name)
                    result = suite.run(command, model_env, log, f"score {loader} ({engine}, 12 rows)")
                    if result:
                        append_unique(failures, {"profile": name, "phase": "shape score", "reason": f"exit {result}; see {log_path.relative_to(suite.REPO)}"})
                    else:
                        append_unique(performance, {"profile": name, "engine": engine, "rows": score_rows, "commit": SUPPLEMENTAL_COMMIT, "source": source_method})
                        failures = [
                            item for item in failures
                            if not (item.get("profile") == name and item.get("phase") == "shape score")
                        ]
                elif not any(x.get("profile") == name and x.get("rows") == score_rows for x in performance):
                    append_unique(performance, {"profile": name, "engine": engine, "rows": score_rows, "commit": SUPPLEMENTAL_COMMIT, "source": source_method})
                persist(completed, performance, skipped, failures, name, "profile-complete", name)
            finally:
                if temporary_install and (install_root / "manifest.json").is_file():
                    persist(completed, performance, skipped, failures, name, "remove", name)
                    if suite.run([str(suite.OPENKIND), "rm", name], model_env, log, f"remove {name}"):
                        failures.append({"profile": name, "phase": "remove", "reason": "model removal failed"})
                # The disposable campaign store also contains downloaded blobs.
                # Remove this exact store before the next large checkpoint pull.
                if temporary_install:
                    shutil.rmtree(suite.MODEL_STORE, ignore_errors=True)
                persist(completed, performance, skipped, failures, name, "complete", None)

    persist(completed, performance, skipped, failures, None, "supplemental-finished", None)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
