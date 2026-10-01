#!/usr/bin/env python3
"""Run the pinned quality and timing suites once per catalog profile.

The model store and dataset cache are disposable roots under /tmp. Each
profile is pulled, evaluated on applicable registered datasets, then removed.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = Path(__file__).resolve().parent
MODEL_STORE = Path("/tmp/openkind-local-suite-2026-10-01/models")
DATASET_STORE = Path.home() / "Library/Caches/openkind-local-suite-2026-10-01/datasets"
STATUS_PATH = CAMPAIGN / "suite-status.json"
OPENKIND = REPO / "target/release/openkind"
BENCH = REPO / "target/release/openkind-bench"
COMMIT = "e3cb136f431e817ac4aca475ea381b0840ef2cd0"
HOST = "Apple M4 Max, 14 cores, 36 GiB RAM, macOS arm64"

ENGINE_BY_LOADER = {
    "qwen35-state-first": "qwen35-mlx-fp32",
    "laya-english": "laya-english-mlx-fp32",
    "laya-multilingual": "laya-multilingual-mlx-fp32",
    "laya-typed-decisions": "laya-typed-decisions-mlx-fp32",
    "decoder-logit-letter": "decoder-letter",
    "encoder-nli": "encoder-nli",
    "encoder-instruct-label": "encoder-instruct-label-mlx-fp32",
    "decoder-logit-llm": "decoder-llm",
    "schema-scorer": "schema-scorer",
    "qwen3guard": "qwen3-guard",
    "kev": "kev",
    "decoder-logit-qwen35": "decoder-logit-qwen35-mlx-fp32",
    "von": "von",
    "decider-4b": "decider-4b",
    "plumb-4b": "plumb-4b",
    "decoder-logit-qwen3-06b": "decoder-logit-qwen3-06b",
    "decoder-logit-qwen3-17b": "decoder-logit-qwen3-17b",
    "decoder-logit-qwen3-4b": "decoder-logit-qwen3-4b",
    "winnow": "winnow",
}


def run(command: list[str], env: dict[str, str], log, label: str) -> int:
    log.write("$ " + shlex.join(command) + "\n")
    log.flush()
    print(f"[start] {label}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log.write(line)
        log.flush()
        if line.startswith(("[bench]", "Downloading ", "Verifying ", "Installed ")):
            print(line.rstrip(), flush=True)
    status = process.wait()
    log.write(f"[exit {status}] {label}\n\n")
    log.flush()
    print(f"[done] {label}: {'OK' if status == 0 else 'FAILED'}", flush=True)
    return status


def engine_args(loader: str, root: Path) -> list[str]:
    if loader == "qwen35-state-first":
        return [
            "--bundle-root",
            str(root / "bundle"),
            "--checkpoint-root",
            str(root / "checkpoint"),
            "--tokenizer",
            str(root / "checkpoint/tokenizer.json"),
        ]
    if loader == "kev":
        return [
            "--model-root",
            str(root / "adapter"),
            "--checkpoint-root",
            str(root / "base"),
        ]
    if loader == "von":
        return ["--model-root", str(root)]
    if loader.startswith("laya-"):
        return ["--model-root", str(root)]
    return ["--model-root", str(root / "checkpoint")]


def component_embedding(log, env: dict[str, str], root: Path) -> tuple[list[str], list[str]]:
    executable = REPO / "target/release/examples/encoder_embedding_bench"
    reports = []
    failed = []
    for backend in ("cpu", "mlx"):
        output = CAMPAIGN / "performance" / f"encoder-embedding-{backend}.json"
        command = [
            str(executable),
            "--model-root",
            str(root),
            "--backend",
            backend,
            "--host",
            HOST,
            "--commit",
            COMMIT,
            "--working-tree-dirty",
            "true",
        ]
        log.write("$ " + shlex.join(command) + "\n")
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if result.returncode == 0:
            output.write_text(result.stdout)
            reports.append(str(output.relative_to(REPO)))
            log.write(f"wrote {output.relative_to(REPO)}\n")
        else:
            failed.append(backend)
        log.write(result.stdout)
        log.write(f"[exit {result.returncode}] embedding {backend}\n\n")
        log.flush()
        print(f"encoder-embedding/{backend}: {'OK' if result.returncode == 0 else 'FAILED'}", flush=True)
    return reports, failed


def evaluation_complete(output: Path, engine: str, rows: int, tune_threshold: bool) -> bool:
    reports = list(output.glob("dataset-eval-*.json"))
    if not reports:
        return False
    try:
        report = json.loads(reports[0].read_text())
        if report.get("rows") != rows or report.get("metrics", {}).get("rows") != rows:
            return False
        summaries = [json.loads(path.read_text()) for path in output.glob("summary-*.json")]
        if not any(
            summary.get("engine_variant", summary.get("engine")) == engine
            and summary.get("fixture", {}).get("rows") == rows
            for summary in summaries
        ):
            return False
        if tune_threshold:
            tuning = output / f"tuning-{report.get('dataset', {}).get('name', '')}"
            if not any(tuning.glob("summary-*.json")):
                return False
            tuned_threshold = report.get("metrics", {}).get("tuned_threshold")
            if not isinstance(tuned_threshold, dict):
                return False
        return True
    except (OSError, ValueError, AttributeError):
        return False


def score_complete(output: Path, engine: str, rows: int) -> bool:
    try:
        for path in output.glob("summary-*.json"):
            summary = json.loads(path.read_text())
            if summary.get("engine_variant", summary.get("engine")) != engine:
                continue
            if not any(strategy.get("rows") == rows for strategy in summary.get("strategies", [])):
                continue
            for strategy, digest in summary.get("prediction_sha256", {}).items():
                prediction = output / f"predictions-{engine}-{strategy}.jsonl"
                if prediction.is_file() and prediction.stat().st_size > 0 and digest:
                    return True
    except (OSError, ValueError, AttributeError):
        return False
    return False


def write_status(
    *,
    completed: list[dict[str, str | int]],
    performance: list[dict[str, str | int]],
    skipped: list[dict[str, str]],
    failures: list[dict[str, str]],
    current_profile: str | None,
    current_phase: str,
    installed_profile: str | None,
    embedding_reports: list[str] | None = None,
) -> None:
    status_doc = {
        "schema": "openkind-local-model-suite-status/v1",
        "host": HOST,
        "commit": COMMIT,
        "working_tree_dirty": True,
        "dataset_limit_per_eval": 50,
        "current_profile": current_profile,
        "current_phase": current_phase,
        "installed_profile": installed_profile,
        "encoder_embedding_reports": embedding_reports or [],
        "completed_evaluations": completed,
        "completed_performance_runs": performance,
        "skipped_combinations": skipped,
        "failures": failures,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=STATUS_PATH.parent, delete=False
    ) as temporary:
        json.dump(status_doc, temporary, indent=2)
        temporary.write("\n")
        temp_path = Path(temporary.name)
    os.replace(temp_path, STATUS_PATH)


def prepare_workloads() -> tuple[Path, Path]:
    performance = CAMPAIGN / "performance"
    performance.mkdir(parents=True, exist_ok=True)
    full = CAMPAIGN / "shape777.jsonl"
    command = [
        str(BENCH),
        "gen-workload",
        "--states",
        "37",
        "--criteria",
        "21",
        "--seed",
        "291607",
        "--output",
        str(full),
    ]
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    source_rows = [json.loads(line) for line in full.read_text().splitlines() if line]
    if len(source_rows) != 777:
        raise RuntimeError(f"expected 777 shape rows, found {len(source_rows)}")

    # Keep three decisions from each of four distinct states for documented
    # smoke-scale controls, without biasing the sample to a single state.
    state_counts: dict[str, int] = {}
    smoke_rows = []
    for row in source_rows:
        key = json.dumps(row["state"], sort_keys=True, separators=(",", ":"))
        if key not in state_counts and len(state_counts) == 4:
            continue
        count = state_counts.get(key, 0)
        if count < 3:
            smoke_rows.append(row)
            state_counts[key] = count + 1
    smoke = CAMPAIGN / "shape-smoke-12.jsonl"
    smoke.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in smoke_rows))
    if len(smoke_rows) != 12 or len(state_counts) != 4:
        raise RuntimeError(f"expected 12 smoke rows across four states, found {len(smoke_rows)}")
    return full, smoke


def main() -> int:
    catalog_root = REPO / "registry/v1"
    catalog = json.loads((catalog_root / "catalog.json").read_text())
    datasets = json.loads(
        (REPO / "crates/openkind-datasets/registry/v1/datasets.json").read_text()
    )["datasets"]
    env = os.environ.copy()
    env["OPENKIND_MODELS_DIR"] = str(MODEL_STORE)
    failures: list[dict[str, str]] = []
    completed: list[dict[str, str | int]] = []
    performance: list[dict[str, str | int]] = []
    skipped: list[dict[str, str]] = []
    embedding_reports: list[str] = []
    full_workload, smoke_workload = prepare_workloads()
    DATASET_STORE.mkdir(parents=True, exist_ok=True)

    # Dataset files survive a host restart. Models stay in /tmp and are
    # removed after each profile so a crash cannot strand a persistent model.
    write_status(
        completed=completed,
        performance=performance,
        skipped=skipped,
        failures=failures,
        current_profile=None,
        current_phase="dataset-cache-setup",
        installed_profile=None,
    )
    for dataset in datasets:
        if not (DATASET_STORE / "datasets" / dataset["name"] / "entry.json").is_file():
            with (CAMPAIGN / "logs" / "dataset-pull.log").open("a") as log:
                status = run(
                    [str(BENCH), "dataset", "pull", dataset["name"], "--datasets-dir", str(DATASET_STORE)],
                    env,
                    log,
                    f"pull dataset {dataset['name']}",
                )
            if status:
                failures.append({"profile": "dataset-cache", "phase": dataset["name"], "reason": f"exit {status}"})
                write_status(
                    completed=completed,
                    performance=performance,
                    skipped=skipped,
                    failures=failures,
                    current_profile="dataset-cache",
                    current_phase=f"pull-failed:{dataset['name']}",
                    installed_profile=None,
                )
                return 1

    for entry in catalog["models"]:
        loader = entry["loader_id"]
        name = entry["name"]
        manifest = json.loads((catalog_root / entry["manifest_path"]).read_text())
        root = MODEL_STORE / "models" / name
        log_path = CAMPAIGN / "logs" / f"dataset-{loader}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a") as log:
            log.write(f"profile={name}\nloader={loader}\ncommit={COMMIT}\nhost={HOST}\n\n")
            write_status(
                completed=completed,
                performance=performance,
                skipped=skipped,
                failures=failures,
                current_profile=name,
                current_phase="before-pull",
                installed_profile=None,
            )
            if loader == "encoder-embedding":
                model_env = env.copy()
                try:
                    if not (root / "manifest.json").is_file():
                        write_status(
                            completed=completed,
                            performance=performance,
                            skipped=skipped,
                            failures=failures,
                            current_profile=name,
                            current_phase="pull",
                            installed_profile=name,
                        )
                        status = run([str(OPENKIND), "pull", name], model_env, log, f"pull {name}")
                        if status:
                            failures.append({"profile": name, "phase": "pull", "reason": f"exit {status}"})
                            continue
                    reports, failed = component_embedding(log, model_env, root)
                    embedding_reports.extend(reports)
                    for backend in failed:
                        failures.append({"profile": name, "phase": f"embedding {backend}", "reason": "benchmark failed"})
                finally:
                    if (root / "manifest.json").is_file():
                        write_status(
                            completed=completed,
                            performance=performance,
                            skipped=skipped,
                            failures=failures,
                            current_profile=name,
                            current_phase="remove",
                            installed_profile=name,
                        )
                        status = run([str(OPENKIND), "rm", name], model_env, log, f"remove {name}")
                        if status:
                            failures.append({"profile": name, "phase": "remove", "reason": f"exit {status}"})
                write_status(
                    completed=completed,
                    performance=performance,
                    skipped=skipped,
                    failures=failures,
                    current_profile=name,
                    current_phase="complete",
                    installed_profile=None,
                )
                continue

            if loader == "winnow":
                skipped.append({"profile": name, "reason": "score harness wires mock siblings; routing-cost evidence only"})
                log.write("Skipped model-quality metrics; score harness uses mock siblings for routing-cost evidence.\n")

            engine = ENGINE_BY_LOADER.get(loader)
            if engine is None:
                failures.append({"profile": name, "phase": "engine mapping", "reason": "no dataset engine"})
                print(f"{loader}: FAILED (no engine mapping)", flush=True)
                continue

            model_env = env.copy()
            try:
                if not (root / "manifest.json").is_file():
                    write_status(
                        completed=completed,
                        performance=performance,
                        skipped=skipped,
                        failures=failures,
                        current_profile=name,
                        current_phase="pull",
                        installed_profile=name,
                    )
                    status = run([str(OPENKIND), "pull", name], model_env, log, f"pull {name}")
                    if status:
                        failures.append({"profile": name, "phase": "pull", "reason": f"exit {status}"})
                        continue

                if loader != "winnow":
                    types = set(manifest.get("question_types", []))
                    for dataset in datasets:
                        primitive = dataset.get("primitives", [""])[0]
                        if primitive not in types:
                            skipped.append({"profile": name, "dataset": dataset["name"], "reason": f"profile does not declare {primitive}"})
                            continue
                        output = CAMPAIGN / "quality" / loader / dataset["name"]
                        if evaluation_complete(output, engine, 50, primitive == "noul"):
                            completed.append({"profile": name, "dataset": dataset["name"], "rows": 50})
                            log.write(f"[resume] verified existing 50-row report for {dataset['name']} ({engine})\n")
                            write_status(
                                completed=completed,
                                performance=performance,
                                skipped=skipped,
                                failures=failures,
                                current_profile=name,
                                current_phase=f"verified-existing:{dataset['name']}",
                                installed_profile=name,
                            )
                            continue
                        command = [
                            str(BENCH),
                            "dataset",
                            "eval",
                            dataset["name"],
                            "--datasets-dir",
                            str(DATASET_STORE),
                            "--limit",
                            "50",
                            "--engine",
                            engine,
                            "--host",
                            HOST,
                            "--commit",
                            COMMIT,
                            "--output-dir",
                            str(output),
                            "--pretty",
                            *engine_args(loader, root),
                        ]
                        if primitive == "noul":
                            command.append("--tune-threshold")
                        write_status(
                            completed=completed,
                            performance=performance,
                            skipped=skipped,
                            failures=failures,
                            current_profile=name,
                            current_phase=f"eval:{dataset['name']}",
                            installed_profile=name,
                        )
                        status = run(command, model_env, log, f"eval {loader}/{dataset['name']} ({engine})")
                        if status:
                            excerpt = "\n".join(log_path.read_text().splitlines()[-16:])
                            if "candidate" in excerpt.lower() and any(term in excerpt.lower() for term in ("max", "maximum", "at most", "supports")):
                                skipped.append({"profile": name, "dataset": dataset["name"], "reason": "input exceeds engine candidate-count capacity"})
                            else:
                                failures.append({"profile": name, "dataset": dataset["name"], "phase": "eval", "reason": f"exit {status}; see {log_path.relative_to(REPO)}"})
                        else:
                            completed.append({"profile": name, "dataset": dataset["name"], "rows": 50})
                        write_status(
                            completed=completed,
                            performance=performance,
                            skipped=skipped,
                            failures=failures,
                            current_profile=name,
                            current_phase=f"eval-complete:{dataset['name']}",
                            installed_profile=name,
                        )

                score_engine = engine
                score_input = smoke_workload
                score_rows = 12
                if loader in {"decoder-logit-qwen3-06b", "plumb-4b"}:
                    score_input = full_workload
                    score_rows = 777
                supported_types = set(manifest.get("question_types", []))
                if "score" not in supported_types:
                    filtered_rows = []
                    for line in score_input.read_text().splitlines():
                        if not line:
                            continue
                        row = json.loads(line)
                        if row.get("primitive") in supported_types:
                            filtered_rows.append(row)
                    score_input = CAMPAIGN / "shape-filtered" / f"{loader}.jsonl"
                    score_input.parent.mkdir(parents=True, exist_ok=True)
                    score_input.write_text(
                        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in filtered_rows)
                    )
                    score_rows = len(filtered_rows)
                if loader == "plumb-4b":
                    # This path is present in the active checkout but has no
                    # frozen parity result yet; keep its status diagnostic.
                    score_engine = "plumb-4b-mlx-fp32"
                score_dir = CAMPAIGN / "performance" / loader / score_engine
                score_command = [
                    str(BENCH),
                    "score",
                    str(score_input),
                    "--engine",
                    score_engine,
                    "--output-dir",
                    str(score_dir),
                    "--reps",
                    "1",
                    "--host",
                    HOST,
                    "--commit",
                    COMMIT,
                    *engine_args(loader, root),
                ]
                if loader == "winnow":
                    score_command.extend(["--adapter", str(root / "adapter.safetensors")])
                if loader == "qwen35-state-first":
                    score_command.extend([
                        "--strategies",
                        "repeated_full,nested_sequential,nested_batched,choose_strategy",
                    ])
                if score_complete(score_dir, score_engine, score_rows):
                    performance.append({"profile": name, "engine": score_engine, "rows": score_rows})
                    log.write(f"[resume] verified existing {score_rows}-row timing report for {score_engine}\n")
                else:
                    write_status(
                        completed=completed,
                        performance=performance,
                        skipped=skipped,
                        failures=failures,
                        current_profile=name,
                        current_phase=f"score:{score_rows}-rows",
                        installed_profile=name,
                    )
                    status = run(score_command, model_env, log, f"score {loader} ({score_engine}, {score_rows} rows)")
                    if status:
                        failures.append({"profile": name, "phase": "shape score", "reason": f"exit {status}; see {log_path.relative_to(REPO)}"})
                    else:
                        performance.append({"profile": name, "engine": score_engine, "rows": score_rows})
                    write_status(
                        completed=completed,
                        performance=performance,
                        skipped=skipped,
                        failures=failures,
                        current_profile=name,
                        current_phase="score-complete" if status == 0 else "score-failed",
                        installed_profile=name,
                    )
            finally:
                if (root / "manifest.json").is_file():
                    write_status(
                        completed=completed,
                        performance=performance,
                        skipped=skipped,
                        failures=failures,
                        current_profile=name,
                        current_phase="remove",
                        installed_profile=name,
                    )
                    status = run([str(OPENKIND), "rm", name], model_env, log, f"remove {name}")
                    if status:
                        failures.append({"profile": name, "phase": "remove", "reason": f"exit {status}"})
            write_status(
                completed=completed,
                performance=performance,
                skipped=skipped,
                failures=failures,
                current_profile=name,
                current_phase="complete",
                installed_profile=None,
            )

    control_log_path = CAMPAIGN / "logs" / "router-script.log"
    control_log_path.parent.mkdir(parents=True, exist_ok=True)
    with control_log_path.open("a") as log:
        control_dir = CAMPAIGN / "performance" / "router-script"
        command = [
            str(BENCH),
            "score",
            str(full_workload),
            "--engine",
            "router-script",
            "--output-dir",
            str(control_dir),
            "--reps",
            "1",
            "--host",
            HOST,
            "--commit",
            COMMIT,
        ]
        if score_complete(control_dir, "router-script", 777):
            performance.append({"profile": "router-script", "engine": "router-script", "rows": 777})
        else:
            status = run(command, env, log, "score router-script routing control")
            if status:
                failures.append({"profile": "router-script", "phase": "shape score", "reason": f"exit {status}"})
            else:
                performance.append({"profile": "router-script", "engine": "router-script", "rows": 777})

    write_status(
        completed=completed,
        performance=performance,
        skipped=skipped,
        failures=failures,
        current_profile=None,
        current_phase="finished",
        installed_profile=None,
        embedding_reports=embedding_reports,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
