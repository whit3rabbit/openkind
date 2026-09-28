#!/usr/bin/env python3
"""Measure the pinned upstream Jev-Style MLX runtime and emit Qwen harness workloads.

Requires the upstream requirements.txt in an isolated Python 3.12 environment,
the Hub files at the pinned revision, and the local OpenKind checkout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer


TARGET_STATE_TOKENS = (878, 3950, 24436)
REPETITIONS = 3
PARAGRAPH = (
    "The service accepts a structured request, validates every criterion, and "
    "returns a probability for each available outcome. Operators can inspect "
    "the deployment record, compare the request with recent changes, and "
    "choose a follow-up action based on the evidence.\n"
    "fn evaluate_request(state: &State, questions: &[Question]) -> Result<Answers> {\n"
    "    let context = render_state(state)?;\n"
    "    let features = model.forward(&context)?;\n"
    "    score_questions(features, questions)\n"
    "}\n"
)


def serialize_document(document: str) -> str:
    return json.dumps({"document": document}, ensure_ascii=True, separators=(",", ":"))


def make_state(tokenizer: Tokenizer, target: int) -> tuple[str, dict[str, str], int]:
    # Grow the document until its serialized state reaches the requested size.
    document = ""
    encoded = 0
    while encoded < target:
        document += PARAGRAPH
        encoded = len(tokenizer.encode(serialize_document(document), add_special_tokens=False).ids)

    # Find a character prefix at the target token count. The final narrow scan
    # handles tokenizer merges at the cut boundary.
    low, high = 0, len(document)
    chosen: str | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate = document[:middle]
        n_tokens = len(tokenizer.encode(serialize_document(candidate), add_special_tokens=False).ids)
        if n_tokens == target:
            chosen = candidate
            break
        if n_tokens < target:
            low = middle + 1
        else:
            high = middle - 1
    if chosen is None:
        start, stop = max(0, high - 256), min(len(document), low + 256)
        for chars in range(start, stop + 1):
            candidate = document[:chars]
            if len(tokenizer.encode(serialize_document(candidate), add_special_tokens=False).ids) == target:
                chosen = candidate
                break
    if chosen is None:
        # Keep the closest reproducible prefix if a tokenizer version makes an
        # exact count unreachable at a JSON string boundary.
        candidates = [document[:max(0, high)], document[:min(len(document), low)]]
        chosen = min(
            candidates,
            key=lambda value: abs(
                len(tokenizer.encode(serialize_document(value), add_special_tokens=False).ids) - target
            ),
        )

    serialized = serialize_document(chosen)
    observed = len(tokenizer.encode(serialized, add_special_tokens=False).ids)
    state = json.loads(serialized)
    return serialized, state, observed


def question_set() -> list[dict]:
    result = []
    for i in range(4):
        result.append({
            "t": "choice",
            "ins": f"Which response best fits evidence item {i + 1}?",
            "crit": {
                "investigate": "Inspect the evidence and recent changes",
                "ask": "Ask for the missing diagnostic details",
                "resolve": "Apply the verified correction",
                "escalate": "Escalate the unresolved issue",
                "__none__": "None of these responses fits",
            },
        })
    for i in range(4):
        result.append({
            "t": "noul",
            "ins": f"Does the available evidence support claim {i + 1}?",
            "crit": {"false": "The evidence does not support the claim", "true": "The evidence supports the claim"},
        })
    for i in range(2):
        result.append({
            "t": "score",
            "ins": f"How urgent is follow-up item {i + 1}?",
            "crit": ["routine", "elevated", "high", "critical"],
        })
    return result


def openkind_row(row_id: str, state: dict[str, str], question: dict) -> dict:
    common = {"id": row_id, "state": state, "text": question["ins"]}
    if question["t"] == "choice":
        return {
            **common,
            "primitive": "choice",
            "options": [{"id": key, "description": value} for key, value in question["crit"].items()],
        }
    if question["t"] == "noul":
        return {**common, "primitive": "noul", "criteria": question["crit"]}
    return {**common, "primitive": "score", "levels": question["crit"]}


def write_openkind_workloads(output_dir: Path, states: dict[int, tuple[str, dict, int]], questions: list[dict]) -> None:
    for target, (_, state, state_tokens) in states.items():
        for count in (1, 10):
            path = output_dir / f"openkind-s{state_tokens}-q{count}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for i, question in enumerate(questions[:count]):
                    row = openkind_row(f"s{target}-q{i:02}", state, question)
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def timed_call(engine, state: str, questions: list[dict]) -> tuple[float, dict, list[dict]]:
    started = time.perf_counter()
    results = engine.score_many(state, questions)
    elapsed = time.perf_counter() - started
    timing = dict(engine.last_timing)
    if len(results) != len(questions):
        raise RuntimeError("score_many returned a different question count")
    for result in results:
        probabilities = list(result["probabilities"].values())
        if not probabilities or not all(math.isfinite(p) for p in probabilities):
            raise RuntimeError("model returned a non-finite or empty probability map")
        if abs(sum(probabilities) - 1.0) > 1e-9:
            raise RuntimeError("model probabilities do not sum to one")
    return elapsed, timing, results


def run(args: argparse.Namespace) -> dict:
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(model_dir))
    from jev_style_decision_mlx import JevStyleDecisionMLX, REPO_ID

    tokenizer_path = model_dir / "bf16" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    states = {
        target: make_state(tokenizer, target)
        for target in TARGET_STATE_TOKENS
    }
    write_openkind_workloads(output_dir, states, question_set())

    started = time.perf_counter()
    engine = JevStyleDecisionMLX(model_dir, precision=args.precision, verify=True)
    load_seconds = time.perf_counter() - started
    engine.engine.reset()
    mx.reset_peak_memory()

    questions = question_set()
    timings = []
    for target, (state, _state_object, state_tokens) in states.items():
        for count in (1, 10):
            selected = questions[:count]
            warmup_seconds, _, _ = timed_call(engine, state, selected)
            engine.engine.reset()
            samples, stage_samples, last_results = [], [], []
            for _ in range(args.reps):
                engine.engine.reset()
                elapsed, timing, results = timed_call(engine, state, selected)
                if timing.get("state_reused"):
                    raise RuntimeError("fresh-state timing unexpectedly reused cached state")
                samples.append(elapsed)
                stage_samples.append(timing)
                last_results = results
            timings.append({
                "state_tokens": state_tokens,
                "questions": count,
                "warmup_seconds": warmup_seconds,
                "samples_seconds": samples,
                "median_seconds": statistics.median(samples),
                "median_state_seconds": statistics.median(x["state_ms"] for x in stage_samples) / 1000,
                "median_questions_seconds": statistics.median(x["questions_ms"] for x in stage_samples) / 1000,
                "max_input_tokens": max(x["input_tokens"] for x in last_results),
                "max_blocks": max(x["blocks"] for x in last_results),
            })

    cached = []
    for target, (state, _state_object, state_tokens) in states.items():
        engine.engine.reset()
        engine.score_many(state, [questions[0]])
        engine.score_many(state, [questions[1]])  # warm the question path with the state already present
        samples, stages = [], []
        for _ in range(args.reps):
            elapsed, timing, _ = timed_call(engine, state, [questions[1]])
            if not timing.get("state_reused"):
                raise RuntimeError("cached-state timing recomputed the state")
            samples.append(elapsed)
            stages.append(timing)
        cached.append({
            "state_tokens": state_tokens,
            "questions": 1,
            "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "median_questions_seconds": statistics.median(x["questions_ms"] for x in stages) / 1000,
            "state_reused": True,
        })

    result = {
        "schema": "openkind-external-mlx-benchmark/v1",
        "model": REPO_ID,
        "revision": "b86e4cbc6f420f5d7d1691299d62db272934ebf4",
        "precision": args.precision,
        "host": {
            "chip": platform.processor(),
            "machine": platform.machine(),
            "system": platform.platform(),
            "load_average": os.getloadavg(),
        },
        "runtime": {"python": platform.python_version(), "mlx": "0.32.2", "mlx_lm": "0.31.3"},
        "model_load_seconds_including_manifest_verification": load_seconds,
        "active_memory_bytes_after_load": int(mx.get_active_memory()),
        "peak_memory_bytes_after_measurements": int(mx.get_peak_memory()),
        "cache_memory_bytes_after_measurements": int(mx.get_cache_memory()),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "repetitions": args.reps,
        "timings": timings,
        "cached_state_timings": cached,
    }
    engine.close()
    path = output_dir / f"jev-style-{args.precision}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="pinned Hub repository directory")
    parser.add_argument("--precision", choices=("bf16", "8bit"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=REPETITIONS)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be at least one")
    run(args)


if __name__ == "__main__":
    main()
