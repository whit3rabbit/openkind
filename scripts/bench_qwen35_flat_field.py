#!/usr/bin/env python3
"""Run paired fresh-process Rust MLX flat-field versus nested benchmarks."""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def timed_median(run: dict) -> float:
    samples = run["samples"]
    return statistics.median(sample["total_ms"] for sample in samples[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--max-lanes", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.q < 1 or args.k < 1 or args.iterations < 2 or args.pairs < 1:
        parser.error("q, k, pairs must be positive and iterations must be at least 2")

    payload = {
        "schema": "openkind-flat-field-paired/v1",
        "q": args.q,
        "k": args.k,
        "iterations": args.iterations,
        "warmup_samples_per_process": 1,
        "max_lanes": args.max_lanes,
        "runs": [],
        "pairs": [],
    }
    for pair in range(args.pairs):
        order = ["nested_batched", "flat"] if pair % 2 == 0 else ["flat", "nested_batched"]
        pair_runs = {}
        for strategy in order:
            command = [
                str(args.binary),
                "--checkpoint", str(args.checkpoint),
                "--strategy", strategy,
                "--q", str(args.q),
                "--k", str(args.k),
                "--iterations", str(args.iterations),
                "--max-lanes", str(args.max_lanes),
            ]
            started = time.monotonic()
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=args.timeout
            )
            run = json.loads(completed.stdout)
            run["pair"] = pair + 1
            run["process_wall_s"] = time.monotonic() - started
            payload["runs"].append(run)
            pair_runs[strategy] = run
            save(args.output, payload)
            print(
                f"Q{args.q}/K{args.k} pair {pair + 1} {strategy}: "
                f"warm median {timed_median(run):.1f} ms",
                flush=True,
            )
        baseline = pair_runs["nested_batched"]
        flat = pair_runs["flat"]
        if baseline["profile"] != flat["profile"] or baseline["revision"] != flat["revision"]:
            raise ValueError("model identities differ within a pair")
        probabilities = zip(
            baseline["samples"][-1]["probabilities"], flat["samples"][-1]["probabilities"]
        )
        max_probability_error = max(
            abs(a - b) for left, right in probabilities for a, b in zip(left, right)
        )
        baseline_ms = timed_median(baseline)
        flat_ms = timed_median(flat)
        payload["pairs"].append({
            "pair": pair + 1,
            "order": order,
            "nested_batched_median_ms": baseline_ms,
            "flat_median_ms": flat_ms,
            "flat_change_percent": 100.0 * (flat_ms - baseline_ms) / baseline_ms,
            "max_probability_error": max_probability_error,
            "selections_equal": baseline["samples"][-1]["selected_indices"]
            == flat["samples"][-1]["selected_indices"],
        })
        save(args.output, payload)


if __name__ == "__main__":
    main()
