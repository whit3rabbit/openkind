#!/usr/bin/env python3
"""Aggregate `openkind-bench` summaries into the recommendation dataset.

Reads one or more `openkind-bench/v1` summary JSON files and writes a single
machine-readable document mapping each registry profile to its measured
backends: context limits, memory, CPU usage, load time, and speed. The output
feeds model-recommendation work; it records request-path evidence only and
carries no model-quality claim.

Usage:
    python3 scripts/build-recommendation-data.py \
        --output docs/benchmarks/<campaign>/recommendation-data.json \
        --campaign <campaign label> \
        <summary1.json> [summary2.json ...]

Missing telemetry fields (summaries written before the CPU/host/context
additions) aggregate as null. Exit status is nonzero when a summary is not
readable JSON with the expected schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "openkind-recommendation-data/v1"

# Per-engine execution qualities. Records must never present an unqualified
# engine as decision-safe; the recommendation consumer reads these labels.
ENGINE_QUALIFICATION = {
    "qwen35": "cpu-parity-qualified",
    "qwen35-mlx-fp32": "parity-qualified",
    "qwen35-mlx-bf16": "unqualified-candidate",
    "laya-english": "reference-parity-readout",
    "laya-multilingual": "reference-parity-readout",
    "laya-typed-decisions": "reference-parity-readout",
    "laya": "reference-parity-readout",
    "mock": "not-a-model",
}


def strategy_of(summary: dict) -> dict:
    """Pick the scheduler-choice strategy row, or the single strategy row."""
    strategies = summary.get("strategies") or []
    if not strategies:
        raise ValueError("summary has no strategy rows")
    for row in strategies:
        if row.get("strategy") == "choose_strategy":
            return row
    if len(strategies) == 1:
        return strategies[0]
    return min(strategies, key=lambda row: row.get("p50_seconds", float("inf")))


def all_strategy_rows(summary: dict) -> list[dict]:
    """Compact per-strategy rows kept alongside the primary measurement."""
    rows = []
    for row in summary.get("strategies") or []:
        rows.append(
            {
                "strategy": row.get("strategy"),
                "p50_seconds": row.get("p50_seconds"),
                "p95_seconds": row.get("p95_seconds"),
                "decisions_per_second": row.get("decisions_per_second"),
                "input_tokens_total": row.get("input_tokens_total"),
                "cpu_time_seconds": row.get("cpu_time_seconds"),
                "avg_cpu_percent": row.get("avg_cpu_percent"),
                "model_load_seconds": row.get("model_load_seconds"),
            }
        )
    return rows


def entry_from_summary(path: Path) -> dict:
    summary = json.loads(path.read_text())
    if summary.get("schema") != "openkind-bench/v1":
        raise ValueError(f"{path}: unexpected schema {summary.get('schema')!r}")
    primary = strategy_of(summary)
    return {
        "engine": summary.get("engine"),
        "profile_id": summary.get("profile_id"),
        "model_revision": summary.get("model_revision"),
        "support_label": ENGINE_QUALIFICATION.get(summary.get("engine"), "recorded"),
        "summary_file": str(path),
        "commit": summary.get("commit"),
        "host": summary.get("host"),
        "host_hardware": summary.get("host_hardware"),
        "context": summary.get("context"),
        "workload": {
            "sha256": (summary.get("fixture") or {}).get("sha256"),
            "rows": (summary.get("fixture") or {}).get("rows"),
            "groups": (summary.get("fixture") or {}).get("groups"),
        },
        "grouping": summary.get("grouping"),
        "reps": summary.get("reps"),
        "peak_resident_bytes": summary.get("peak_resident_bytes"),
        "primary": {
            "strategy": primary.get("strategy"),
            "p50_seconds": primary.get("p50_seconds"),
            "p95_seconds": primary.get("p95_seconds"),
            "decisions_per_second": primary.get("decisions_per_second"),
            "input_tokens_total": primary.get("input_tokens_total"),
            "cpu_time_seconds": primary.get("cpu_time_seconds"),
            "avg_cpu_percent": primary.get("avg_cpu_percent"),
            "model_load_seconds": primary.get("model_load_seconds"),
        },
        "strategies": all_strategy_rows(summary),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path, help="openkind-bench summary JSON files")
    parser.add_argument("--output", type=Path, required=True, help="aggregate JSON path")
    parser.add_argument("--campaign", required=True, help="campaign label recorded in the output")
    args = parser.parse_args(argv)

    entries = []
    for path in args.summaries:
        try:
            entries.append(entry_from_summary(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

    host = next(
        (entry["host_hardware"] for entry in entries if entry.get("host_hardware")),
        None,
    )
    # Registry names join the benchmark evidence to registry/v1 catalog names.
    registry_names = {
        "a047d6802c3f06f085b8": "qwen35-state-first:a047d6802c3f06f085b8",
        "c8ea29bf1e33a343c4b7": "laya-english:c8ea29bf1e33a343c4b7",
        "f4064eb56fb7f7d325e1": "laya-multilingual:f4064eb56fb7f7d325e1",
        "9d28cfa9567902801ed1": "laya-typed-decisions:9d28cfa9567902801ed1",
    }
    for entry in entries:
        entry["registry_name"] = registry_names.get(entry.get("profile_id"))

    aggregate = {
        "schema": SCHEMA,
        "campaign": args.campaign,
        "scope": (
            "request-path timing, memory, CPU, and context evidence per registry "
            "profile and executable backend; no model-quality claim"
        ),
        "host_hardware": host,
        "models": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(f"wrote {args.output} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
