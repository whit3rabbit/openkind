#!/usr/bin/env python3
"""Offline Choice quality and fresh-process history checks for Qwen profiles.

The quality report never selects a policy threshold or opens a final split.
The history command uses two separate openkind-bench processes and disables its
warmup, so the first anchor is actually the first evaluated request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

NONE = "__none__"
SCHEMA = "openkind-choice-qualification/v1"


def read_jsonl(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows, hashlib.sha256(raw).hexdigest()


def keyed(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        ident = row.get("id")
        if not isinstance(ident, str) or not ident:
            raise ValueError(f"{label}: row has no nonempty id")
        if ident in result:
            raise ValueError(f"{label}: duplicate id {ident!r}")
        result[ident] = row
    return result


def same_ids(left: dict, right: dict, left_name: str, right_name: str) -> None:
    if left.keys() != right.keys():
        missing = sorted(left.keys() - right.keys())
        extra = sorted(right.keys() - left.keys())
        raise ValueError(
            f"{right_name} does not match {left_name}: missing={missing[:5]}, extra={extra[:5]}"
        )


def choice_answer(row: dict, options: set[str], label: str) -> dict:
    answer = row.get("answer")
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError(f"{label}: expected a Choice answer")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or probabilities.keys() != options:
        raise ValueError(f"{label}: probability keys differ from declared options")
    values = list(probabilities.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values
    ) or abs(sum(values) - 1) > 1e-6:
        raise ValueError(f"{label}: invalid probability distribution")
    selected = answer.get("choice")
    if not isinstance(selected, str) or selected not in options or probabilities[selected] + 1e-6 < max(values):
        raise ValueError(f"{label}: selected choice is absent or disagrees with argmax")
    confidence = answer.get("confidence")
    if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError(f"{label}: confidence is invalid")
    return answer


def aggregate(rows: list[tuple[dict, dict]], threshold: float | None) -> dict:
    count = len(rows)
    correct = 0
    false_none = 0
    non_none = 0
    none_total = 0
    none_correct = 0
    nll = 0.0
    zero_gold_probability = 0
    brier = 0.0
    none_brier = 0.0
    calibration_bins = [{"count": 0, "confidence_sum": 0.0, "correct": 0} for _ in range(10)]
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})
    accepted = 0
    wrong_accepted = 0
    accepted_sources = set()
    wrong_accepted_sources = set()
    sources = set()
    for gold, answer in rows:
        label = gold["gold"]
        prediction = answer["choice"]
        probabilities = answer["probabilities"]
        source = gold["source_id"]
        sources.add(source)
        hit = prediction == label
        correct += hit
        per_class[label]["total"] += 1
        per_class[label]["correct"] += hit
        non_none += label != NONE
        false_none += label != NONE and prediction == NONE
        none_total += label == NONE
        none_correct += label == NONE and hit
        if probabilities[label] == 0:
            zero_gold_probability += 1
        else:
            nll += -math.log(probabilities[label])
        brier += sum(
            (probability - float(option == label)) ** 2
            for option, probability in probabilities.items()
        )
        none_brier += (probabilities[NONE] - float(label == NONE)) ** 2
        top_probability = probabilities[prediction]
        bin_result = calibration_bins[min(int(top_probability * 10), 9)]
        bin_result["count"] += 1
        bin_result["confidence_sum"] += top_probability
        bin_result["correct"] += hit
        if threshold is not None and prediction != NONE and probabilities[prediction] >= threshold:
            accepted += 1
            wrong_accepted += not hit
            accepted_sources.add(source)
            if not hit:
                wrong_accepted_sources.add(source)
    policy = None
    if threshold is not None:
        policy = {
            "threshold": threshold,
            "accepted_questions": accepted,
            "wrong_accepted_questions": wrong_accepted,
            "coverage": accepted / count,
            "accepted_error": wrong_accepted / accepted if accepted else None,
            "accepted_sources": len(accepted_sources),
            "sources_with_wrong_acceptance": len(wrong_accepted_sources),
        }
    calibration = [
        {
            "lower": index / 10,
            "upper": (index + 1) / 10,
            "count": result["count"],
            "mean_probability": result["confidence_sum"] / result["count"] if result["count"] else None,
            "accuracy": result["correct"] / result["count"] if result["count"] else None,
        }
        for index, result in enumerate(calibration_bins)
    ]
    ece = sum(
        result["count"] / count * abs(result["accuracy"] - result["mean_probability"])
        for result in calibration if result["count"]
    )
    return {
        "questions": count,
        "sources": len(sources),
        "accuracy": correct / count,
        "false_none_rate": false_none / non_none if non_none else None,
        "none_recall": none_correct / none_total if none_total else None,
        "nll": nll / count if not zero_gold_probability else None,
        "nll_infinite_zero_gold_probability": zero_gold_probability,
        "brier_sum_classes": brier / count,
        "none_brier": none_brier / count,
        "top_label_ece_10": ece,
        "top_label_calibration_bins": calibration,
        "per_class_recall": {
            label: {
                "count": result["total"],
                "recall": result["correct"] / result["total"],
            }
            for label, result in sorted(per_class.items())
        },
        "policy": policy,
    }


def paired_retention(rows: list[tuple[dict, dict, dict]]) -> dict:
    reference_correct = 0
    retained_correct = 0
    lost_correct = 0
    gained_correct = 0
    supported_reference_correct = 0
    supported_lost_to_none = 0
    for gold, answer, reference in rows:
        label = gold["gold"]
        was_correct = reference["choice"] == label
        is_correct = answer["choice"] == label
        reference_correct += was_correct
        retained_correct += was_correct and is_correct
        lost_correct += was_correct and not is_correct
        gained_correct += not was_correct and is_correct
        supported_reference_correct += was_correct and label != NONE
        supported_lost_to_none += was_correct and label != NONE and answer["choice"] == NONE
    return {
        "reference_correct": reference_correct,
        "retained_correct": retained_correct,
        "lost_correct": lost_correct,
        "gained_correct": gained_correct,
        "correct_retention": retained_correct / reference_correct if reference_correct else None,
        "supported_reference_correct": supported_reference_correct,
        "supported_lost_to_none": supported_lost_to_none,
    }


def load_scored_predictions(summary_path: Path, predictions_path: Path, workload_hash: str, workload: dict[str, dict]) -> tuple[dict, dict[str, dict], str, str]:
    rows, prediction_hash = read_jsonl(predictions_path)
    predictions = keyed(rows, "predictions")
    same_ids(workload, predictions, "workload", "predictions")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != "openkind-bench/v1":
        raise ValueError("summary: expected openkind-bench/v1")
    if summary.get("fixture", {}).get("sha256") != workload_hash:
        raise ValueError("summary workload digest does not match supplied workload")
    if summary.get("fixture", {}).get("rows") != len(workload):
        raise ValueError("summary row count does not match workload")
    strategies = {row.get("strategy") for row in rows}
    if len(strategies) != 1 or not all(isinstance(value, str) for value in strategies):
        raise ValueError("predictions must contain exactly one named strategy")
    strategy = next(iter(strategies))
    prediction_hashes = summary.get("prediction_sha256")
    if not isinstance(prediction_hashes, dict) or prediction_hashes.get(strategy) != prediction_hash:
        raise ValueError("summary prediction digest does not match supplied predictions")
    return summary, predictions, prediction_hash, strategy


def quality_report(args: argparse.Namespace) -> dict:
    workload_rows, workload_hash = read_jsonl(args.workload)
    gold_rows, gold_hash = read_jsonl(args.gold)
    workload = keyed(workload_rows, "workload")
    gold = keyed(gold_rows, "gold")
    same_ids(workload, gold, "workload", "gold")
    summary, predictions, prediction_hash, strategy = load_scored_predictions(
        args.summary, args.predictions, workload_hash, workload
    )
    reference_requested = args.reference_summary is not None or args.reference_predictions is not None
    if reference_requested and (args.reference_summary is None or args.reference_predictions is None):
        raise ValueError("reference summary and predictions must be supplied together")
    reference = None
    reference_meta = None
    if reference_requested:
        ref_summary, reference, ref_hash, ref_strategy = load_scored_predictions(
            args.reference_summary, args.reference_predictions, workload_hash, workload
        )
        reference_meta = {
            "prediction_sha256": ref_hash,
            "strategy": ref_strategy,
            "run": {key: ref_summary.get(key) for key in ("engine", "profile_id", "model_revision", "host", "commit", "grouping")},
        }
    if args.policy_threshold is not None and (not math.isfinite(args.policy_threshold) or not 0 <= args.policy_threshold <= 1):
        raise ValueError("policy threshold must be in [0, 1]")

    source_split = {}
    groups = defaultdict(list)
    paired_groups = defaultdict(list)
    for ident, row in gold.items():
        split = row.get("split")
        source = row.get("source_id")
        family = row.get("family")
        label = row.get("gold")
        options = row.get("options")
        if not all(isinstance(value, str) and value for value in (split, source, family, label)):
            raise ValueError(f"gold {ident}: split, source_id, family and gold are required")
        if "final" in split.casefold():
            raise ValueError(f"gold {ident}: final split remains closed")
        if not isinstance(options, list) or any(not isinstance(option, str) or not option for option in options) or len(options) != len(set(options)) or NONE not in options:
            raise ValueError(f"gold {ident}: options must be unique and include {NONE}")
        if "/" in split or "/" in family or "all" in (split, family):
            raise ValueError(f"gold {ident}: split and family cannot contain '/' or equal 'all'")
        workload_options = workload[ident].get("options", [])
        if not isinstance(workload_options, list) or any(not isinstance(option, dict) or not isinstance(option.get("id"), str) for option in workload_options):
            raise ValueError(f"workload {ident}: invalid Choice options")
        declared = {option["id"] for option in workload_options}
        if workload[ident].get("primitive") != "choice" or declared | {NONE} != set(options):
            raise ValueError(f"gold {ident}: options differ from Choice workload")
        if label not in options:
            raise ValueError(f"gold {ident}: gold is absent from options")
        if source in source_split and source_split[source] != split:
            raise ValueError(f"source {source!r} crosses evaluation splits")
        source_split[source] = split
        answer = choice_answer(predictions[ident], set(options), f"prediction {ident}")
        pair = (row, answer)
        for group in (("all", "all"), (split, "all"), (split, family)):
            groups[group].append(pair)
            if reference is not None:
                ref_answer = choice_answer(reference[ident], set(options), f"reference {ident}")
                paired_groups[group].append((row, answer, ref_answer))

    return {
        "schema": SCHEMA,
        "status": "non-final; no model or policy promotion",
        "input_sha256": {
            "workload": workload_hash,
            "gold": gold_hash,
            "predictions": prediction_hash,
        },
        "run": {
            key: summary.get(key)
            for key in ("engine", "profile_id", "model_revision", "host", "commit", "grouping")
        },
        "strategy": strategy,
        "reference": reference_meta,
        "metrics": {
            f"{split}/{family}": {
                **aggregate(pairs, args.policy_threshold),
                **({"retention_vs_reference": paired_retention(paired_groups[(split, family)])} if reference is not None else {}),
            }
            for (split, family), pairs in sorted(groups.items())
        },
        "note": "Policy results describe only the supplied locked threshold. Source counts are not confidence intervals or a release gate.",
    }


def policy_signature(answer: dict, threshold: float) -> str:
    selected = answer["choice"]
    if selected == NONE or answer["probabilities"][selected] < threshold:
        return "review"
    return f"accept:{selected}"


def compare_answers(left: dict, right: dict, threshold: float, tolerance: float) -> dict:
    keys = set(left["probabilities"])
    if keys != set(right["probabilities"]):
        raise ValueError("history comparison has different option sets")
    max_delta = max(abs(left["probabilities"][key] - right["probabilities"][key]) for key in keys)
    answer_changed = left["choice"] != right["choice"]
    policy_changed = policy_signature(left, threshold) != policy_signature(right, threshold)
    return {
        "max_probability_delta": max_delta,
        "exact_probability_match": max_delta == 0,
        "answer_changed": answer_changed,
        "policy_changed": policy_changed,
        "passed": max_delta <= tolerance and not answer_changed and not policy_changed,
    }


def score_process(args: argparse.Namespace, workload: Path, output_dir: Path, history_aba: bool) -> tuple[dict, list[dict]]:
    command = [
        str(args.bench_bin), "score", str(workload), "--engine", args.engine,
        "--output-dir", str(output_dir), "--strategies", args.strategy,
        "--reps", "1", "--no-group", "--no-warmup",
        "--bundle-root", str(args.bundle_root),
        "--checkpoint-root", str(args.checkpoint_root),
        "--tokenizer", str(args.tokenizer),
    ]
    if history_aba:
        command.append("--history-aba")
    if args.host:
        command.extend(["--host", args.host])
    if args.commit:
        command.extend(["--commit", args.commit])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"benchmark process failed ({completed.returncode}): {completed.stderr[-4000:]}")
    summary_paths = list(output_dir.glob("summary-*.json"))
    prediction_paths = list(output_dir.glob("predictions-*.jsonl"))
    if len(summary_paths) != 1 or len(prediction_paths) != 1:
        raise ValueError("benchmark process did not produce exactly one strategy result")
    summary = json.loads(summary_paths[0].read_text())
    if summary.get("warmup") is not False or summary.get("reps") != 1 or summary.get("history_aba") is not history_aba:
        raise ValueError("history process used unexpected warmup, repetition or sequence settings")
    if summary.get("fixture", {}).get("sha256") != hashlib.sha256(workload.read_bytes()).hexdigest():
        raise ValueError("history summary workload digest does not match input")
    rows, prediction_hash = read_jsonl(prediction_paths[0])
    if len(summary.get("prediction_sha256", {})) != 1 or prediction_hash not in summary["prediction_sha256"].values():
        raise ValueError("history prediction digest does not match summary")
    return summary, rows


def history_report(args: argparse.Namespace) -> dict:
    if not math.isfinite(args.tolerance) or not 0 <= args.tolerance <= 1:
        raise ValueError("tolerance must be finite and in [0, 1]")
    profile = json.loads((args.bundle_root / "profile.json").read_text())
    threshold = profile["policy"]["selected"]["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("bundle policy threshold is invalid")
    rows, workload_hash = read_jsonl(args.workload)
    if len(rows) != 2 or any(row.get("primitive") != "choice" for row in rows):
        raise ValueError("history workload must contain exactly two Choice rows: anchor then intervention")
    if rows[0].get("state") == rows[1].get("state"):
        raise ValueError("intervention must use a different state")
    if any(not isinstance(row.get("id"), str) or not row["id"] for row in rows):
        raise ValueError("history rows need nonempty request IDs")
    if rows[0]["id"] == rows[1]["id"]:
        raise ValueError("anchor and intervention need distinct request IDs")
    for row in rows:
        declared = row.get("options")
        if not isinstance(declared, list) or not declared or any(not isinstance(option, dict) or not isinstance(option.get("id"), str) or not option["id"] for option in declared):
            raise ValueError(f"history {row['id']}: invalid Choice options")
        if len({option["id"] for option in declared}) != len(declared):
            raise ValueError(f"history {row['id']}: duplicate Choice options")
    options = {option["id"] for option in rows[0]["options"]} | {NONE}
    with tempfile.TemporaryDirectory(prefix="openkind-history-") as directory:
        root = Path(directory)
        same_process_input = root / "same.jsonl"
        fresh_input = root / "fresh.jsonl"
        same_process_input.write_text("".join(json.dumps(row) + "\n" for row in rows))
        fresh_input.write_text(json.dumps(rows[0]) + "\n")
        first_summary, first_predictions = score_process(args, same_process_input, root / "same", True)
        fresh_summary, fresh_predictions = score_process(args, fresh_input, root / "fresh", False)
    if any(first_summary.get(key) != fresh_summary.get(key) for key in ("engine", "profile_id", "model_revision", "bundle_version")):
        raise ValueError("history processes used different model identities")
    if [row.get("id") for row in first_predictions] != [rows[0]["id"], rows[1]["id"], rows[0]["id"]] or [row.get("sequence_index") for row in first_predictions] != [0, 1, 2]:
        raise ValueError("history run did not emit A, B, A in order")
    if len(fresh_predictions) != 1 or fresh_predictions[0].get("id") != rows[0]["id"]:
        raise ValueError("fresh run did not emit anchor A")
    before = choice_answer(first_predictions[0], options, "anchor before")
    after = choice_answer(first_predictions[2], options, "anchor after")
    fresh = choice_answer(fresh_predictions[0], options, "fresh anchor")
    in_process = compare_answers(before, after, threshold, args.tolerance)
    reset = compare_answers(after, fresh, threshold, args.tolerance)
    initial_vs_fresh = compare_answers(before, fresh, threshold, args.tolerance)
    report = {
        "schema": "openkind-qwen-history/v1",
        "status": "non-final checkpoint-gated observation",
        "engine": first_summary["engine"],
        "profile_id": first_summary["profile_id"],
        "model_revision": first_summary["model_revision"],
        "host": first_summary["host"],
        "commit": first_summary["commit"],
        "workload_sha256": workload_hash,
        "strategy": args.strategy,
        "policy_threshold": threshold,
        "probability_tolerance": args.tolerance,
        "sequence": "identical request A, unrelated request B, A again; then A in a fresh process",
        "in_process": in_process,
        "after_vs_fresh": reset,
        "before_vs_fresh": initial_vs_fresh,
        "observations": {"before": before, "after": after, "fresh": fresh},
        "passed": in_process["passed"] and reset["passed"] and initial_vs_fresh["passed"],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    quality = commands.add_parser("choice", help="non-final labeled Choice quality report")
    for flag in ("workload", "gold", "predictions", "summary"):
        quality.add_argument(f"--{flag}", type=Path, required=True)
    quality.add_argument("--reference-summary", type=Path)
    quality.add_argument("--reference-predictions", type=Path)
    quality.add_argument("--policy-threshold", type=float)
    quality.add_argument("--output", type=Path)
    history = commands.add_parser("history", help="checkpoint-gated A/B/A/fresh process comparison")
    history.add_argument("--workload", type=Path, required=True)
    history.add_argument("--bench-bin", type=Path, required=True)
    history.add_argument("--engine", choices=("qwen35", "qwen35-mlx-fp32", "qwen35-mlx-bf16"), required=True)
    history.add_argument("--strategy", choices=("repeated_full", "nested_sequential", "nested_batched", "choose_strategy"), default="repeated_full")
    history.add_argument("--bundle-root", type=Path, required=True)
    history.add_argument("--checkpoint-root", type=Path, required=True)
    history.add_argument("--tokenizer", type=Path, required=True)
    history.add_argument("--tolerance", type=float, default=0.005)
    history.add_argument("--host")
    history.add_argument("--commit")
    history.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = quality_report(args) if args.command == "choice" else history_report(args)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    body = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    else:
        print(body, end="")
    return 1 if args.command == "history" and not report["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
