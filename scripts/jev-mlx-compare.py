#!/usr/bin/env python3
"""Compare OpenKind's JEV MLX engine with the mlx-vlm `feat/jev` reference.

Runs the same requests through mlx-vlm and, given the JSON printed by the
`jev_mlx_smoke` binary, reports the largest probability difference per
question. Needs Apple silicon and:

    pip install "git+https://github.com/Lazarus-931/mlx-vlm.git@6ef5c0d13b847ef2a3c3586276af9c4b75da4686"

OpenKind sorts choice labels lexicographically, so this script hands mlx-vlm the
options in that order; otherwise the two prompts would differ.
"""

import argparse
import json
from pathlib import Path

DEFAULT_REQUESTS = (
    Path(__file__).resolve().parents[1]
    / "crates/openkind-backends/tests/fixtures/jev_gev/smoke_requests.json"
)


def reference_questions(request):
    """Translate wire questions into mlx-vlm question specs."""
    questions = {}
    for name, question in sorted(request["questions"].items()):
        kind = question["type"]
        spec = {
            "type": "bool" if kind == "noul" else kind,
            "instructions": question.get("instructions") or name,
        }
        criteria = question.get("criteria")
        if kind == "choice":
            spec["criteria"] = {k: (criteria[k] or "") for k in sorted(criteria)}
        elif kind == "score":
            spec["criteria"] = criteria
        questions[name] = spec
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, help="local JEV-27B-VL-MLX-8bit directory")
    parser.add_argument("--requests", default=str(DEFAULT_REQUESTS))
    parser.add_argument("--rust", help="JSON printed by the jev_mlx_smoke binary")
    args = parser.parse_args()

    from mlx_vlm import load, predict

    model, processor = load(args.model_root)
    requests = json.loads(Path(args.requests).read_text())
    rust_cases = json.loads(Path(args.rust).read_text())["cases"] if args.rust else None
    worst = 0.0
    for index, request in enumerate(requests):
        result = predict(model, processor, request["state"], reference_questions(request))
        for name, answer in result["answers"].items():
            reference = answer.get("probabilities") or {"true": answer["probability"]}
            line = f"case {index} {name}: {json.dumps(reference, sort_keys=True)}"
            if rust_cases is not None:
                ours = rust_cases[index]["response"]["answers"][name]
                if "noul" in ours:
                    pairs = [(reference["true"], ours["noul"])]
                else:
                    pairs = [(reference[k], v) for k, v in ours["probabilities"].items()]
                gap = max(abs(a - b) for a, b in pairs)
                worst = max(worst, gap)
                line += f"  max |diff| {gap:.4f}"
            print(line)
    if rust_cases is not None:
        print(f"largest probability difference: {worst:.4f}")


if __name__ == "__main__":
    main()
