#!/usr/bin/env python3
"""Cross-check a gen_drex_fixture golden.json against the bundled Drex reference.

One-time operator tool (never part of builds or tests). It loads the pinned
nace-ai/drex-v1.5 checkpoint with the reference `code/kev/` package shipped in
the checkpoint directory, replays every request in the Rust-generated golden
fixture, and reports the largest probability difference and any argmax flip.

Arithmetic: the weights load in BF16 (FP32 would need about 36 GB). Each
decoder layer, the embedding output, and the final norm are widened to FP32
only while they run, so the reference computes in FP32 over BF16-rounded
weights. That is the same arithmetic as the Rust oracle
(`candle-cpu-bf16w-fp32c-qwen35-text`) at a peak near the BF16 footprint.

Wire mapping mirrored from crates/openkind-backends/src/families/drex/:
Choice criteria are sorted by key before scoring, because the Rust engine
scores options in sorted label order and the reference keeps caller order.

The reference code is Nace.AI's (Apache-2.0). Running it imports
`code/kev/` from the model root. Review it at the pinned revision first.

Usage:

    ~/.cache/openkind/venvs/drex-ref/bin/python scripts/drex_reference.py \\
        --model-root ~/.cache/openkind/drex-v1.5 \\
        --golden crates/openkind-backends/tests/fixtures/drex_2d495e07ea056d6c4317/golden.json \\
        --output /path/to/drex-reference.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def widen_while_running(module, torch):
    """Run `module` in FP32 and restore its BF16 storage afterwards."""

    def before(mod, _args):
        mod.float()

    def after(mod, _args, _output):
        mod.to(torch.bfloat16)

    module.register_forward_pre_hook(before)
    module.register_forward_hook(after)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.model_root.resolve() / "code"))
    import torch
    import transformers
    from kev import api, checkpoint, model as kev_model

    torch.set_grad_enabled(False)
    tok, model = checkpoint.load(str(args.model_root), device="cpu", dtype=torch.bfloat16)
    lm = model.lm
    # The embedding output is upcast exactly (every BF16 value is an FP32
    # value), so the residual stream is FP32 from the first layer on.
    lm.embed_tokens.register_forward_hook(lambda _m, _a, out: out.float())
    for layer in lm.layers:
        widen_while_running(layer, torch)
    lm.norm.float()

    golden = json.loads(args.golden.read_text())
    results = []
    worst = 0.0
    flips = []
    for case in golden["cases"]:
        request = dict(case["request"])
        questions = {}
        for qid, question in request["questions"].items():
            question = dict(question)
            if question["type"] == "choice":
                question["criteria"] = dict(sorted(question["criteria"].items()))
            questions[qid] = question
        request["questions"] = questions
        parsed = api.SystemOneRequest(**request)
        record, meta = api.to_record(parsed)
        enc = model.encode(
            tok,
            record,
            max_state=kev_model.SERVE_MAX_STATE,
            max_branch=kev_model.SERVE_MAX_BRANCH,
            strict=True,
        )
        started = time.perf_counter()
        probs = [p.float().tolist() for p in model.probs(enc)]
        elapsed = time.perf_counter() - started

        expected_answers = case["response"]["answers"]
        case_result = {"name": case["name"], "seconds": round(elapsed, 2), "questions": {}}
        for p, m in zip(probs, meta):
            expected = expected_answers[m["id"]]
            if m["type"] == "noul":
                ours = {"true": expected["noul"]}
                theirs = {"true": p[1]}
            else:
                ours = expected["probabilities"]
                theirs = dict(zip(m["keys"], p))
            diff = max(abs(ours[k] - theirs[k]) for k in ours)
            worst = max(worst, diff)
            if m["type"] != "noul":
                if max(ours, key=ours.get) != max(theirs, key=theirs.get):
                    flips.append(f"{case['name']}/{m['id']}")
            elif (ours["true"] >= 0.5) != (theirs["true"] >= 0.5):
                flips.append(f"{case['name']}/{m['id']}")
            case_result["questions"][m["id"]] = {
                "reference": theirs,
                "openkind": ours,
                "max_abs_diff": diff,
            }
        print(f"[reference] {case['name']:<18} {elapsed:7.1f} s", flush=True)
        results.append(case_result)

    report = {
        "profile_id": golden["profile_id"],
        "backbone": golden["backbone"],
        "reference": {
            "code": "checkpoint code/kev/ at the pinned revision",
            "arithmetic": "bf16 weights, per-layer fp32 compute, cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "max_abs_probability_diff": worst,
        "argmax_flips": flips,
        "cases": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[reference] max |p_ref - p_openkind| = {worst:.3e}; argmax flips: {flips or 'none'}")


if __name__ == "__main__":
    main()
