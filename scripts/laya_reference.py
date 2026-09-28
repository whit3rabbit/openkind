#!/usr/bin/env python3
"""Run the Python laya reference over a gen_laya_fixture golden.json.

One-time operator tool (never part of builds or tests): reads the wire
requests from a Rust-generated golden fixture, maps them to laya questions
exactly as the openkind `laya` family does, runs the pinned reference
checkpoint on CPU in fp32, and writes a reference file that
`gen_laya_fixture --compare` quantifies parity against.

Wire mapping mirrored from crates/openkind-backends/src/families/laya/:

- choice: labels sorted lexicographically (the family's deterministic marker
  order); criteria carry the description string, or None when absent/empty so
  the reference renders the bare label;
- score: ordered level list, rendered `level i: <name>` by the reference;
- noul: fixed `[false, true]` pair; criteria forwarded only when non-empty so
  the reference applies its documented defaults;
- instructions: strings pass through, structured JSON serializes compactly;
- state: text passes through, objects/arrays serialize as
  `json.dumps(..., sort_keys=True, separators=(", ", ": "), ensure_ascii=False)`.

Usage:

    python3 scripts/laya_reference.py --profile english \
        --golden /tmp/laya-golden-english.json \
        --output /tmp/laya-reference-english.json
"""

import argparse
import atexit
import json
import os

PROFILES = {
    "english": "laya-english",
    "multilingual": "laya-multilingual",
    "typed-decisions": "laya-typed-decisions",
}


class TokenizerConfigGuard:
    """Restore `tokenizer/tokenizer_config.json` after the reference runs.

    The reference's `Agent` loader rewrites that file in place when the
    checkpoint ships values older transformers versions cannot parse (the
    multilingual checkpoint's list-valued `extra_special_tokens`). The
    openkind loader digest-pins the pristine hub bytes, so this guard keeps
    the model root reproducible for the Rust verification path.
    """

    def __init__(self, model_root):
        self.path = os.path.join(model_root, "tokenizer", "tokenizer_config.json")
        self.saved = None
        if os.path.isfile(self.path):
            with open(self.path, "rb") as handle:
                self.saved = handle.read()

    def restore(self):
        if self.saved is not None:
            with open(self.path, "wb") as handle:
                handle.write(self.saved)


def wire_state_to_text(state):
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))


def instruction_text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def question_to_laya(question):
    kind = question["type"]
    instructions = instruction_text(question["instructions"])
    if kind == "choice":
        criteria = {}
        for label in sorted(question["criteria"]):
            description = question["criteria"][label]
            criteria[label] = description if isinstance(description, str) and description else None
        return {"type": "choice", "instructions": instructions, "criteria": criteria}
    if kind == "score":
        return {"type": "score", "instructions": instructions, "criteria": list(question["criteria"])}
    mapped = {}
    criteria = question.get("criteria") or {}
    for key in ("false", "true"):
        value = criteria.get(key)
        if isinstance(value, str) and value:
            mapped[key] = value
    out = {"type": "noul", "instructions": instructions}
    if mapped:
        out["criteria"] = mapped
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--golden", required=True, help="golden.json written by gen_laya_fixture")
    parser.add_argument("--output", required=True, help="reference file for --compare")
    parser.add_argument("--model-root", default=None)
    args = parser.parse_args()

    model_root = args.model_root or os.path.expanduser(
        f"~/.cache/openkind/{PROFILES[args.profile]}"
    )
    from laya import Agent

    guard = TokenizerConfigGuard(model_root)
    atexit.register(guard.restore)
    agent = Agent(model_root, device="cpu")
    golden = json.load(open(args.golden, encoding="utf-8"))
    cases_out = []
    for case in golden["cases"]:
        request = case["request"]
        state_text = wire_state_to_text(request["state"])
        questions = {qid: question_to_laya(q) for qid, q in request["questions"].items()}
        result = agent.system_one(state_text, questions)
        answers = {}
        for qid, question in questions.items():
            reference = result["answers"][qid]
            answer = {"type": question["type"]}
            if question["type"] == "choice":
                answer["choice"] = reference["choice"]
                answer["probabilities"] = reference["probabilities"]
                answer["confidence"] = reference["confidence"]
            elif question["type"] == "score":
                answer["score"] = reference["score"]
                answer["legend"] = reference["legend"]
                answer["probabilities"] = reference["probabilities"]
                answer["confidence"] = reference["confidence"]
            else:
                answer["noul"] = reference["noul"]
            answers[qid] = answer
        cases_out.append(
            {
                "name": case["name"],
                "request": case["request"],
                "response": {
                    "model": request["model"],
                    "answers": answers,
                    "usage": {
                        "input_tokens": result.get("usage", {}).get("input_tokens", 0),
                        "output_tokens": 0,
                    },
                },
            }
        )
        print(f"[reference] {case['name']}: {json.dumps(answers['q0'])[:140]}")
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"profile": args.profile, "cases": cases_out}, handle, indent=2)
    guard.restore()
    print(f"[reference] wrote {args.output}")


if __name__ == "__main__":
    main()
