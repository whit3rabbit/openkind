#!/usr/bin/env python3
"""Reference-side parity evidence for the pinned d3-edge profile.

Runs the actual shipped reference logic (research/.../reference-code/
d3_format.py, imported verbatim) over the checkpoint's real tokenizer and
renders the checkpoint's real chat_template.jinja with a jinja2 shim, then
records per-case prompt byte/token statistics for comparison against the
Rust renderer.

Usage:
  python3 d3_reference_parity.py <model-root> <out-json> [repo_id] [revision]

The model root must contain tokenizer.json, chat_template.jinja,
decision_config.json, and config.json at the pinned revision. Nothing here
is wired into builds or tests; the recorded hashes are pinned into the
golden fixture provenance.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REFERENCE = Path(__file__).resolve().parent / "reference-code"
sys.path.insert(0, str(REFERENCE))
import d3_format  # noqa: E402  (the shipped reference module, Apache-2.0)

from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def raise_exception(message):
    raise ValueError(message)


def tojson(value, ensure_ascii=False, **_):
    return json.dumps(value, ensure_ascii=ensure_ascii, separators=(",", ":"))


def load_template(model_root: Path):
    source = (model_root / "chat_template.jinja").read_text()
    environment = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
    environment.globals["raise_exception"] = raise_exception
    environment.filters["tojson"] = tojson
    return environment.from_string(source)


# The fixture cases, in gen_d3_edge_fixture.rs order. `options` lists the
# rendered option texts in engine code order (wire Choice renders the
# wire's sorted label order; Noul the fixed false/true pair; Score the
# ordered levels).
CASES = [
    ("choice-two", "The kitchen light is on. The office light is off.",
     {"type": "choice", "instructions": "Which option is explicitly recorded in the state?",
      "criteria": {"kitchen": "the room whose light is on", "office": None}},
     ["kitchen: the room whose light is on", "office"]),
    ("choice-five-text-state",
     "Incident record: severity high, service billing, region eu-west, status mitigated, owner oncall.",
     {"type": "choice", "instructions": "Which option is explicitly recorded in the state?",
      # the wire's criteria map is unordered; the engine renders the wire's
      # sorted label order, so the reference input is given in that order
      "criteria": {"owner": None, "region": None, "service": None, "severity": None,
                   "status": None}},
     ["owner", "region", "service", "severity", "status"]),
    # the object state is given in the wire's serialized key order (the
    # openkind wire map is unordered; the engine renders exactly the bytes
    # it received)
    ("noul-default-criteria", {"ticket": {"escalated": True, "priority": "p1"}},
     {"type": "noul", "instructions": "Is the ticket escalated?"},
     ["No / false", "Yes / true"]),
    ("noul-described-criteria", "Sensor report: temperature 41.2C, threshold 40C.",
     {"type": "noul", "instructions": "Did the temperature exceed the threshold?",
      "criteria": {"false": "The temperature is at or below the threshold.",
                   "true": "The temperature is above the threshold."}},
     ["The temperature is at or below the threshold.",
      "The temperature is above the threshold."]),
    ("score-three-levels",
     "Queue depth is 420 events; the SLO allows up to 500 with a soft warn at 350.",
     {"type": "score", "instructions": "How loaded is the queue?",
      "criteria": ["low", "moderate", "high"]},
     ["low", "moderate", "high"]),  # levels render by text alone, in order
    ("choice-empty-state", "",
     {"type": "choice", "instructions": "Reply with any option.",
      "criteria": {"alpha": None, "beta": None}},
     ["alpha", "beta"]),
    ("choice-none-option", "The request matches neither the allowlist nor the denylist.",
     {"type": "choice", "instructions": "Which list applies?",
      "criteria": {"__none__": "neither list applies", "allowlist": None, "denylist": None}},
     ["__none__: neither list applies", "allowlist", "denylist"]),
]


def stats(text: str) -> dict:
    return {
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_bytes": len(text.encode("utf-8")),
    }


def main() -> int:
    model_root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    repo_id = sys.argv[3] if len(sys.argv) > 3 else "vllm-sr/d3-edge"
    revision = sys.argv[4] if len(sys.argv) > 4 else "18d61df78684675463af9ae732872478845c5a79"
    tokenizer = Tokenizer.from_file(str(model_root / "tokenizer.json"))
    decision_config = json.loads((model_root / "decision_config.json").read_text())
    template = load_template(model_root)

    # 1. Derivation prefix from the real template (bare user message,
    #    add_generation_prompt=True, enable_thinking=False).
    prefix = template.render(
        messages=[{"role": "user", "content": "Choose an option."}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids

    # 2. Reference answer_codes derivation over the real vocabulary.
    codes, ids = [], []
    for code in d3_format.candidate_codes():
        if len(codes) == d3_format.MAX_OPTIONS:
            break
        encoded = tokenizer.encode(code, add_special_tokens=False).ids
        if len(encoded) != 1 or encoded[0] in ids:
            continue
        if tokenizer.encode(prefix + code, add_special_tokens=False).ids != prefix_ids + encoded:
            continue
        codes.append(code)
        ids.append(encoded[0])
    derivation_matches = (codes == list(decision_config["codes"])
                          and ids == list(decision_config["token_ids"]))
    print(f"code-table derivation matches decision_config: {derivation_matches}")
    if not derivation_matches:
        first_bad = next(
            (i for i, (a, b) in enumerate(zip(codes, decision_config["codes"])) if a != b),
            min(len(codes), len(decision_config["codes"])),
        )
        print(f"  first divergence at {first_bad}: derived={codes[first_bad:first_bad+4]} "
              f"pinned={decision_config['codes'][first_bad:first_bad+4]}")
        return 1

    # 3. Per-case prompts: d3_format.user_prompt wrapped by the real template.
    results = []
    for name, state, question, options in CASES:
        if question["type"] == "score":
            # The runtime normalizes Score to a choice over the level texts
            # (d3_runtime.normalize_question renders criteria as
            # {text: None} with code order = level order); options() is
            # choice/noul only, so serve the rendered form onward.
            question = {
                "type": "choice",
                "instructions": question["instructions"],
                "criteria": {text: None for text in question["criteria"]},
            }
            texts = list(question["criteria"])
        else:
            _, texts = d3_format.options(question)
        assert texts == options, f"{name}: option texts drifted from the reference contract"
        user = d3_format.user_prompt(state, question, codes)
        prompt = template.render(
            messages=d3_format.messages(state, question, codes),
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert user in prompt, f"{name}: the template must embed the reference user block"
        encoding = tokenizer.encode(prompt, add_special_tokens=False)
        entry = {"name": name, "option_texts": texts, **stats(prompt)}
        entry["token_count"] = len(encoding.ids)
        entry["token_ids_sha256"] = hashlib.sha256(
            b",".join(str(i).encode() for i in encoding.ids)
        ).hexdigest()
        results.append(entry)
        print(f"  {name}: {entry['token_count']} tokens, prompt {entry['prompt_bytes']} bytes")

    document = {
        "repository": repo_id,
        "model_root_revision": revision,
        "format_id": decision_config["format_id"],
        "derivation_prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "code_table_matches_decision_config": derivation_matches,
        "cases": results,
    }
    out_path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
