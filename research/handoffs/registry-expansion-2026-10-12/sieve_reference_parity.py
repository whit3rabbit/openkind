#!/usr/bin/env python3
"""Reference-side parity evidence for the pinned sieve-4b profile.

Runs the actual shipped reference logic (the checkpoint's sieve4b package,
imported verbatim) over the real tokenizer, and records per-case prefix and
branch token statistics for comparison against the Rust renderer.

Usage:
  python3 sieve_reference_parity.py <model-root> <out-json>

The model root must contain checkpoint/tokenizer.json and
checkpoint/sieve_config.json at the pinned revision. Nothing here is wired
into builds or tests; the recorded hashes are pinned into the golden
fixture provenance.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sieve4b-ref"))
from sieve4b.prompt import PromptBuilder  # noqa: E402  (the shipped reference)

from tokenizers import Tokenizer  # noqa: E402


class _Enc:
    """HF-tokenizer-shaped wrapper: .encode(text) returns a list of ids."""

    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, index):
        return self.ids[index]

    def __eq__(self, other):
        return list(self.ids) == list(other)

    def __bool__(self):
        return bool(self.ids)

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __add__(self, other):
        return _Enc(list(self.ids) + list(other))


class _Tok:
    def __init__(self, inner):
        self.inner = inner

    def encode(self, text, add_special_tokens=False):
        return _Enc(self.inner.encode(text, add_special_tokens=False).ids)


# The fixture cases, in gen_sieve_4b_fixture.rs order. Each entry is
# (name, state, question-dict) in the engine's API shape.
CASES = [
    ("choice-two", "The kitchen light is on. The office light is off.",
     {"type": "choice", "instructions": "Which option is explicitly recorded in the state?",
      "criteria": {"kitchen": "the room whose light is on", "office": None}}),
    ("choice-five-text-state",
     "Incident record: severity high, service billing, region eu-west, status mitigated, owner oncall.",
     {"type": "choice", "instructions": "Which option is explicitly recorded in the state?",
      "criteria": {"owner": None, "region": None, "service": None, "severity": None,
                   "status": None}}),
    ("noul-default-criteria", {"ticket": {"escalated": True, "priority": "p1"}},
     {"type": "noul", "instructions": "Is the ticket escalated?"}),
    ("noul-described-criteria", "Sensor report: temperature 41.2C, threshold 40C.",
     {"type": "noul", "instructions": "Did the temperature exceed the threshold?",
      "criteria": {"false": "The temperature is at or below the threshold.",
                   "true": "The temperature is above the threshold."}}),
    ("score-three-levels",
     "Queue depth is 420 events; the SLO allows up to 500 with a soft warn at 350.",
     {"type": "score", "instructions": "How loaded is the queue?",
      "criteria": ["low", "moderate", "high"]}),
    ("choice-empty-state", "",
     {"type": "choice", "instructions": "Reply with any option.",
      "criteria": {"alpha": None, "beta": None}}),
    ("choice-none-option", "The request matches neither the allowlist nor the denylist.",
     {"type": "choice", "instructions": "Which list applies?",
      "criteria": {"__none__": "neither list applies", "allowlist": None, "denylist": None}}),
]

# Listing orders the Rust engine uses: identity for choice/score, Yes-first
# and No-first for noul.
def listing_orders(kind: str):
    if kind == "noul":
        return [[1, 0], [0, 1]]
    return None  # identity, length-dependent


def ids_digest(ids) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def main() -> int:
    model_root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    tokenizer = Tokenizer.from_file(str(model_root / "checkpoint" / "tokenizer.json"))
    sieve_config = json.loads((model_root / "checkpoint" / "sieve_config.json").read_text())
    builder = PromptBuilder(_Tok(tokenizer))

    # The derived code table must match the pinned sieve_config.
    matches = (builder.codes == list(sieve_config["codes"])
               and builder.code_ids == list(sieve_config["code_ids"]))
    print(f"code-table derivation matches sieve_config: {matches}")
    if not matches:
        return 1

    prefix_ids = builder.prefix("state-placeholder")  # stats only, not per-case
    results = []
    for name, state, question in CASES:
        kind = question["type"]
        state_rendered = builder.prefix(state)
        enc = builder.encode(state, [question], noul_both_orders=(kind == "noul"))
        entry = {"name": name, "prefix_token_count": len(enc.prefix_ids),
                 "prefix_ids_sha256": ids_digest(enc.prefix_ids), "branches": []}
        for branch in enc.branches:
            entry["branches"].append({
                "token_count": len(branch.ids) + len(enc.prefix_ids),
                "branch_ids_sha256": ids_digest(enc.prefix_ids + branch.ids),
                "n_opts": branch.n_opts,
                "perm": branch.perm,
            })
        assert enc.prefix_ids == state_rendered, f"{name}: prefix is state-independent"
        results.append(entry)
        print(f"  {name}: prefix {len(enc.prefix_ids)} tokens, "
              f"{len(entry['branches'])} branch(es)")

    document = {
        "repository": "sthanika-ai/Sieve-4B",
        "model_root_revision": "ea7cd1edee29590a82ed9fbfcb644227a8eed5f1",
        "code_table_matches_sieve_config": matches,
        "cases": results,
    }
    out_path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
