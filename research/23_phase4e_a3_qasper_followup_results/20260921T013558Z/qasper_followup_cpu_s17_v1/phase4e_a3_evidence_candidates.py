"""Extract conservative, exact state-span leads for incomplete gold evidence.

Only candidate offsets are produced; locked QASPER evidence is never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ANCHORS = {
    "ok4e2_d028985da8e1da5c": [
        "sharing of its second-order term, referred to as the intermediate state",
    ],
    "ok4e2_02a83790d9770e0a": [
        "lexical,implicit incongruity and explicit incongruity",
        "readability and word count of the text",
    ],
    "ok4e2_f810e03cf231c09f": [
        "Unigrams and Pragmatic features",
        "Stylistic patterns BIBREF4 and patterns related to situational disparity",
        "Hastag interpretations",
    ],
    "ok4e2_5ebfa3821daae755": [
        "out-perform our Lead-1-AMR baseline by 0.3 ROGUE-1",
    ],
    "ok4e2_be57f840ebfa16ba": [
        "We carefully removed the bias present in the dataset (e.g., the speakers' reputations, popularity gained by publicity",
    ],
    "ok4e2_59ecbec749bd0dc5": [
        "contains all states who voted and delivered a GD statement",
    ],
    "ok4e2_e5ecb3990e0484a7": [
        "4.85% WER on the VLSP 2018 and 15.09% WER on the VLSP 2019",
    ],
    "ok4e2_32a61ef8b3e30284": [
        "Rebuttal is a statement that attacks the claim",
        "refutation – which is used for attacking the rebuttal",
        "backing as an additional support to the whole argument",
    ],
}
GAPS = {
    "ok4e2_2e059c32a14d29c3": "The 'shared encoder for English' text describes prior cited work; the authors' own task names French and German NMT but does not explicitly establish English as its own source language in the serialized state.",
    "ok4e2_63fa0f4da9616df9": "The paper asserts that Chinese-oriented ABSA is neglected and that prior work emphasizes polarity over extraction, but does not clearly answer why Chinese-oriented research is neglected.",
    "ok4e2_1f5356f7d2f80896": "99.53% accuracy is reported for the proposed method and 82.9% F1 for a shared-task leader; these cannot support a numerical apples-to-apples comparison with state of the art.",
}


def extract(v2_csv: Path, output_dir: Path):
    records = list(csv.DictReader(v2_csv.open(newline="", encoding="utf-8")))
    by_id = {record["review_id"]: record for record in records}
    assert len(records) == len(by_id) == 51
    assert set(ANCHORS).isdisjoint(GAPS)
    assert all(by_id[r]["adjudication_reason_code"] == "gold_evidence_incomplete" for r in set(ANCHORS) | set(GAPS))
    assert len(ANCHORS) + len(GAPS) == 11
    spans = []
    for rid, anchors in ANCHORS.items():
        r = by_id[rid]
        t = r["state_text"]
        for anchor in anchors:
            assert t.count(anchor) == 1, f"Anchor missing or nonunique: {rid} / {anchor}"
            start = t.index(anchor)
            end = start + len(anchor)
            assert t[start:end] == anchor
            spans.append({
                "review_id": rid, "partition": r["partition"], "state_id": r["state_id"],
                "question_id": r["question_id"],
                "state_text_sha256": hashlib.sha256(t.encode("utf-8")).hexdigest(),
                "char_start": start, "char_end": end,
                "short_exact_anchor": anchor,
                "status": "candidate_span_not_gold_evidence",
                "provenance": "frozen_qasper_state_text",
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "EVIDENCE_SPAN_CANDIDATES.csv"
    if csv_path.exists():
        saved = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        assert saved == [{k: str(v) for k, v in row.items()} for row in spans]
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(spans[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(spans)
    qa = {
        "schema": "openkind-phase4e-a3-evidence-span-candidates/v1",
        "adjudication_v2_sha256": hashlib.sha256(v2_csv.read_bytes()).hexdigest(),
        "source": "frozen_state_text_in_v2; every span exact and unique",
        "candidate_answerable_questions": len(ANCHORS),
        "candidate_spans": len(spans),
        "answerable_questions_with_unresolved_evidence": GAPS,
        "ambiguous_gold_evidence_case": "ok4e2_a68941f0e2145806; not treated as an answerable evidence repair",
        "original_gold_evidence_mutated": False,
        "automatic_training_authorization": False,
        "final_opened": False,
    }
    qa_path = output_dir / "EVIDENCE_SPAN_QA.json"
    if qa_path.exists():
        assert json.loads(qa_path.read_text()) == qa
    else:
        qa_path.write_text(json.dumps(qa, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Exact candidate spans: {len(spans)} across {len(ANCHORS)} questions; unresolved: {len(GAPS)}; no gold edited.")
    return csv_path, qa_path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    extract(here / "phase4e_a3_final/AUDIT_ADJUDICATION_COMPLETED_V2.csv", here / "phase4e_a3_final")
