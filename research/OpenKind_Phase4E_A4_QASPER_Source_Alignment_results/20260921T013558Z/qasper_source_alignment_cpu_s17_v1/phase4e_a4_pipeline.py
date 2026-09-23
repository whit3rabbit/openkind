"""CPU-only Phase 4E-A4 preflight and pinned QASPER source alignment.

Emits versioned research overlays only. Original labels, evidence, states, model
predictions and final data are never written by this module.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd


RUN_ID = "20260921T013558Z"
RUN_LABEL = "qasper_source_alignment_cpu_s17_v1"
REPO = "allenai/qasper"
DATA_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
EXPECTED_INPUTS = {
    "ledger": "46d5f69317afd377d664837efae5f6e365398e6402cc6e8719010a8f4f994721",
    "evidence_spans": "634f5ad51b7c94dcb468dcbcca69ee9204197766d02bb9f14fe296e230412909",
    "evidence_qa": "215af447abde527a3489c51afda32d304648055f5ce76fda957551e3bc0d6279",
    "adjudication_v2": "ffc83cc7cdbc70b5b390b74ec0194e9ff71f5f6578e5f7579465cbe6ef7d1bc7",
    "representation": "9a1ebb2c49d641e74950fdec46aa1b782706a6ca822165a5d180339872ea3325",
    "states": "7e6333202ad934e66148724a565b27176fcaad4133a566d2c8f8b57044ef3120",
    "manifest": "c68b813e04eeff671c0c92fc6f7d7253f5d6671e76b54439c3e4c426f55ff815",
}
UNRESOLVED = {
    "ok4e2_1f5356f7d2f80896": "No matched state-of-the-art metric/comparator supports the requested comparison.",
    "ok4e2_2e059c32a14d29c3": "French/German task names do not prove all model input/output languages.",
    "ok4e2_63fa0f4da9616df9": "The source reports Chinese ABSA neglect but does not explain why.",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_once(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f"Existing result differs; use a fresh RUN_LABEL: {path}"
    else:
        path.write_bytes(data)


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_once(path, stream.getvalue().encode("utf-8"))


def write_json(path: Path, obj: dict):
    write_once(path, (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def input_paths(a3: Path, corpus: Path) -> dict[str, Path]:
    return {
        "ledger": a3 / "AUDIT_REPAIR_LEDGER.csv",
        "evidence_spans": a3 / "EVIDENCE_SPAN_CANDIDATES.csv",
        "evidence_qa": a3 / "EVIDENCE_SPAN_QA.json",
        "adjudication_v2": a3 / "AUDIT_ADJUDICATION_COMPLETED_V2.csv",
        "representation": a3 / "REPRESENTATION_SOURCE_CANDIDATES.csv",
        "states": corpus / "states.parquet",
        "manifest": corpus / "manifest.json",
    }


def prepare(inputs: dict[str, Path], output: Path):
    assert set(inputs) == set(EXPECTED_INPUTS)
    for key, expected in EXPECTED_INPUTS.items():
        assert sha(inputs[key]) == expected, f"Frozen {key} differs from A3 input: {inputs[key]}"
    manifest = json.loads(inputs["manifest"].read_text())
    source = manifest["sources"]["qasper"]
    assert source["repo"] == REPO and source["data_revision"] == DATA_REVISION
    assert manifest["tables"]["states"] == EXPECTED_INPUTS["states"]

    ledger = csv_read(inputs["ledger"])
    spans = csv_read(inputs["evidence_spans"])
    qa_a3 = json.loads(inputs["evidence_qa"].read_text())
    refs = csv_read(inputs["representation"])
    adjudication = csv_read(inputs["adjudication_v2"])
    states = pd.read_parquet(inputs["states"]).set_index("state_id", verify_integrity=True)
    assert len(ledger) == 150 and len(spans) == 13 and len(refs) == 5 and len(adjudication) == 51
    assert set(qa_a3["answerable_questions_with_unresolved_evidence"]) == set(UNRESOLVED)
    assert qa_a3["candidate_spans"] == 13 and qa_a3["candidate_answerable_questions"] == 8
    assert qa_a3["adjudication_v2_sha256"] == EXPECTED_INPUTS["adjudication_v2"]
    assert qa_a3["original_gold_evidence_mutated"] is False and qa_a3["final_opened"] is False

    revised = []
    for r in ledger:
        rec = dict(r)
        if r["review_id"] in UNRESOLVED:
            assert r["partition"] == "policy_development"
            assert r["action"] == "candidate_label_and_evidence_repair"
            rec["action"] = "quarantine_unresolved_evidence"
        revised.append(rec)
    assert len({r["review_id"] for r in revised}) == 150
    assert all(x["action"] == "gate_diagnostic_only" for x in revised if x["partition"] == "calibration_gate")
    policy = []
    for r in spans:
        row = next(x for x in ledger if x["review_id"] == r["review_id"])
        state = states.loc[r["state_id"]]
        text = state["text"]
        assert row["state_id"] == r["state_id"] and row["question_id"] == r["question_id"]
        assert row["source_state_sha256"] == r["state_text_sha256"] == state["text_sha256"]
        assert hashlib.sha256(text.encode()).hexdigest() == r["state_text_sha256"]
        a, b = int(r["char_start"]), int(r["char_end"])
        assert text[a:b] == r["short_exact_anchor"] and text.count(r["short_exact_anchor"]) == 1
        assert r["status"] == "candidate_span_not_gold_evidence"
        if r["partition"] == "policy_development":
            policy.append(r)
    assert len(policy) == 12 and len({x["review_id"] for x in policy}) == 7
    assert len([x for x in spans if x["partition"] == "calibration_gate"]) == 1
    counts = Counter((x["partition"], x["action"]) for x in revised)
    assert counts[("policy_development", "candidate_label_and_evidence_repair")] == 17
    assert counts[("policy_development", "quarantine_unresolved_evidence")] == 3
    assert all(x["action"] == "candidate_label_and_evidence_repair" for x in revised if x["review_id"] in {p["review_id"] for p in policy})
    write_csv(output / "AUDIT_REPAIR_LEDGER_V2.csv", revised, list(ledger[0]))
    write_csv(output / "EVIDENCE_SPAN_POLICY_CANDIDATES.csv", policy, list(spans[0]))
    report = {
        "schema": "openkind-phase4e-a4-preflight/v1", "run_id": RUN_ID,
        "a3_input_sha256": EXPECTED_INPUTS,
        "revised_policy_actions": {action: n for (partition, action), n in sorted(counts.items()) if partition == "policy_development"},
        "calibration_gate_diagnostic_only": 60,
        "all_exact_candidate_spans": 13, "policy_development_exact_candidate_spans": 12,
        "policy_development_questions_with_exact_candidates": 7,
        "unresolved_policy_review_ids": sorted(UNRESOLVED),
        "source_repo": REPO, "source_parquet_commit": DATA_REVISION,
        "original_gold_evidence_mutated": False, "locked_labels_mutated": False,
        "automatic_training_authorization": False, "final_opened": False,
        "important_limit": "Exact state spans and data-parquet alignment do not establish published PDF version equivalence or independent adjudication.",
    }
    write_json(output / "PHASE4E_A4_PREFLIGHT.json", report)
    return refs, revised, states, report


def _list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def align_pinned_qasper(refs: list[dict], inputs: dict[str, Path], output: Path,
                        cache: Path, *, allow_download: bool):
    """Inspect the exact original QASPER parquet commit; emit source leads only.

    No data are borrowed from paper PDFs. Published PDF revisions are still not
    proven equivalent to these dataset source rows.
    """
    ledger = csv_read(inputs["ledger"])
    chosen = [x for x in ledger if x["review_id"] in UNRESOLVED]
    # Upstream test contains the custom final partition, so it is never fetched.
    # Gate leads stay in the frozen A3 packet for diagnostic provenance only.
    policy_refs = [x for x in refs if x["partition"] == "policy_development"]
    targets = {(x["state_id"].split(":")[1], x["state_id"].split(":")[2]): x["state_id"] for x in policy_refs + chosen}
    assert len(policy_refs) == 2 and len(targets) == 4
    assert {split for split, _ in targets} == {"train", "validation"}
    states = pd.read_parquet(inputs["states"]).set_index("state_id", verify_integrity=True)
    source_rows = []
    parquet_hashes = {}
    for split in ("train", "validation"):
        filename = f"qasper/{split}/0000.parquet"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{DATA_REVISION}/{filename}"
        path = cache / (split + "-0000.parquet")
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(f"Missing pinned QASPER parquet for {split}; enable source download")
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            assert payload[:4] == payload[-4:] == b"PAR1", f"Not parquet: {url}"
            write_once(path, payload)
        assert path.read_bytes()[:4] == path.read_bytes()[-4:] == b"PAR1"
        parquet_hashes[split] = {"url": url, "sha256": sha(path), "bytes": path.stat().st_size}
        df = pd.read_parquet(path)
        assert len(df) in range(200, 1000), f"Unexpected QASPER {split} source count"
        for _, source in df.iterrows():
            paper_id = str(source["id"])
            if (split, paper_id) in targets:
                source_rows.append((split, paper_id, source))
    assert len(source_rows) == len(targets), "Pinned data does not contain exactly one row per audited paper"
    aligned, captions = [], []
    for split, paper_id, source in source_rows:
        state_id = targets[(split, paper_id)]
        state = states.loc[state_id]
        frozen_text = state["text"]
        assert state["upstream_split"] == split and state["source"] == "qasper"
        assert hashlib.sha256(frozen_text.encode()).hexdigest() == state["text_sha256"]
        assert source["title"] == json.loads(state["metadata_json"])["title"]
        abstract_match = str(source["abstract"]) in frozen_text
        body = source["full_text"]
        paras = [str(p) for section in _list(body["paragraphs"]) for p in _list(section) if str(p).strip()]
        hits = sum(1 for p in paras if p in frozen_text)
        row = {
            "state_id": state_id, "source_split": split, "paper_id": paper_id,
            "source_parquet_sha256": parquet_hashes[split]["sha256"],
            "source_title": source["title"], "frozen_state_sha256": state["text_sha256"],
            "title_exact": source["title"] in frozen_text,
            "abstract_exact": abstract_match,
            "paragraphs_total": len(paras), "paragraphs_exactly_present": hits,
            "paragraph_exact_coverage": round(hits / len(paras), 5) if paras else 0,
            "figure_table_captions": len(_list(source["figures_and_tables"]["caption"])),
            "status": "source_dataset_aligned" if abstract_match and hits == len(paras) else "inspect_source_dataset_difference",
        }
        aligned.append(row)
        for index, caption in enumerate(_list(source["figures_and_tables"]["caption"])):
            text = str(caption or "").strip()
            if not text:
                continue
            captions.append({
                "state_id": state_id, "source_split": split,
                "source_parquet_sha256": parquet_hashes[split]["sha256"],
                "caption_index": index, "caption_text": text,
                "in_frozen_state": text in frozen_text,
                "status": "caption_candidate_not_pdf_table_cells_or_gold",
            })
    aligned.sort(key=lambda x: x["state_id"])
    captions.sort(key=lambda x: (x["state_id"], x["caption_index"]))
    write_csv(output / "QASPER_SOURCE_ROW_ALIGNMENT.csv", aligned, list(aligned[0]))
    if captions:
        write_csv(output / "QASPER_SOURCE_CAPTION_CANDIDATES.csv", captions, list(captions[0]))
    report = {
        "schema": "openkind-phase4e-a4-pinned-qasper-alignment/v1", "source_repo": REPO,
        "parquet_revision": DATA_REVISION, "split_sources": parquet_hashes,
        "audited_papers": len(aligned), "audited_policy_papers": len(aligned),
        "policy_caption_candidates": len(captions),
        "source_dataset_aligned": sum(x["status"] == "source_dataset_aligned" for x in aligned),
        "pdf_revision_equivalence_proven": False, "table_values_recovered": 0,
        "upstream_test_downloaded": False, "locked_gold_mutated": False, "automatic_training_authorization": False,
        "final_opened": False,
    }
    write_json(output / "QASPER_SOURCE_ALIGNMENT_QA.json", report)
    return report
