"""Phase 4E-A3: CPU-only, nonfinal, immutable-source QASPER audit follow-up.

This script is embedded verbatim in the companion Colab notebook. It prepares
versioned research overlays; it does not train, mutate a benchmark, or open final.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


BASE_RUN_ID = "20260921T013558Z"
SCHEMA = "openkind-phase4e-a3-qasper-followup/v1"
SOURCE_HASHES = {
    "packet": "2d6b3be4dfa532de7c0c320d4e84b67d2be571ff4ddcc7f886b22f437e88edeb",
    "independent_csv": "aaec70a8ae64deebc45c33945748480eafd472c0b00b467422f15a36e5278701",
    "independent_report": "30181a4492880d1d39d00aa89760ac561a5dbb5c9794cb27de17160fe9e7b4d2",
    "primary_review": "ca5e1ccbd31e575fce4ec97b07f93bc7c5e215ed6a796902f8eb2a1821eed4d8",
    "audit_key": "690f429cff7ab653e2b4ea0db8847701d4121e3842132c57625bc64a9b39f69e",
    "states": "7e6333202ad934e66148724a565b27176fcaad4133a566d2c8f8b57044ef3120",
    "questions": "549b446f609a9407eceb5df7ef4c0f14779bc73d72396ae749424626d8777434",
    "evidence": "566e0e69e8f57ed0ce626021647a3b770a89ae5a3eb318e1d4f8bea284f2c6be",
}
REVIEWER_ID = "openai_codex_followup_20260923_not_independent"
FOLLOWUP = {
    "ok4e2_e0bce4d10d224a08": (
        "semantic_none", "insufficient_state_evidence", "medium",
        "The state identifies a 2015 Wikipedia dump and English word examples, "
        "but never specifies which Wikipedia language edition was used. "
        "Examples do not prove the training languages.",
    ),
    "ok4e2_083b976c9596994b": (
        "ambiguous", "question_underspecified", "medium",
        "The paper calls its proposed system an MLP and also compares LSTM, CNN, "
        "SVM, logistic regression and ridge classifiers. It does not designate a "
        "unique 'baseline system'; the referent is unspecified.",
    ),
    "ok4e2_34e501e6024d2440": (
        "ambiguous", "question_underspecified", "medium",
        "The state compares an eight-language MLDoc corpus with earlier "
        "four-language Reuters subsets and lists multiple training sizes, but the "
        "question asks for a difference in size versus an unidentified previous model.",
    ),
    "ok4e2_a68941f0e2145806": (
        "ambiguous", "question_underspecified", "medium",
        "BERT has 95.7% accuracy for the paper's relation-scoring subtask; its "
        "full KBQA system has 70.45% F1. Neither establishes which model is "
        "state of the art for the unspecified task in the question.",
    ),
    "ok4e2_9b456e5390cda66e": (
        "semantic_none", "insufficient_state_evidence", "medium",
        "The experiments list Twitter keyword/event datasets and English-looking "
        "examples, but do not specify the language coverage of every reported "
        "result. The independent semantic_none decision stands.",
    ),
    "ok4e2_1dcb9834a58c5cd5": (
        "semantic_none", "insufficient_state_evidence", "medium",
        "The state identifies English for MultiNLI, but does not establish the "
        "language coverage of every medical QA and entailment resource. The "
        "universal 'only English' claim is not supported by the supplied state.",
    ),
}
EXTERNAL_CANDIDATES = {
    "ok4e2_d78ba52f5ee567a3": {
        "paper_url": "https://arxiv.org/pdf/1904.09131",
        "location": "Figure 2, PDF page 8 (zero-based page 7)",
        "finding": "The figure gives InKB micro/macro F1, not an unqualified accuracy. OpenTapioca scores include 0.870/0.858 on ISTEX-1000 and 0.335/0.310 on RSS-500; comparator, dataset and metric must be specified.",
    },
    "ok4e2_fc21ba651eb25a89": {
        "paper_url": "https://arxiv.org/pdf/1912.07976",
        "location": "Tables 3 and 4, PDF pages 14–15 (zero-based 13–14)",
        "finding": "Chinese and SemEval results span ATE macro-F1, APC accuracy and APC macro-F1 over multiple datasets and baselines. No single 'how much better' scalar is well defined.",
    },
    "ok4e2_399e7449ffcccce6": {
        "paper_url": "https://aclanthology.org/P19-1220.pdf",
        "location": "Table 5, PDF page 8 (zero-based 7)",
        "finding": "Masque (NQA) on the NarrativeQA test set: BLEU-1 54.11, BLEU-4 30.43, METEOR 26.13, ROUGE-L 59.87. Distinct validation scores also appear; do not conflate splits.",
    },
    "ok4e2_0a598ccd447a4a6b": {
        "paper_url": "https://aclanthology.org/W17-4209.pdf",
        "location": "Table 3, PDF page 3 (zero-based 2)",
        "finding": "AUC ueRNN minus text-only RNN is +1.28 points G-DEV (80.68−79.40) and +1.47 points G-TEST (80.71−79.24). These are percentage-point differences, not relative percentages.",
    },
    "ok4e2_7198c8c9af7a6caf": {
        "paper_url": "https://arxiv.org/pdf/1909.02855",
        "location": "Section 3, PDF page 4 (zero-based 3)",
        "finding": "The paper names Artetxe et al. (2016), Artetxe et al. (2017), and Ruder et al. (2018) as its three evaluated BLI models; the serialized QASPER state contains BIBREF placeholders instead.",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def emit_csv(path: Path, records, fields):
    # Data-engineering CSV serialization; canonical 20 source fields are checked
    # cell-for-cell below. CSV is not passed through a floating-point dataframe.
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def emit_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def inputs_for_drive(root: Path):
    a2 = root / "OpenKind_Phase4E_A2_QASPER_Audit_Repair_results" / BASE_RUN_ID / "qasper_error_audit_repair_s17"
    corpus = root / "OpenDecision_Phase4A_StateQuery_results" / BASE_RUN_ID / "opendecision-mq-v1"
    return {
        "packet": a2 / "AUDIT_ADJUDICATION_PACKET.csv",
        "independent_csv": a2 / "AUDIT_ADJUDICATION_COMPLETED.csv",
        "independent_report": a2 / "AUDIT_ADJUDICATION_REPORT.json",
        "primary_review": a2 / "AUDIT_REVIEW_COMPLETED.csv",
        "audit_key": a2 / "AUDIT_KEY.parquet",
        "states": corpus / "states.parquet",
        "questions": corpus / "questions.parquet",
        "evidence": corpus / "evidence.parquet",
    }


def inputs_for_local(root: Path):
    return {
        "packet": root / "analysis" / "AUDIT_ADJUDICATION_PACKET.csv",
        "independent_csv": root / "adjudicated" / "AUDIT_ADJUDICATION_COMPLETED.csv",
        "independent_report": root / "adjudicated" / "AUDIT_ADJUDICATION_REPORT.json",
        "primary_review": root / "current" / "AUDIT_REVIEW_COMPLETED.csv",
        "audit_key": root / "local_drive" / "OpenKind_Phase4E_A2_QASPER_Audit_Repair_results" / BASE_RUN_ID / "qasper_error_audit_repair_s17" / "AUDIT_KEY.parquet",
        "states": root / "states.parquet",
        "questions": root / "questions.parquet",
        "evidence": root / "evidence.parquet",
    }


def run(input_paths: dict[str, Path], output: Path, *, open_final=False, train=False):
    if open_final or train:
        raise PermissionError("Phase 4E-A3 is CPU audit only: training and final are closed.")
    if set(input_paths) != set(SOURCE_HASHES):
        raise AssertionError("Unexpected input set; do not silently omit a source.")
    for name, expected in SOURCE_HASHES.items():
        actual = sha256_file(input_paths[name])
        if actual != expected:
            raise AssertionError(f"Frozen {name} hash differs: {actual} != {expected}")

    packet, source_fields = rows(input_paths["packet"])
    independent, full_fields = rows(input_paths["independent_csv"])
    primary, _ = rows(input_paths["primary_review"])
    prior_report = json.loads(input_paths["independent_report"].read_text())
    key = pd.read_parquet(input_paths["audit_key"])
    states = pd.read_parquet(input_paths["states"])
    questions = pd.read_parquet(input_paths["questions"])
    evidence = pd.read_parquet(input_paths["evidence"])
    assert len(packet) == len(independent) == 51 and len(primary) == len(key) == 150
    assert source_fields == full_fields and len(source_fields) == 24
    immutable_fields = source_fields[:20]
    assert full_fields[20:] == [
        "adjudicated_target_kind", "adjudication_reason_code", "adjudication_notes", "adjudicator_id"
    ]
    assert prior_report["counts_by_adjudicated_target_kind"] == {
        "answerable": 38, "semantic_none": 11, "ambiguous": 2
    }
    by_source = {item["review_id"]: item for item in packet}
    by_independent = {item["review_id"]: item for item in independent}
    by_primary = {item["review_id"]: item for item in primary}
    by_key = key.set_index("review_id")
    by_state = states.set_index("state_id")
    by_question = questions.set_index("question_id")
    assert len(by_source) == len(by_independent) == 51
    assert set(by_source) == set(by_independent) and set(by_primary) == set(by_key.index)
    assert set(FOLLOWUP).issubset(by_source)
    float_roundtrips = 0
    for rid, original in by_source.items():
        converted = by_independent[rid]
        for field in immutable_fields:
            if original[field] != converted[field]:
                if field not in {"none_probability", "operating_threshold"}:
                    raise AssertionError(f"Non-numeric source cell changed: {rid}/{field}")
                if not math.isclose(float(original[field]), float(converted[field]), rel_tol=0, abs_tol=1e-12):
                    raise AssertionError(f"Numeric source cell changed materially: {rid}/{field}")
                float_roundtrips += 1
    assert float_roundtrips == 87, f"Unexpected float-cell drift: {float_roundtrips}"

    corrected = []
    followup_rows = []
    for source in packet:
        rid = source["review_id"]
        individual = by_independent[rid]
        record = {**source, **{field: individual[field] for field in full_fields[20:]}}
        if rid in FOLLOWUP:
            label, reason, confidence, why = FOLLOWUP[rid]
            previous = record["adjudicated_target_kind"]
            record.update(adjudicated_target_kind=label, adjudication_reason_code=reason,
                          adjudication_notes="Follow-up review (not independent): " + why,
                          adjudicator_id=REVIEWER_ID)
            state = by_state.loc[source["state_id"]]
            followup_rows.append({
                "review_id": rid, "partition": source["partition"],
                "question_id": source["question_id"], "instruction": source["instruction"],
                "source_state_sha256": state["text_sha256"],
                "independent_label": previous, "followup_label": label,
                "followup_reason": reason, "confidence": confidence,
                "rationale": why, "reviewer_id": REVIEWER_ID,
            })
        corrected.append(record)
    assert len(corrected) == 51
    for record in corrected:
        source = by_source[record["review_id"]]
        assert all(record[field] == source[field] for field in immutable_fields), "Immutable CSV source cell changed"
        assert record["adjudicated_target_kind"] in {"answerable", "semantic_none", "ambiguous"}

    corrected_by_id = {record["review_id"]: record for record in corrected}
    evidence_count = Counter(evidence.question_id)
    ledger = []
    for rid, original_review in by_primary.items():
        freeze = by_key.loc[rid]
        assert freeze["question_id"] == original_review["question_id"]
        assert freeze["state_id"] == original_review["state_id"]
        state = by_state.loc[freeze["state_id"]]
        question = by_question.loc[freeze["question_id"]]
        assert state["split"] == freeze["partition"] == question["split"]
        assert state["source"] == "qasper"
        state_hash = hashlib.sha256(original_review["state_text"].encode("utf-8")).hexdigest()
        assert state_hash == original_review["state_text_sha256"] == state["text_sha256"]
        if rid in corrected_by_id:
            adjud = corrected_by_id[rid]
            final_label = adjud["adjudicated_target_kind"]
            reason = adjud["adjudication_reason_code"]
            provenance = "independent_adjudication_plus_followup" if rid in FOLLOWUP else "independent_adjudication"
        else:
            final_label = original_review["review_target_kind"]
            reason = original_review["review_reason_code"]
            provenance = "primary_agreed_locked_not_independently_adjudicated"
            assert final_label == freeze["locked_target_kind"]
        if freeze["partition"] == "calibration_gate":
            action = "gate_diagnostic_only"
        elif final_label == "ambiguous":
            action = "quarantine_ambiguous"
        elif rid in FOLLOWUP:
            action = "quarantine_low_confidence_followup"
        elif reason in {"missing_table_values", "unresolved_reference_placeholder"}:
            action = "representation_source_required"
        elif final_label != freeze["locked_target_kind"]:
            action = "candidate_label_and_evidence_repair"
        else:
            action = "no_label_change"
        ledger.append({
            "review_id": rid, "partition": freeze["partition"],
            "selection_role": freeze["selection_role"],
            "state_id": freeze["state_id"], "question_id": freeze["question_id"],
            "source_state_sha256": state_hash, "locked_target_kind": freeze["locked_target_kind"],
            "primary_target_kind": original_review["review_target_kind"],
            "independent_target_kind": by_independent[rid]["adjudicated_target_kind"] if rid in by_independent else "",
            "followup_target_kind": final_label, "review_provenance": provenance,
            "reason_code": reason, "existing_gold_evidence_count": evidence_count[freeze["question_id"]],
            "label_disagrees_with_locked": final_label != freeze["locked_target_kind"],
            "action": action,
        })

    representation = []
    for rid, candidate in EXTERNAL_CANDIDATES.items():
        record = corrected_by_id[rid]
        state = by_state.loc[record["state_id"]]
        representation.append({
            "review_id": rid, "partition": record["partition"],
            "question_id": record["question_id"], "state_id": record["state_id"],
            "state_sha256": state["text_sha256"], "source_title": json.loads(state["metadata_json"]).get("title", ""),
            "defect": record["adjudication_reason_code"], "primary_paper_url": candidate["paper_url"],
            "candidate_location": candidate["location"], "source_finding": candidate["finding"],
            "status": "external_candidate_not_integrated_version_unverified",
            "proposed_action": "Align PDF revision with QASPER source; capture table/figure/reference with exact provenance; validate parser on development only; keep gate diagnostic and final closed",
        })
    assert Counter(r["defect"] for r in representation) == {"missing_table_values": 4, "unresolved_reference_placeholder": 1}

    counts = Counter(row["followup_target_kind"] for row in ledger)
    by_partition = defaultdict(Counter)
    for row in ledger:
        by_partition[row["partition"]][row["followup_target_kind"]] += 1
    source_changed = sum(by_source[rid][field] != by_independent[rid][field]
                         for rid in by_source for field in immutable_fields)
    assert source_changed == 87
    assert counts == {"answerable": 108, "semantic_none": 37, "ambiguous": 5}, counts
    assert Counter(r["adjudicated_target_kind"] for r in corrected) == {
        "answerable": 33, "semantic_none": 13, "ambiguous": 5
    }
    assert all(r["action"] == "gate_diagnostic_only" for r in ledger if r["partition"] == "calibration_gate")

    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "corrected_adjudication": output / "AUDIT_ADJUDICATION_COMPLETED_V2.csv",
        "followup": output / "SIX_ROW_FOLLOWUP_REVIEW.csv",
        "ledger": output / "AUDIT_REPAIR_LEDGER.csv",
        "representation": output / "REPRESENTATION_SOURCE_CANDIDATES.csv",
        "adjudication_report": output / "AUDIT_ADJUDICATION_REPORT_V2.json",
        "report": output / "PHASE4E_A3_NONFINAL_REPORT.json",
        "manifest": output / "PHASE4E_A3_LOCK.json",
    }
    csv_payloads = [
        ("corrected_adjudication", corrected, full_fields),
        ("followup", followup_rows, list(followup_rows[0])),
        ("ledger", ledger, list(ledger[0])),
        ("representation", representation, list(representation[0])),
    ]
    for name, values, fields in csv_payloads:
        target = outputs[name]
        if target.exists():
            # Never silently replace a prior versioned artifact.
            existing, existing_fields = rows(target)
            if existing != [{field: str(item[field]) for field in fields} for item in values] or existing_fields != fields:
                raise FileExistsError(f"Output differs from existing file; use a new run label: {target}")
        else:
            emit_csv(target, values, fields)
    written, fields = rows(outputs["corrected_adjudication"])
    assert fields == full_fields
    assert all(w[field] == by_source[w["review_id"]][field] for w in written for field in immutable_fields)
    def matrix(row_key, col_key):
        return [
            {row_key: left, col_key: right, "n": n}
            for (left, right), n in sorted(Counter((row[row_key], row[col_key]) for row in corrected).items())
        ]
    adjudication_report = {
        "schema": "openkind-phase4e-a3-adjudication-v2/v1",
        "source_packet_sha256": SOURCE_HASHES["packet"],
        "independent_adjudication_sha256": SOURCE_HASHES["independent_csv"],
        "v2_adjudication_sha256": sha256_file(outputs["corrected_adjudication"]),
        "immutable_first_20_columns_cell_identical": True,
        "numeric_source_cells_restored_from_packet": float_roundtrips,
        "n": 51,
        "counts_by_target_kind": dict(sorted(Counter(r["adjudicated_target_kind"] for r in corrected).items())),
        "counts_by_reason_code": dict(sorted(Counter(r["adjudication_reason_code"] for r in corrected).items())),
        "counts_by_partition": {
            name: dict(sorted(Counter(r["adjudicated_target_kind"] for r in corrected if r["partition"] == name).items()))
            for name in ("policy_development", "calibration_gate")
        },
        "locked_vs_v2": matrix("locked_target_kind", "adjudicated_target_kind"),
        "primary_vs_v2": matrix("review_target_kind", "adjudicated_target_kind"),
        "followup_review_ids": list(FOLLOWUP),
        "followup_reviewer_not_independent": REVIEWER_ID,
        "original_gold_evidence_unchanged": True,
        "automatic_phase4e_b_authorization": False,
        "final_opened": False,
    }
    if outputs["adjudication_report"].exists():
        assert json.loads(outputs["adjudication_report"].read_text()) == adjudication_report
    else:
        emit_json(outputs["adjudication_report"], adjudication_report)
    report = {
        "schema": SCHEMA, "base_run_id": BASE_RUN_ID,
        "source_sha256": SOURCE_HASHES,
        "independent_csv_roundtripped_source_cells": float_roundtrips,
        "immutable_source_cells_after_v2": True,
        "adjudicated_51_v2": dict(sorted(Counter(r["adjudicated_target_kind"] for r in corrected).items())),
        "full_150_provisional": dict(sorted(counts.items())),
        "full_150_by_partition": {k: dict(sorted(v.items())) for k, v in sorted(by_partition.items())},
        "followup_rows": 6, "changed_from_independent": 5,
        "representation_candidates": len(representation),
        "provenance_note": "Six follow-up reviews are by the same assistant preparing this workflow, not an additional independent adjudicator. The other 45 independent rows retain independent decisions.",
        "selection_bias_note": "The 150 were selected for error-focused audit. Counts and model outcomes on this pack are not unbiased QASPER performance estimates; do not retune on calibration_gate.",
        "repair_status": "versioned decisions and source leads only; no original labels, gold spans or state strings mutated",
        "external_source_version_status": "not verified against original QASPER paper revisions; not integrated into state or training",
        "automatic_phase4e_b_authorization": False,
        "train_executed": False, "final_opened": False,
        "required_next_decision": "Version-align source paper/figure/table captures and validate a general representation repair on policy_development; then preregister one seed-17 arm before any training. Gate stays diagnostic and final stays closed.",
    }
    if outputs["report"].exists():
        assert json.loads(outputs["report"].read_text()) == report
    else:
        emit_json(outputs["report"], report)
    digest = {name: sha256_file(path) for name, path in outputs.items() if name != "manifest"}
    lock = {"schema": SCHEMA + "-lock", "input_sha256": SOURCE_HASHES,
            "output_sha256": digest, "final_opened": False, "train_executed": False}
    if outputs["manifest"].exists():
        assert json.loads(outputs["manifest"].read_text()) == lock
    else:
        emit_json(outputs["manifest"], lock)
    print(json.dumps(report, indent=2))
    print("Artifacts:", {name: str(path) for name, path in outputs.items()})
    return report, outputs


if __name__ == "__main__":
    local = os.environ.get("OPENKIND_A3_LOCAL_INPUT")
    source = inputs_for_local(Path(local)) if local else inputs_for_drive(Path(os.environ.get(
        "OPENKIND_COLAB_ROOT", "/content/drive/MyDrive/Colab Notebooks")))
    destination = Path(os.environ.get("OPENKIND_A3_OUTPUT", str(Path(os.environ.get(
        "OPENKIND_COLAB_ROOT", "/content/drive/MyDrive/Colab Notebooks")) /
        "OpenKind_Phase4E_A3_QASPER_Followup_results" / BASE_RUN_ID / "qasper_followup_cpu_s17_v1")))
    run(source, destination, open_final=False, train=False)
