# OpenKind Phase 4E-A4 — audited preflight

Run label: `20260921T013558Z/qasper_source_alignment_cpu_s17_v1`. This is a CPU audit, with **no model training and no final split access**. The source-alignment section of the companion notebook still needs to run against the pinned Hugging Face QASPER source. The locally completed preflight is versioned here.

## Frozen A3 result

A3 reported 13 exact frozen-state span candidates across eight questions, with three unresolved evidence cases. Rechecking the offsets against the immutable `states.parquet` and its recorded SHA-256 confirmed that **12 spans across seven questions** belong to policy development. The remaining span/question is from the calibration gate and stays diagnostic only. Candidate spans are short exact substrings, not proven complete support for an answer and not committed gold evidence.

All three unresolved cases were still counted in A3 as `candidate_label_and_evidence_repair`. `AUDIT_REPAIR_LEDGER_V2.csv` preserves all 150 rows and all fields except those three `action` values, which become `quarantine_unresolved_evidence`. The policy-development dispositions are: 17 candidate label/evidence repairs, three unresolved-evidence quarantines, two ambiguous quarantines, three low-confidence follow-up quarantines, two representation-source requirements, and 63 no-label-change rows (90 in total). The 60 calibration-gate rows remain diagnostic only.

| Review ID | Reason candidate is held |
| --- | --- |
| `ok4e2_2e059c32a14d29c3` | French and German NMT task names are present; the English-source-language statement describes cited earlier work, and the authors' exact language coverage is not established by the current state. |
| `ok4e2_63fa0f4da9616df9` | The paper reports neglected Chinese-oriented work and emphasis on polarity over extraction, but no reliable text explains why Chinese work was neglected. |
| `ok4e2_1f5356f7d2f80896` | The proposed method's 99.53% accuracy and a leaderboard's 82.9% F1 are different metrics, so they cannot answer a precise numerical state-of-the-art comparison. |

These are same-assistant disposition checks, **not independent reviews**. The A3 corrected 51-row adjudication and the frozen original labels/evidence remain unchanged. The error-enriched 150-row pack cannot provide an unbiased QASPER prevalence or performance estimate.

## Controlled source check in the notebook

The notebook downloads only upstream **train** and **validation** from pinned `allenai/qasper` parquet commit `06806e4608976fc2fac0a090ac425d5b2b29caf4`. Four unique policy-development papers cover two missing-table-value cases and three unresolved questions (one paper overlaps). It checks source ID, split, title, abstract, paragraphs and hashes, then emits original dataset figure/table captions as candidate leads. The upstream `test` parquet includes the closed final partition and is **never downloaded**. Published PDF version equivalence and missing table-cell values are *not* established by this check.

The output contract explicitly keeps `automatic_training_authorization=false` and `final_opened=false`. A later preregistered Phase 4E-B arm needs source-aligned representation provenance and a reviewed label/evidence contract; it must not treat audit-review proposals as silently modified benchmark gold.
