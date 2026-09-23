# OpenKind Phase 4E-A3 — QASPER audit follow-up

Run ID: `20260921T013558Z/qasper_followup_cpu_s17_v1`. Scope: CPU-only, nonfinal, error-enriched 150-row QASPER audit. The original audit packets, full paper states, model predictions, labels and gold evidence remain untouched. Training was not run and the final split was not opened.

## What was corrected

The independent 51-row adjudication was substantively valid as a historical review, but a CSV float round-trip altered the string representation of 87 source numeric cells. `AUDIT_ADJUDICATION_COMPLETED_V2.csv` reconstructs its first 20 columns exactly from the frozen 51-row packet and changes only the four adjudication columns. All 51 × 20 original source cells passed an exact equality check. The original independent file is retained for provenance.

Six low-confidence independent rows received a same-assistant follow-up against full state text. Five provisional answerable calls changed (two to `semantic_none`, three to `ambiguous`); one `semantic_none` call stood. These are **not six more independent votes**. The corrected 51-row counts are 33 answerable, 13 semantic-none, five ambiguous. The 150-row reviewed overlay is provisionally 108 answerable, 37 semantic-none, five ambiguous (policy-development: 67 / 21 / 2 of 90; calibration gate: 41 / 16 / 3 of 60). Of the 51 adjudicated rows, 33 locked semantic-none labels are proposed answerable, six locked answerable labels proposed semantic-none, and five locked semantic-none rows proposed ambiguous. These are audit-review decisions, **not** committed changes to QASPER gold.

The earlier repaired primary review had one ambiguous row of 150 (0.67%), compared with 79 of 150 (52.67%) in the first truncated-context audit. This large shift is a review-context correction, not a model improvement.

## Representation defects and external source leads

Four numeric values were lost from serialized tables/figures; one question refers to bibliography placeholders. The linked authors' papers supply possible recovery sources, but paper-version alignment and exact cell/span provenance are not verified. No external text was silently inserted into a benchmark state.

| Audit row | Problem | Source lead | Current disposition |
| --- | --- | --- | --- |
| `ok4e2_d78ba52f5ee567a3` | OpenTapioca metrics omitted; question says accuracy although paper reports F1 | [OpenTapioca Figure 2](https://arxiv.org/pdf/1904.09131) | Gate diagnostic only; comparator/metric must be specified |
| `ok4e2_fc21ba651eb25a89` | LCF-ATEPC tables omitted; gains depend on dataset, metric and baseline | [LCF-ATEPC Tables 3–4](https://arxiv.org/pdf/1912.07976) | Development source candidate; no single gain |
| `ok4e2_399e7449ffcccce6` | NarrativeQA values omitted | [NarrativeQA Table 5](https://aclanthology.org/P19-1220.pdf) | Development candidate; Masque (NQA) test B-1 54.11, B-4 30.43, METEOR 26.13, R-L 59.87, subject to version alignment |
| `ok4e2_0a598ccd447a4a6b` | User-embedding AUC table omitted | [User embeddings Table 3](https://aclanthology.org/W17-4209.pdf) | Gate diagnostic only; ueRNN vs RNN +1.28 G-DEV and +1.47 G-TEST AUC points, subject to version alignment |
| `ok4e2_7198c8c9af7a6caf` | Three model names reduced to BIBREF placeholders | [BLI paper §3](https://arxiv.org/pdf/1909.02855) | Gate diagnostic only; authors name Artetxe et al. (2016), Artetxe et al. (2017), Ruder et al. (2018) |

These PDFs may reflect newer paper revisions than the QASPER source. Table transcription and split/metric matching must be verified before a reproducible parser adds cells or resolves references. The representation candidates file deliberately labels all five `external_candidate_not_integrated_version_unverified`.

## Frozen-state evidence follow-up

Among the 11 answerable adjudications classified as `gold_evidence_incomplete`, `EVIDENCE_SPAN_CANDIDATES.csv` now carries 13 exact, uniquely located character spans across eight questions. Every short anchor is extracted from the original state text, with offsets and the state SHA-256; it is **not** substituted for locked gold evidence. Three other questions have an explicit QA gap in `EVIDENCE_SPAN_QA.json`: French/German NMT tasks do not themselves establish the authors' English source language; an assertion of neglect does not explain *why* Chinese-oriented ABSA was neglected; and an accuracy number cannot be directly compared with a separate F1 leaderboard. The twelfth missing-evidence row was downgraded to ambiguous in the six-row follow-up and is not an answerable evidence-repair candidate. These three are quarantined conceptually pending a source-backed resolution; the main ledger still describes them only as *candidates*, never as committed gold.

## Controlled next sequence

1. Reconcile the exact QASPER source paper versions, record PDF/source checksums, and implement a general table/figure/reference serializer using *policy_development* only. Preserve state hashes as an immutable baseline and emit an aligned sidecar with exact cell and page provenance.
2. Verify gold evidence for proposed label changes, explicitly quarantine ambiguous/referent-dependent rows and the six low-confidence follow-up cases, and obtain review independent of the implementation before treating the candidates as repaired gold. Do not infer evidence spans from answer notes.
3. Freeze the representation and label-repair contract, then preregister one seed-17 Phase 4E-B arm. The calibration gate is for diagnostics only; the error-enriched audit sample is not an unbiased performance estimate, and the final split stays closed until its release gate is met.

`AUDIT_REPAIR_LEDGER.csv` marks all 60 gate rows diagnostic-only. Among 90 policy-development rows it marks 20 candidate label/evidence repairs, two representation-source requirements, two ambiguous quarantines, three low-confidence follow-up quarantines and 63 no-label-change rows (the latter total includes no gate rows). These operational categories do not grant training authorization. See `PHASE4E_A3_NONFINAL_REPORT.json` and `PHASE4E_A3_LOCK.json` for the exact checksums and boundaries.
