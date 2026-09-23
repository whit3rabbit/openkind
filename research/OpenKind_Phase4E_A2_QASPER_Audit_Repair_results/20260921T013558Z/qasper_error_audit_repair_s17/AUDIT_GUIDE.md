# OpenKind Phase 4E-A2 QASPER Repaired Audit Guide

Audit experiment: `89d0e283a2b743efcb95de3f73189c6fb0157620c5f6252beea149a44d1ecefd`

Review `AUDIT_REVIEW_BLINDED.csv` without opening `AUDIT_KEY.parquet` or `AUDIT_EVIDENCE_LOCKED.parquet`.

For each row, decide whether the **paper text in `state_text`** contains enough information to answer `instruction`:

- `answerable`: the state text explicitly or implicitly supports an answer;
- `semantic_none`: the state text does not contain enough information to answer; or
- `ambiguous`: the question/state is genuinely underspecified, contradictory, or otherwise cannot be classified after reading the state.

Complete every reviewer field:

- `review_target_kind`;
- `review_reason_code` (recommended: `explicit_answer`, `implicit_answer`, `missing_evidence`, `question_underspecified`, `conflicting_evidence`, `other`);
- `review_confidence` from 0 through 1;
- `review_notes`; and
- `reviewer_id`.

Save the completed file as `AUDIT_REVIEW_COMPLETED.csv` in this directory. Do not reorder, delete, duplicate, or add rows; do not edit IDs, hashes, lengths, or source text. Do not use gold evidence, model predictions, none scores, error types, locked labels, or the superseded review during the primary review.

After the primary analysis, inspect only the rows emitted to `AUDIT_ADJUDICATION_PACKET.csv`. Phase 4E-B is never automatically authorized. Final remains closed.
