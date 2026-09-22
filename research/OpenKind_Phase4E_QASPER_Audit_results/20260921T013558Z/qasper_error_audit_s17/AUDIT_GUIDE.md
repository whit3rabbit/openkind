# OpenKind Phase 4E QASPER Audit Guide

Audit experiment: `055683587cc016e33cfd889795d5d70f9e16623acf9c75bcb3b5db9ae276ee63`

Review `AUDIT_REVIEW_BLINDED.csv` without opening `AUDIT_KEY.parquet`.

For every row, complete:

- `review_target_kind`: exactly `answerable`, `semantic_none`, or `ambiguous`;
- `review_reason_code`: a concise reusable category such as `explicit_answer`, `implicit_answer`, `missing_evidence`, `question_underspecified`, `option_mismatch`, `annotation_disagreement`, or `other`;
- `review_confidence`: a number from 0 through 1;
- `review_notes`: a short evidence-based explanation; and
- `reviewer_id`: a stable reviewer identifier.

Save the completed file as `AUDIT_REVIEW_COMPLETED.csv` in this directory. Do not reorder, delete, duplicate, or add rows; do not edit IDs or source text. Do not use model predictions, none scores, error types, or locked labels during review.

Phase 4E-B is never automatically authorized. After analysis, resolve ambiguous cases and reviewer/locked-label disagreements, then preregister either data repair or one representation-aware seed-17 experiment. Final remains closed.
