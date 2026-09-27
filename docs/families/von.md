# Family: von

> Encoder head trained against TypeSafe's published `von` reference
> contract.

## Status in openkind

**Blocked — external-reference-only (re-evaluated 2026-09-26).** Same
disposition as [`kev.md`](kev.md): `openkind` does not own the `von`
weights, training pipeline, or evaluation contract, and no open checkpoint
reproduces the reference. The family's identity *is* the upstream contract
— unlike the [`schema-scorer`](schema-scorer.md) shape, whose scalar
cross-encoder architecture transfers to an open checkpoint, an encoder head
"trained against the `von` contract" cannot be realized without that
contract. It stays blocked until OpenKind owns or is granted the reference,
or defines its own successor contract on a new family page.

Re-verified during the 2026-09-26 unblocking push: a Hugging Face search
found no `von` decision-model checkpoint, and the reference author whose
open release unblocked [`kev`](kev.md) has published no `von`-line
artifact. The kev unblock therefore does not extend to this family.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Encoder backbone with a custom decision head trained against TypeSafe's published `von` reference contract |
| Tokenization | Per the published `von` reference layout |
| Forward pattern | Per the published reference |
| Readout | Per the published reference |
| Continuation state | Per the published reference |
| Text generation | None |
| Calibration | Per the published reference |
| Semantic none | Per the published reference |

The contract-level details are owned by the upstream `von` reference
spec. `openkind` does not reproduce them here.

## Why this family is a candidate

- Provides a second external comparison point against TypeSafe's
  published reference behavior, distinct from `kev` because the
  reference covers a different decision shape.

## What blocks implementation

Same blockers as [`kev.md`](kev.md): `openkind` does not own the
weights, training pipeline, or evaluation contract, and reproducing the
contract is out of scope. The architecture rules in
[`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) require explicit
ownership of every external contract.

## Open questions

- Is the `von` contract a useful benchmark for the M2 reviewed-decision
  gate, or is it redundant with `kev`?
- Can `openkind` evaluate against `von` without reproducing the
  contract? Methodology is owned by [`../../BENCHMARKS.md`](../../BENCHMARKS.md).

## What this page does not say

No timings, no accuracy numbers, no contract deltas. The cited
comparison values belong to ollaya's family page on `von`, not to
`openkind`. `openkind` has no measurements to report.
