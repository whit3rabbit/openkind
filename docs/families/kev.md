# Family: kev

> LoRA adapter + pointer head on a Qwen base, trained against the
> TypeSafe `kev` reference contract.

## Status in openkind

**Surveyed — contract mapping only.** No profile, no vendored parity
fixtures, no daemon registration. The family page exists to record that
the architectural pattern was reviewed as part of the family survey and
to document why it is not in the implementation backlog.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Qwen-class decoder backbone with a small LoRA adapter and a pointer head (two `Linear(hidden → 256)` projections, option score `k(h_opt) · q(h_decide) / 16`) |
| Tokenization | Per the published `kev` reference layout |
| Forward pattern | Per the published reference |
| Readout | Per the published reference |
| Continuation state | Per the published reference |
| Text generation | None |
| Calibration | Per the published reference |
| Semantic none | Per the published reference |

The contract-level details are owned by the upstream `kev` reference
spec. `openkind` does not reproduce them here; if they change, this
page does not need to change.

## Why this family is a candidate

- Provides a third-party benchmark point against TypeSafe's published
  reference behavior.
- Useful as a calibration reference when comparing `openkind`'s
  encoder-state-first probabilities against TypeSafe's reference model
  for the same task.

## What blocks implementation

- `openkind` does not own the `kev` training pipeline, weights, or
  evaluation contract. Re-implementing the family would require
  reproducing a reference that `openkind` does not have the right to
  redistribute.
- The family is interesting only as an external comparison point. Any
  release-quality claim in `openkind` is owned by
  [`../../ROADMAP.md`](../../ROADMAP.md); the family would not change
  those claims.
- Reproducing an external contract adds a dependency surface whose
  maintenance and version-pinning story `openkind` does not currently
  own. The architecture rules in
  [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) require every
  external contract to be owned explicitly.

## Open questions

- Is there a non-reproducing way to use `kev` as a benchmark
  reference for the implemented encoder-state-first family?
  Methodology for external comparison is owned by
  [`../../BENCHMARKS.md`](../../BENCHMARKS.md).
- Does the published contract drift, and how would `openkind` track
  drift without owning the contract?

## What this page does not say

No timings, no accuracy numbers, no contract deltas. The cited
comparison values belong to ollaya's family page on `kev`, not to
`openkind`. `openkind` has no measurements to report.
