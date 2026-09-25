> Encoder head trained against the TypeSafe question schema. The
> upstream contract covers all three question types (`Choice`, `Score`,
> `Noul`); the family is specialised at the **schema-conformance**
> level, not at any single question type. Mirrors the
> `schema-pairs-v1` family in ollaya's nomenclature.

## Status in openkind

**Surveyed — contract mapping only.** No profile, no vendored parity
fixtures, no daemon registration. See [`kev.md`](kev.md) for the
broader rationale on contract-mapping families.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | DeBERTa-v3-large cross-encoder with a single scalar logit, fine-tuned against TypeSafe's published question schema |
| Tokenization | One sequence per `(state, question-schema + candidate)` pair; the question schema is rendered into the hypothesis text alongside the candidate description |
| Forward pattern | One row per candidate across every question; softmax over each question's candidates gives the answer |
| Readout | Per-question softmax over the candidate set; `Choice` argmax, `Score` ordered-level mapping, `Noul` over the `true`/`false` pair |
| Continuation state | KV cache only; family does not exercise hybrid-state backbones |
| Text generation | None |
| Calibration | Per-profile temperature fitted offline |
| Semantic none | Schema-conformance failure class; the `__none__` semantics owned by [`encoder-state-first.md`](encoder-state-first.md) would have to be reviewed against the schema-scorer contract |

The contract-level details are owned by the upstream family spec.
`openkind` does not reproduce them here.

## Why this family is a candidate

- The schema-scorer contract spans all three question types
  (`Choice`, `Score`, `Noul`), so a single profile would cover the
  full Jev surface rather than a single question type. That is the
  main architectural advantage over [`encoder-nli.md`](encoder-nli.md)
  and [`encoder-instruct-label.md`](encoder-instruct-label.md), each of
  which is restricted to a narrower question set.
- The candidate encoding folds the question schema into the hypothesis,
  so the same model can answer arbitrary typed questions without
  per-question prompt engineering.
- The single-logit cross-encoder head is small enough to fit comfortably
  on CPU-only Apple Silicon with the same latency profile as
  [`encoder-nli.md`](encoder-nli.md).

## What blocks implementation

- `openkind` does not own the schema-scorer weights, training data, or
  evaluation contract. Same blockers as [`kev.md`](kev.md) and
  [`von.md`](von.md).
- The tokenization folds the rendered question schema into the
  hypothesis, which means the candidate encoding is question-type
  dependent. Maintaining the rendering in lockstep with the Jev wire
  contract owned by
  [`../../crates/openkind-engine/AGENTS.md`](../../crates/openkind-engine/AGENTS.md)
  is a non-trivial maintenance surface.
- The semantic-none handling owned by the implemented family already
  reserves the `__none__` option key. Any schema-scorer profile would
  have to be reviewed against that contract; ollaya notes that list-
  valued choice criteria must round-trip without an Ollaya-side
  mapping layer, so the M0 wire contract in
  [`../../ROADMAP.md`](../../ROADMAP.md) would have to approve the
  criterion-shape constraint before a profile could land.

## Open questions

- Does the schema-scorer's question-type-dependent candidate encoding
  compose cleanly with the existing `__none__` contract owned by the
  implemented family?
- Would a schema-scorer profile be a candidate for the M3 lower-cost
  gate in [`../../ROADMAP.md`](../../ROADMAP.md), or is its CPU cost
  comparable to [`encoder-nli.md`](encoder-nli.md) (which is the
  existing cross-encoder baseline)? No measurements exist.
- How does the upstream tokenizer's baked-in truncation interact with
  `openkind`'s state-length policy owned by
  [`../../crates/openkind-core/AGENTS.md`](../../crates/openkind-core/AGENTS.md)?

## What this page does not say

No timings, no accuracy numbers, no schema-conformance rates. The cited
comparison values belong to ollaya's family page on `schema-scorer`, not
to `openkind`. `openkind` has no measurements to report.
