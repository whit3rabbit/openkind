> Encoder head trained against the TypeSafe question schema. The
> upstream contract covers all three question types (`Choice`, `Score`,
> `Noul`); the family is specialised at the **schema-conformance**
> level, not at any single question type. Mirrors the
> `schema-pairs-v1` family in ollaya's nomenclature.

## Status in openkind

**Rust-loadable (prototype profile, open-weights realization).** The pinned
profile `5a7350af556f0ee66566` loads
`cross-encoder/ms-marco-MiniLM-L-6-v2` at
`233902d25c440f23af6f7d6e94d2946bac0bee0a` (Apache-2.0) through the candle
`bert` implementation with the checkpoint's tanh pooler and single-logit
classifier, FP32 on CPU. It implements `DecisionEngine` behind the bounded
family scaffold in
[`families/schema_scorer/`](../../crates/openkind-backends/src/families/schema_scorer/mod.rs),
registers in `openkindd` via `--schema-scorer-aliases` /
`--schema-scorer-model-root`, and is benchmarked through
`openkind-bench --engine schema-scorer`.

It is catalog-installable offline-first: `openkind pull schema-scorer:5a7350af556f0ee66566` downloads the pinned artifacts, verifies every SHA-256, and installs them for `--installed-models` (see [`../MODELS.md`](../MODELS.md)).

The upstream TypeSafe contract remains unowned by openkind; this profile
pins the same *architecture shape* — one `(query, passage)` row per
candidate through a scalar cross-encoder, per-question softmax — with the
Jev schema rendering defined by the family module: the query side renders
the state plus the question instruction, and each candidate criterion is
the passage side. Probability space: `ConditionalOnOfferedOptions`; the
fitted calibration temperature sharpens the scalar-logit differences
strongly (the raw relevance head produces small gaps). No M0/M2
reviewed-decision evidence exists; the recorded benchmark is request-path
timing only.

## Architectural shape

The shape below describes the **surveyed upstream contract** (the
TypeSafe-trained DeBERTa cross-encoder). The landed Rust-loadable profile
realizes the same architecture shape on open weights — the MS MARCO
MiniLM cross-encoder described under Status above.

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

## What remains open

- The upstream TypeSafe-trained schema-scorer weights, training data, and
  evaluation contract remain unowned by `openkind`. The landed profile
  realizes the same architecture shape on open weights instead; it is a
  different checkpoint, not a reproduction of the upstream model.
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
  mapping layer, so the supported wire contract
  would have to approve the
  criterion-shape constraint before a profile could land.

## Open questions

- Does the schema-scorer's question-type-dependent candidate encoding
  compose cleanly with the existing `__none__` contract owned by the
  implemented family?
- Would a schema-scorer profile be a candidate for the M3 lower-cost
  gate, or is its CPU cost
  comparable to [`encoder-nli.md`](encoder-nli.md) (which is the
  existing cross-encoder baseline)? Request-path timings for both are
  recorded in [`../BENCHMARKS.md`](../BENCHMARKS.md); the M3 comparison
  has not run.
- How does the upstream tokenizer's baked-in truncation interact with
  `openkind`'s state-length policy owned by
  [`../../crates/openkind-core/AGENTS.md`](../../crates/openkind-core/AGENTS.md)?

## What this page does not say

No model-quality or accuracy numbers, no schema-conformance rates. The cited
comparison values belong to ollaya's family page on `schema-scorer`, not
to `openkind`. Request-path timing and peak-RSS measurements for the pinned
profile are recorded in [`../BENCHMARKS.md`](../BENCHMARKS.md).
