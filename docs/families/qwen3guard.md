# Family: qwen3guard

> Decoder preset that generates a fixed safety-verdict text and is mapped
> to typed questions through an embedded question schema. Mirrors the
> `qwen3guard-gen-v1` family in ollaya's nomenclature, used there to
> serve a guardrail preset (`Safe` / `Unsafe` / `Controversial` plus
> nine category labels).

## Status in openkind

**Surveyed.** No profile, no vendored parity fixtures, no daemon
registration. `openkind` does not adopt the upstream `qwen3guard` name
as a commitment to any specific checkpoint; the name is preserved here
so future selection work can pick up the family with the same vocabulary.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Decoder-only fine-tune of a small Qwen-3-class backbone, trained to emit a fixed safety-verdict text in a chat-template-bound format |
| Tokenization | Hard-coded chat template carrying the policy, the nine category labels, and the verdict output format; the user message is only the text to judge |
| Forward pattern | Single forward pass per request; the model emits a short verdict (`Safety: Safe\|Unsafe\|Controversial` plus a categories line) |
| Readout | String-parse the verdict back into a typed distribution over the verdict classes plus the nine categories; mapping onto `Choice` / `Score` / `Noul` happens at the runtime boundary |
| Continuation state | Decoder KV cache; no DeltaNet or convolution state on the published Qwen-3 0.6B backbone |
| Text generation | Yes — limited to the verdict text. `openkind`'s general rule against autoregressive text generation is family-specific, not global; this family is the documented exception and is reviewed per profile |
| Calibration | Per-profile temperature fitted offline; calibrated probabilities are derived from the verdict-string parse, not from raw logits |
| Semantic none | The `Controversial` class absorbs the failure-to-decide mass; `openkind`'s `__none__` semantics are owned by [`encoder-state-first.md`](encoder-state-first.md) and any profile here would have to be reviewed against them |

## Why this family is a candidate

- Guardrail decisions are a common request shape on top of decision
  models and would let `openkind` serve a known customer use case
  directly without forcing the caller to write their own guardrail
  classifier.
- A guardrail-specialised head could be a smaller model than the
  current encoder-state-first profile and could serve as a candidate
  for the M3 lower-cost gate in [`../../ROADMAP.md`](../../ROADMAP.md).
- The family ships as a fixed-preset model (the question schema is
  embedded), which simplifies the wire contract for the caller and
  removes the need for arbitrary-question prompt engineering.

## What blocks implementation

- `openkind`'s general architecture rule
  ([`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)) forbids
  autoregressive text generation in the engine. Adopting this family
  would be a documented family-level exception, not a relaxation of
  the global rule; the exception would need its own review against
  the M0/M1/M2 milestone sequence in
  [`../../ROADMAP.md`](../../ROADMAP.md).
- The published upstream preset ignores arbitrary custom
  instructions; any profile here can answer only the embedded
  question schema. Questions outside the embedded schema must be
  rejected at the wire with a 422.
- The M0 supported workload in [`../../ROADMAP.md`](../../ROADMAP.md)
  has not selected guardrail decisions as in-scope, and no guardrail
  training or evaluation contract is owned by `openkind`.
- The verdict is parsed back from generated text, which adds a
  parsing failure mode absent from the implemented
  encoder-state-first family. The parsing contract would need its
  own fixtures and review.

## Open questions

- Does the M0 supported workload in
  [`../../ROADMAP.md`](../../ROADMAP.md) authorize guardrail-shaped
  decisions?
- How would the family's guardrail class set interact with the
  `__none__` semantic-none contract owned by the implemented
  encoder-state-first family?
- Does the verdict-parse failure mode belong in the wire error model
  owned by [`../../crates/openkind-core/AGENTS.md`](../../crates/openkind-core/AGENTS.md),
  or as a profile-level rejection? That decision is unresolved.

## What this page does not say

No timings, no accuracy numbers, no guardrail-class distributions. The
cited comparison values belong to ollaya's family page on
`qwen3guard-gen-v1`, not to `openkind`. `openkind` has no measurements
to report.
