# Family: encoder-instruct-label

> Instruction-tuned encoder, all candidate label markers in one sequence,
> span pooling, sigmoid per label.

## Status in openkind

**Surveyed.** No profile, no vendored parity fixtures, no daemon
registration. Implementation would require a new profile id, new
parity fixtures, and a fresh review through the active milestone sequence
in [`../../ROADMAP.md`](../../ROADMAP.md).

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Instruction-tuned encoder (DeBERTa-v3-large instruct-tuned) |
| Tokenization | One sequence per question: label markers, task prompt, state document, up to context-length tokens |
| Forward pattern | Single forward pass per question regardless of candidate count; pool each label span, sigmoid |
| Readout | Sigmoid per label-marker span; `Choice` maps to argmax after renormalization, `Score` to ordered-level softmax, `Noul` to the `true`-label sigmoid |
| Continuation state | None — each question is an independent forward pass |
| Text generation | None |
| Calibration | Models are zero-shot; probabilities are uncalibrated and require a fitted temperature |
| Semantic none | Reserved option key `__none__` becomes an explicit label marker |

## Why this family is a candidate

- All candidate probabilities come from one forward pass, so cost grows
  with the number of questions, not candidates. Encoder-NLI does not
  share this property; see [`encoder-nli.md`](encoder-nli.md).
- No per-candidate batch dimension, which keeps memory low on Apple
  Silicon CPU.
- The label-marker mechanism maps naturally to `Choice`, `Score`, and
  `Noul` shapes without bespoke heads.

## What blocks implementation

- Label-marker count is bounded by the encoder's pretraining context.
  A `Choice` question with more options than fit in the context window
  is rejected with a wire-level `TOO_MANY_OPTIONS` error. The encoder
  families in this category typically top out around 250 options per
  question; the boundary is encoder-specific and is owned by the
  profile contract in
  [`../../crates/openkind-engine/AGENTS.md`](../../crates/openkind-engine/AGENTS.md).
- `Noul` questions with empty criteria are the family's weakest point:
  a single-label sigmoid can be confidently wrong. A profile contract
  that rejects empty `Noul` criteria would have to be designed and
  reviewed before the family could match the encoder-state-first Noul
  semantics.
- The family is zero-shot; the M2 useful-decision gate in
  [`../../ROADMAP.md`](../../ROADMAP.md) requires a retained useful
  operating point with calibrated probabilities on reviewed inputs.

## Open questions

- Does the label-marker context limit conflict with the M0 supported
  workload's option-count budget?
- Can the empty-`Noul` weakness be addressed by always requiring
  `criteria: {true, false}` without breaking Jev wire compatibility
  (see [`../../JEV_COMPATIBILITY.md`](../../JEV_COMPATIBILITY.md))?
- What calibration temperature fits the family on the M2 retained
  reference set?

## What this page does not say

No timings, no accuracy numbers, no calibration error. The cited
comparison values belong to ollaya's family page on `gliclass`, not to
`openkind`. `openkind` has no measurements to report.
