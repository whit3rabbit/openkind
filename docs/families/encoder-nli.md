# Family: encoder-nli

> Encoder backbone, one premise–hypothesis forward pass per candidate,
> softmax over entailment probabilities.

## Status in openkind

**Surveyed.** No profile, no vendored parity fixtures, no daemon
registration. Implementation would require a new profile id, new
parity fixtures, and a fresh review through the active milestone sequence
in [`../../ROADMAP.md`](../../ROADMAP.md).

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Encoder (DeBERTa, ModernBERT, or comparable) fine-tuned for natural-language inference |
| Tokenization | One sequence per candidate: premise (state) followed by hypothesis (candidate-statement) |
| Forward pattern | N forward passes per question, one per candidate; softmax over per-candidate entailment probability |
| Readout | Softmax over candidate entailment probabilities; `Noul` maps to probability of the `true` hypothesis being entailed |
| Continuation state | None — every candidate is an independent forward pass over the full state |
| Text generation | None |
| Calibration | Models are zero-shot entailment, not decision-trained; probabilities are typically uncalibrated and require a fitted temperature per label space |
| Semantic none | Reserved option key `__none__` becomes an explicit "the state entails nothing" hypothesis |

## Why this family is a candidate

- Off-the-shelf checkpoints are widely available and licensed permissively
  (MIT, Apache-2.0).
- No custom training data is required for a first useful-decision signal;
  the same zero-shot NLI model can answer `Choice`, `Score`, and `Noul`
  questions by rewriting candidates as hypotheses.
- Latency per question is comparable to encoder-state-first on small
  option sets and degrades predictably with option count.

## What blocks implementation

- Cost scales linearly with the number of candidates per question, not
  with the number of questions. A 20-option choice question performs 20
  full forward passes; the same question under encoder-state-first is
  one forward pass per candidate suffix reusing the shared root.
- Zero-shot checkpoints are over-confident; the M2 useful-decision gate
  in [`../../ROADMAP.md`](../../ROADMAP.md) requires a retained useful
  operating point with calibrated probabilities. Implementing the family
  without a fitted temperature is known to fail downstream threshold
  reviews.
- The published NLI checkpoints that perform best on typed-decisions
  carry partial non-commercial training-data licenses. The
  `-c` commercially-clean variant exists but is not the default.
- Branchable continuation state does not apply. Reusing the state-document
  prefill across candidates would require a custom NLI head, which moves
  the architecture out of the off-the-shelf NLI family.

## Open questions

- Can a custom NLI head make the family competitive with
  encoder-state-first on encoder-token-budget grounds while preserving
  the off-the-shelf license story?
- Does a fitted temperature per label space survive natural-data
  confirmation (M1/M2 in [`../../ROADMAP.md`](../../ROADMAP.md))?
- How does the family behave under the policy threshold when the state
  is shorter than the encoder's pretraining distribution? State
  truncation semantics are owned by
  [`../../crates/openkind-core/AGENTS.md`](../../crates/openkind-core/AGENTS.md).

## What this page does not say

No timings, no accuracy numbers, no calibration error. The cited
comparison values belong to ollaya's family page on `nli`, not to
`openkind`. `openkind` has no measurements to report.
