# Family: encoder-nli

> Encoder backbone, one premise–hypothesis forward pass per candidate,
> softmax over entailment probabilities.

## Status in openkind

**Rust-loadable (prototype profile).** The pinned profile
`1041a4c362338a61b820` loads `typeform/distilbert-base-uncased-mnli` at
`cfa538a0fddbbd978fefe8966c1aeff7ad409c90` (Apache-2.0) through the candle
`distilbert` implementation, FP32 on CPU. It implements `DecisionEngine`
behind the bounded family scaffold in
[`families/encoder_nli/`](../../crates/openkind-backends/src/families/encoder_nli/mod.rs),
registers in `openkindd` via `--encoder-nli-aliases` /
`--encoder-nli-model-root`, and is benchmarked through
`openkind-bench --engine encoder-nli`.

It is catalog-installable offline-first: `openkind pull encoder-nli:1041a4c362338a61b820` downloads the pinned artifacts, verifies every SHA-256, and installs them for `--installed-models` (see [`../MODELS.md`](../MODELS.md)).

Profile readout contract (documented because the off-the-shelf checkpoint
needs explicit Noul semantics):

- `Choice`/`Score`: one premise–hypothesis pass per candidate; the
  candidate-level distribution is a temperature-calibrated softmax over
  log-entailment logits.
- `Noul`: the hypothesis is the caller's `true` criterion when explicit
  criteria are supplied, and the question instruction itself otherwise; the
  answer is the entailment mass over decided (entailment vs contradiction)
  mass. Interrogative hypotheses leave most mass in the neutral class, which
  the ratio treats as undecided.
- Probability space: `ConditionalOnOfferedOptions`. A offered `__none__` key
  is scored as an ordinary candidate.

No M0/M2 reviewed-decision evidence exists for this profile; the recorded
benchmark is request-path timing only, and the calibration temperature is
fitted on the pinned synthetic calibration workload (see
[`../BENCHMARKS.md`](../BENCHMARKS.md)). The page-level concern below about
uncalibrated zero-shot probabilities remains accurate: the fitted temperature
softens rather than fixes weak entailment separations on rubric-like
hypotheses.

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

## What remains open

- Cost scales linearly with the number of candidates per question, not
  with the number of questions. A 20-option choice question performs 20
  full forward passes; the same question under encoder-state-first is
  one forward pass per candidate suffix reusing the shared root.
- Zero-shot checkpoints are over-confident; the M2 useful-decision gate
  requires a retained useful
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
  confirmation (M1/M2 qualification)?
- How does the family behave under the policy threshold when the state
  is shorter than the encoder's pretraining distribution? State
  truncation semantics are owned by
  [`../../crates/openkind-core/AGENTS.md`](../../crates/openkind-core/AGENTS.md).

## What this page does not say

No model-quality or accuracy numbers, no calibration error. The cited
comparison values belong to ollaya's family page on `nli`, not to
`openkind`. Request-path timing and peak-RSS measurements for the pinned
profile are recorded in [`../BENCHMARKS.md`](../BENCHMARKS.md).
