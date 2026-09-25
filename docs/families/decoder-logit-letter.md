# Family: decoder-logit-letter

> Decoder backbone, prompt with lettered options, next-token logits
> restricted to option-letter token ids.

## Status in openkind

**Surveyed.** No profile, no vendored parity fixtures, no daemon
registration. Implementation would require a new profile id, new
parity fixtures, and a fresh review through the active milestone sequence
in [`../../ROADMAP.md`](../../ROADMAP.md). The on-disk exploration notes
under `crates/openkind-backends/` mention a `decider` adapter as a
design-only artefact; no Rust profile is registered against it.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Decoder-only transformer fine-tuned to place answer-slot mass on option-letter tokens |
| Tokenization | One prompt per question: state, then `A. opt1 / B. opt2 / …`; no further tokens are generated |
| Forward pattern | Single forward pass per question; next-token logits are read at the answer slot and masked to letter tokens only |
| Readout | Softmax over letter tokens at the answer slot; letter → option map is the readout contract |
| Continuation state | Decoder KV cache. For Qwen-class backbones the family additionally carries DeltaNet recurrent state and convolution state (see [`encoder-state-first.md`](encoder-state-first.md) for the hybrid-state contract); pure-attention decoder variants carry KV only |
| Text generation | None — the family never samples an output token; the answer is the argmax over letter logits |
| Calibration | Per-profile temperature fitted offline; probability tolerance and policy threshold live in the profile contract |
| Semantic none | `Noul` questions always have exactly two options (`A. <true-description> / B. <false-description>`); the answer is `[false, true]` |

## Why this family is a candidate

- Decoder backbones can hold long state documents without truncation.
  The published 2B variants carry 32k context windows versus the
  512–1024 tokens typical for encoder families.
- A single letter-logit readout handles `Choice`, `Score`, and `Noul`
  shapes uniformly; the family contract is small.
- Decoder backbones are the strongest available open model class for
  longer-context decision tasks, which the M0 supported-workload
  definition in [`../../ROADMAP.md`](../../ROADMAP.md) may eventually
  require.

## What blocks implementation

- Decoder backbones are slow on Apple Silicon CPU relative to the
  encoder families. The `decider` adapter design notes record roughly
  an order-of-magnitude latency increase over encoder-state-first on
  the named M4 Max; measured numbers, if any, live in
  [`../../BENCHMARKS.md`](../../BENCHMARKS.md).
- The M3 lower-cost gate in [`../../ROADMAP.md`](../../ROADMAP.md) has
  the encoder-state-first operating point as the reference; the family
  would have to demonstrate competitive cost on the same reviewed
  workload before any release-quality claim is possible.
- The published 2B checkpoints carry weight files in the 4–8 GB range.
  In-place verification and memory ceiling rules are owned by
  [`../../crates/openkind-backends/AGENTS.md`](../../crates/openkind-backends/AGENTS.md);
  the family would have to satisfy them at fp32 widening.
- BF16 reference paths are documented as failing the frozen probability
  tolerance gate for the Qwen 3.5 family. Decoder checkpoints at this
  scale have the same risk and would need their own parity fixtures.

## Open questions

- Can the family hold the M2 reviewed-decision operating point at all?
  No measured numbers exist.
- Does the letter-logit readout produce calibrated probabilities on
  common inputs? Calibration methodology is owned by
  [`../../BENCHMARKS.md`](../../BENCHMARKS.md).
- Is there an MLX parity path for the relevant decoder backbone? The
  landed MLX contract in [`../../MLX.md`](../../MLX.md) covers only
  Qwen 3.5 4B; the family would need a separate parity study.

## What this page does not say

No timings, no accuracy numbers, no memory numbers. The cited comparison
values belong to ollaya's family page on `decider`, not to `openkind`.
`openkind` has no measurements to report.
