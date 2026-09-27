# Family: encoder-state-first

> Encoder backbone, **state-first segmented tokenization**, score-summary
> readout over candidate suffixes. The architectural pattern implemented
> by the provisional `openkind` profile.

## Status in openkind

**Implemented.** This is the only family with a native Rust decision-engine
loader. The provisional profile is `a047d6802c3f06f085b8` over
`Qwen/Qwen3.5-4B-Base`. A daemon registers a loaded instance under a configured
model alias. The pinned artifact revisions and a working Rust loading example
are in the [model registry](./README.md#runnable-model-profiles) and its
[Rust loading guide](./README.md#load-the-profile-from-rust). Status,
milestones, and remaining work are owned
by [`../ROADMAP.md`](../ROADMAP.md); landed contracts are owned by
[`../ARCHITECTURE.md`](../ARCHITECTURE.md); MLX backend specifics are
owned by [`../MLX.md`](../MLX.md); benchmark methodology is owned by
[`../BENCHMARKS.md`](../BENCHMARKS.md).

Do not reproduce any quantitative claim from those documents on this page.
If a measurement is missing from them, it is missing from `openkind`.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Encoder + linear-attention hybrid (Qwen 3.5 4B: **24 DeltaNet linear-attention blocks + 8 grouped-query full-attention blocks** + final RMSNorm) |
| Tokenization | State-first segmentation with state-document prefill, question and candidate suffixes branched from the immutable root |
| Forward pattern | Single encoder forward pass per candidate suffix, reusing the shared root prefix |
| Readout | Score-summary rejection head — projection, rejection, calibration, stable softmax |
| Continuation state | Backend-neutral branchable state carrying attention KV, DeltaNet recurrent state, convolution state, and absolute position |
| Text generation | None. Host Rust code serializes structured JSON responses directly from the candidate distribution |
| Calibration | Per-profile temperature scaling fitted offline; threshold policy `0.98` |
| Semantic none | Reserved option key `__none__` in the criteria map; fail-closed on any other declared probability space |

The contract surface is owned by
[`../../crates/openkind-engine/AGENTS.md`](../../crates/openkind-engine/AGENTS.md).
Family-specific quirks live in `crates/openkind-backends/`; see the
backend crate's `AGENTS.md` for invariants.

## Why this family

The pattern was selected by the exploratory `2ij.2.0` screen and survives
because every other candidate family has at least one of these open
questions:

- Encoder-NLI scales linearly with option count rather than producing all
  candidate probabilities in one pass; see [`encoder-nli.md`](encoder-nli.md).
- Encoder-instruct-label cannot exceed its label-marker context window
  for high-cardinality `Choice` questions; see
  [`encoder-instruct-label.md`](encoder-instruct-label.md).
- Decoder-logit-letter and decoder-logit-llm collapse latency by an
  order of magnitude on Apple Silicon and lack a measured MLX parity
  pass; see [`decoder-logit-letter.md`](decoder-logit-letter.md) and
  [`decoder-logit-llm.md`](decoder-logit-llm.md).

Selection rationale, prior-art comparisons, and the rejected alternatives
are catalogued in [`../RESEARCH.md`](../RESEARCH.md) and the
exploratory chapter of [`../whitepaper/WHITEPAPER.md`](../whitepaper/WHITEPAPER.md).
This page does not republish that narrative.

## Open questions blocking promotion

Inherited from the active milestone sequence in
[`../ROADMAP.md`](../ROADMAP.md) and not specific to this family:

- A pinned reviewed workload (M0)
- A repaired source-to-input evidence contract (M1)
- A retained useful-decision reference on common inputs (M2)
- A matched MLX execution comparison, then one cheaper challenger (M3)
- A scoped preview with fresh held-out confirmation and release manifest (M4)

Family-specific open questions that the M2/M3 work would have to answer
before any release-quality claim:

- Whether a smaller backbone can hold the same reviewed operating point
- Whether pooled-root or learned state-query variants reduce complete
  request cost without breaking the policy screen
- Whether the rejection head is portable across non-Qwen encoder
  backbones without re-fitting

## What this page does not say

This page does not state timings, ECE, NLL, memory footprints, or any
other quantitative claim. Those belong to
[`../BENCHMARKS.md`](../BENCHMARKS.md) and the verification reports
linked from [`../ROADMAP.md`](../ROADMAP.md). The pinned profile id,
backbone, bundle sha256, calibration temperature, policy threshold, and
probability tolerance are documented in the parent
[`../../README.md`](../../README.md) and
[`../../crates/openkind-engine/AGENTS.md`](../../crates/openkind-engine/AGENTS.md).
This page deliberately does not restate them so that a single source of
truth governs those values.
