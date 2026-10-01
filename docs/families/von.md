# Family: von

> ModernBERT encoder, one packed sequence per question, every option scored
> jointly at its own `[MASK]` marker in a single bidirectional pass.

## Status in openkind

**Rust-loadable (prototype profile).** The pinned profile
`69219703407bd39cca0c` loads `wfzyx/von` at
`d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd` (von-1.1, Apache-2.0) through
the shared candle ModernBERT body plus a port of the reference
`OptionMarkerScorer` head. The checkpoint is the author's
`option_marker.pt` — a PyTorch pickle carrying the full encoder and scorer
state dict — read natively by candle's pickle reader, so openkind serves
the author's artifact byte-for-byte with no conversion or re-hosting. The
same `.pt` bytes are pinned by the ollaya registry's `von:1.1` weights
layer (digest-verified identical).

It is catalog-installable offline-first: `openkind pull von:69219703407bd39cca0c`
(or the ollaya-compatible alias `von:1.1`) downloads the pinned artifacts,
verifies every SHA-256, and installs them for `--installed-models` (see
[`../MODELS.md`](../MODELS.md)). The engine registers in `openkindd` via
`--von-aliases` / `--von-model-root` and is benchmarked through
`openkind-bench score --engine von`.

This page supersedes the earlier "blocked, external-reference-only"
disposition recorded during the 2026-09-26 survey: that assessment targeted
a `von` reference contract `openkind` did not own. The 2026-09 release of
the open-weight von-sdk changed the premise — the reference implementation,
weights, and calibration are public and Apache-2.0, and the parity fixtures
below are generated from that reference.

Profile semantics:

- Probability space: `ConditionalOnOfferedOptions`. One logit per offered
  option marker; the reported distribution is a temperature-scaled softmax
  over those options only. An offered `__none__` key is scored as an
  ordinary candidate.
- Rendering: `[CLS] <question> <state> [SEP] ([MASK] + option)… [SEP]`,
  with `[MASK]`/`[SEP]` literals in user text neutralised by a zero-width
  joiner (a forged marker would otherwise add a phantom option). Long
  states middle-truncate at 60% head / 40% tail joined by `" ... "`,
  reserving room for the question and option spans.
- Calibration: the shipped input-conditioned temperature map
  (`marker_calibration.json`) — a bounded linear function of the option
  distribution's normalized entropy, `log10(state_tokens)/4`, and the
  option count over 8, clamped to `[0.3, 12.0]`. Temperature is monotonic,
  so it never changes an answer, only the reported confidence.
- Noul zero-shot debias: without explicit criteria the reference's second
  pass over an empty state cancels the intrinsic polarity prior
  (`0.7 × (logit_true − logit_false)`; the pinned file ships no fitted
  prior). openkind reports the raw calibrated posterior — the reference
  SDK's later band-commit policy is a serving layer, not a model semantic.
- Declared determinism divergence: `Choice` renders options in the sorted
  wire-label order; the reference renders caller insertion order. The
  default joint attention makes logits position-sensitive, so the two
  orders can differ numerically. The golden fixtures are generated in the
  sorted order openkind computes.
- Continuation state: none; every question is one full forward pass.
- Text generation: none.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | ModernBERT-large encoder (28 layers, hidden 1024, 16 heads; global attention every 3rd layer, 128-token sliding window otherwise) |
| Scorer head | `LayerNorm → Linear(1024, 512) → GELU → LayerNorm → Linear(512, 1)` over each gathered marker hidden state |
| Tokenization | One packed sequence per question: `[CLS] question state [SEP] [MASK] option… [SEP]`; all options scored jointly in one pass |
| Forward pattern | One bidirectional encoder pass per question regardless of option count |
| Readout | One scalar logit per `[MASK]` marker, in marker order |
| Work limits | 8192-token window; state middle-truncated to fit; at most 255 options |
| Interruption | Caller cancellation and the queue-inclusive deadline checked per question |
| Continuation state | None — every question is an independent full forward |
| Text generation | None |
| Calibration | Pinned input-conditioned temperature map (see above) |
| Semantic none | None of its own; an offered `__none__` key is scored as an ordinary option |

## Parity

Golden fixtures (`tests/fixtures/von_69219703407bd39cca0c/golden.json`,
generated from the von-sdk 1.1 reference semantics) replay through the
engine with max probability drift inside the workspace 0.005 tolerance,
including exact input-token usage counts — which pins the renderer bytes,
the truncation path, and the noul debias pass end to end
(`crates/openkind-backends/tests/von_parity.rs`, env-gated by
`OPENKIND_VON_MODEL_ROOT`).

## Benchmark record

See [`../MODELS.md`](../MODELS.md) for the measured shape777 row and
[`../benchmarks/`](../benchmarks/) for the recorded summary. As everywhere
else, throughput is not quality evidence.

## What this page does not say

No accuracy, calibration-error, or leaderboard claims about von-1.1; the
author's JevBench table is in-sample and belongs to them. The parity record
is distribution-level agreement with the reference, not decision quality.
