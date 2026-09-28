# Family: diffusion-decision (surveyed)

> Diffusion-language-model decision scorers. Surveyed from the JevBench
> top-25 (2026-09-27); no Rust loader exists and the board's ranked
> checkpoint is not published.

## Status in openkind

**Surveyed only — blocked for the ranked system.**

| JevBench rank | System | Checkpoint | Accessibility |
|---|---|---|---|
| 10 | djev (Maisa) | fine-tuned `diffusiongemma-26B-A4B` | No public weights — the benchmark notes mark djev "open runtime, API measured": the serving runtime is open, but the tuned checkpoint is not published |
| 29 | OpenJev, DiffusionGemma 26B NVFP4 | community quantizations | Out of top-25 scope |

The base model `google/diffusiongemma-26B-A4B-it` is openly accessible, so
the architecture could be qualified independently of djev's weights.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Diffusion decoder (masked / iterative parallel refinement) — a fundamentally different forward graph from the causal decoders this workspace executes |
| Readout | Not inspected (no public decision head or protocol for djev) |
| Inference | Multiple refinement steps per decision; conflicts with the single-pass execution model until a profile defines its step budget |

## What remains open

- djev stays external-reference-only until the tuned weights are published
  (the [`von`](./von.md) precedent).
- A base-model qualification would need a diffusion forward implementation
  in this workspace, a decision on non-autoregressive iterative refinement
  versus the no-generation invariant, and its own evidence gates.

## What this page does not say

No accuracy or leaderboard claims about djev; those belong to the operator
and to [`../BENCHMARKS.md`](../BENCHMARKS.md) once a profile exists.
