# Family: gemma4-decision (surveyed)

> Gemma 4 unified decoder backbones fine-tuned or adapter-tuned for typed
> decisions. Surveyed from the JevBench top-25 (2026-09-27); no Rust loader
> exists yet.

## Status in openkind

**Surveyed only.** Four JevBench systems in the top 25 use Gemma 4
backbones; two of their checkpoints are openly accessible, two are gated on
the Hub (HTTP 401 even for metadata with our credentials as of 2026-09-27).

| JevBench rank | System | Checkpoint | Accessibility |
|---|---|---|---|
| 6 | Cygnet (blockbrain) | frozen Gemma-4-12B-it | Gated — `blockbrain-ai/cygnet` returns 401/404 unauthenticated; operator must accept the gate |
| 8 | Winnow-12B Q8 (Eldan Ring) | `EldanRing/Winnow-12B` | Accessible — GGUF only (BF16, Q8_0, plus an `mmproj` vision projector), architecture `Gemma4UnifiedForConditionalGeneration` |
| 11 | Jev-Omni | `akhilaaa3/Jev-Omni` | Accessible — full `model.safetensors` plus a torch-pickle `head.pt` decision head |
| 16 | system-one-open | `mithalouni/system-one-open` | Gated (401/404, same as Cygnet) |

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Gemma 4 unified decoder (sliding-window + full attention layers, multimodal-capable) — no implementation exists in `candle` 0.8 or this repository |
| Readout | Jev-Omni: trained decision head shipped as `head.pt` (torch pickle, mirroring the kev conversion precedent); Winnow-12B: GGUF-only weights, readout protocol not yet inspected |
| Quantization | Winnow-12B distributes Q8_0/BF16 GGUF; no safetensors release |

## What remains open

- A Gemma 4 backbone implementation (candle) is the prerequisite for every
  profile in this family; nothing in the workspace executes this
  architecture today.
- `head.pt` requires an operator conversion to safetensors before a loader
  can pin it (the [`kev`](./kev.md) head conversion is the precedent).
- The gated checkpoints need operator action on the Hub before pins,
  digests, or fixtures can exist.
- No M2 reviewed-decision gate applies: nothing loads.

## What this page does not say

No accuracy or leaderboard claims about any listed system; those belong to
the checkpoint authors and to [`../BENCHMARKS.md`](../BENCHMARKS.md) once a
profile exists.
