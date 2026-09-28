# Family: reranker-logit (surveyed)

> Dense Qwen3 decoders used as typed-decision scorers through two-token
> (yes/no) or option-letter next-token logits. Surveyed from the JevBench
> top-25 (2026-09-27); no Rust loader exists yet.

## Status in openkind

**Surveyed only.** All three checkpoints are openly accessible. The Qwen3
dense backbone (standard causal attention, QK-norm) is already implemented
at 0.6B scale in the [`kev`](./kev.md) family; these systems need that
architecture at 4B plus their readout mappings.

| JevBench rank | System | Checkpoint | Readout |
|---|---|---|---|
| 20 | Qwen3-Reranker-4B | `Qwen/Qwen3-Reranker-4B` (`Qwen3ForCausalLM`, 2 bf16 shards) | Relevance logit over yes/no token logits on an instruction-formatted query-document pair |
| 22 | Raw Qwen3 4B Instruct 2507 direct logits | `Qwen/Qwen3-4B-Instruct-2507` (3 shards) | Un-tuned option-letter next-token logits — the control baseline of the board |
| 24 | ZeroEntropy zerank-2 | `zeroentropy/zerank-2-reranker` (`Qwen3ForCausalLM`, sentence-transformers packaging) | Relevance logit; packaging and head contract need inspection before pinning |

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Qwen3 dense decoder, 4B, bf16 shards — same architecture family as `Qwen/Qwen3-0.6B-Base` already executed by the kev family |
| Readout | Restricted next-token logits (yes/no or option letters) with temperature calibration; no trained decision head |
| Continuation state | KV only, per-pair or per-question; nothing retained across requests |

## What remains open

- Scaling the kev-era Qwen3 implementation to 4B (weights, digest pins, and
  a fresh parity fixture; the 0.6B fixtures do not transfer).
- Whether the reranker yes/no readout maps onto Noul questions alone or
  supports Choice via pairwise scoring; the wire decision must precede any
  loader.
- zerank-2's sentence-transformers wrapper must be unwrapped to the raw
  decoder contract before its head can be pinned.

## What this page does not say

No accuracy, relevance-quality, or leaderboard claims about any listed
system; those belong to the checkpoint authors and to
[`../BENCHMARKS.md`](../BENCHMARKS.md) once profiles exist.
