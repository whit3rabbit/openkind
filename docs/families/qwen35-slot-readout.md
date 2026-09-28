# Family: qwen35-slot-readout (surveyed)

> Full fine-tunes and adapter merges of the Qwen3.5-4B backbone that read
> answers from trained hidden-state positions ("slot readout") instead of
> next-token option letters, returning every question's distribution from
> structured readouts. Surveyed from the JevBench top-25 (2026-09-27); no
> Rust loader exists yet.

## Status in openkind

**Surveyed only.** Three top-25 systems use this pattern; all checkpoints
are openly accessible. The backbone forward is already executed by the
native CPU path (see
[`qwen35/backbone`](../../crates/openkind-backends/src/qwen35/backbone/));
the missing pieces are the per-system readout heads and prompt contracts.

| JevBench rank | System | Checkpoint | Readout |
|---|---|---|---|
| 1 | Imajev-4B | `mohit67890/imajev-4b` (PEFT LoRA r64/α128 over `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) | Trained 256-code decision readout (`decision_readout.safetensors`, 255 option codes + unknown), single pinned temperature 1.3051569717552742, 4 cyclic option rotations averaged |
| 3 | decider-4b v2.1 | `Mapika/decider-4b` (full bf16 weights over `Qwen/Qwen3.5-4B-Base`) | Slot readout over one forward per request; per-type temperatures (`choice` 1.110, `noul` 1.560, `score` 1.287) in `decider_config.json` |
| 12 | metask-jev-4b-policy-mix | `wayfind/metask-jev-4b-policy-mix` (`Qwen3_5ForConditionalGeneration` full weights) | Not yet inspected in detail |
| 21 | decider-35b-a3b | `Mapika/decider-35b-a3b` (`Qwen3_5MoeForCausalLM`, 15 bf16 shards ≈ 70 GB) | decider readout at MoE scale; needs sparse expert execution in the native path before any profile |
| 7 | Hopper | `HopitAI/hopper` (adapter-only `adapter_model.safetensors`) | Adapter-only release; base and head contract not yet identified |
| 9 | reflex 4B | `kshetrajna12/reflex-qwen3.5-4b-lora` (PEFT adapter) | Adapter-only release; base and head contract not yet identified |
| 18 | Malkuth-4B | `dhtocks/malkuth-4b` (PEFT adapter + `head.pt`) | Adapter plus torch-pickle head; conversion needed before pinning |

Imajev-4B additionally publishes an MLX export (`mlx/adapters.safetensors`)
and `SHA256SUMS` for every artifact, and its release spec documents an
explicit unknown/abstain gate that the owner overrode for the shipped
checkpoint — that disclosure must be carried into any profile page.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | The same frozen Qwen3.5 hybrid geometry the native path executes (32 layers, hidden 2,560); Imajev targets the multimodal `ForConditionalGeneration` variant, so text-only execution reads the `language_model.*` weights |
| Readout | Trained projection heads over final or slot hidden states, not letter logits; wire mapping differs per system |
| Calibration | Per-type or single scalar temperatures shipped with the checkpoints |
| Continuation state | None expected (single-pass readouts), pending per-system confirmation |

## What remains open

- Each readout head needs its own profile, digest pins, and offline parity
  fixture; the shared backbone work in
  [`decoder-logit-qwen35`](./decoder-logit-qwen35.md) (generic verified
  shard forward) is reusable but the readouts are materially different
  families of wire mapping.
- Adapter-only releases (Hopper, reflex) lack identified base revisions and
  heads; do not guess them.
- Imajev's rotation averaging and unknown-mass mapping need a wire decision
  before implementation (how `unknown` mass maps onto the reserved
  `__none__` contract).

## What this page does not say

No accuracy, calibration, or leaderboard claims about any listed system;
those belong to the checkpoint authors and to
[`../BENCHMARKS.md`](../BENCHMARKS.md) once profiles exist.
