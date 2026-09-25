# OpenKind Phase 3B — Qwen3.5 Rust backbone handoff

Profile: `a047d6802c3f06f085b8`
Bundle SHA-256: `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`
Base: `Qwen/Qwen3.5-4B-Base`
Revision: `1001bb4d826a52d1f399e183466143f4da7b741b`
Reference repo: `cowWhySo/OpenKind-Qwen3.5-4B-StateFirst`

Training/model selection/model modification in Phase 3B: **none**

## Rust implementation order

1. Load the exact model revision and exported tokenizer.
2. Reproduce `TOKEN_FIXTURES.json` exactly.
3. Reproduce Qwen3.5 full-sequence FP32 inference.
4. Use `LAYER_TRACE.json` + `QWEN35_BACKBONE_GOLDEN.safetensors` to identify the first divergent stage.
5. Reproduce every golden candidate's final 2,560-D hidden vector closely enough that the selected decision distribution satisfies the existing same-profile gate.
6. Reproduce `PROBABILITY_REFERENCE.json` within probability tolerance `0.005` with zero argmax/policy changes.
7. Reproduce `CONTINUATION_TRACE.json`.
8. Then implement the Phase 3A `BranchableState` contract and batched-Q/K execution.

## Important Qwen3.5 requirements

The text backbone is hybrid. The exact model config in `QWEN35_ARCHITECTURE.json` is authoritative.
Do not implement it as a conventional KV-only transformer.

The selected 4B config has 32 decoder blocks with the declared `layer_types` exported by the notebook.
Continuation state must preserve full-attention KV plus linear-attention recurrent and convolution state.

## Do not silently change

- renderer or segment tokenization;
- model revision;
- normalization/head/rejection parameters;
- dtype/precision while claiming strict reference parity;
- option ordering;
- candidate descriptions;
- profile identity.

CUDA timing from this notebook is not a Mac/Metal acceptance threshold.
