# Clef

> Cloudflare's Clef decision models — a post-trained Qwen3.5 hybrid backbone
> with a joint schema head that reads a state and a schema of typed questions
> and scores every allowed option of every question in a single forward pass.

## Status in openkind

Surveyed 2026-10-01, implemented the same week. The reference serving layer is
the model card's `joint_schema_model.py` (Apache-2.0); openkind reproduces its
record encoding, joint-head math, and answer mapping natively, with no
autoregressive text generation.

| Profile | Board | Checkpoint | Execution | Profile ID |
|---|---|---|---|---|
| `clef-flash` | research | `Cloudflare/clef-flash` BF16 | candle CPU, BF16 weights with FP32 compute | `dfe12a21a5c9dd5b2fb1` |
| `clef-flash-gguf` | research | `bartowski/Cloudflare_clef-flash-GGUF` Q4_K_M + official joint head | candle CPU quantized runner | `c330d9ee7e9cc658ad45` |
| `clef-27b-gguf` | research | `bartowski/Cloudflare_clef-GGUF` Q4_K_M + official joint head | candle CPU quantized runner | `48cb5634b4a258de5a6b` |
| `clef-flash-mlx-4bit` | surveyed (implementation in tree, parity open) | `mlx-community/clef-flash-4bit` | MLX/Metal quantized (feature `mlx`) | `0fb395e836f1c22a3fe3` |

Serving flags: `--clef-model-roots flash=<path>;flash-gguf=<path>;27b=<path>`
with `--clef-aliases` of the same shape, or `openkind pull clef:flash`,
`clef:flash-gguf`, and `clef:27b` through the catalog. Bench engine names:
`clef-flash`, `clef-flash-gguf`, `clef-27b-gguf`.

## Architecture

The backbone is the frozen `Qwen3_5ForConditionalGeneration` hybrid text
stack — gated DeltaNet linear-attention layers with causal depthwise
convolution interleaved with grouped-query full-attention layers (interval 4),
partial NeoX rotary over the first quarter of the full-attention head width,
SiLU-gated MLPs. Clef-Flash: hidden 4096, 32 layers, 16 key / 32 value × 128
DeltaNet heads, 16 query / 4 KV × 256 attention heads. Clef (27B): hidden
5120, 64 layers, 16/48 × 128 DeltaNet, 24/4 × 256 attention.

The joint schema head (width 1024, 2 evidence-routing cross-attention layers,
4 pre-norm decoder layers, heads 16, feedforward 4096) reads the final-norm
hidden states:

1. Mean-pool hidden states over each question's instruction span and each
   option's payload span; global vector from the last real token.
2. Option queries = context + lexical (mean output-embedding rows of the
   option's tokens) + parent-question projections; evidence routing over the
   state memory; per-question option summaries; decoder layers across
   questions.
3. Per-option logit = lexical prior (`exp(min(prior_logit_scale, ln 100))`
   cosine against the question-plus-global anchor) plus a sigmoid-gated
   residual-MLP joint term.

The readout is one softmax over the offered options per question;
probability space `ConditionalOnOfferedOptions`; no sampling, no retained
state between requests, no semantic-none mass.

## Control contract (declared)

| Payload | Value |
|---|---|
| Prompt | `<|im_start|>system` + joint-decision system prompt + user turn: `STATE:` JSON, `SCHEMA FIELDS:` block (one `FIELD n / ID / TYPE / INSTRUCTION / ALLOWED OPTIONS` per question), `JOINT SCHEMA DECISIONS:` trigger |
| JSON rendering | Python `json.dumps(..., ensure_ascii=False, separators=(",", ":"), sort_keys=True)` |
| Question order | sorted question id (the Jev wire map is unordered; the Python reference preserves JSON insertion order — a declared renderer difference) |
| Noul options | rendered `(true, false)` with the caller's wire criteria; scored, then reversed back to the wire's `[false, true]` |
| Choice options | sorted lexicographically (wire order preserved) |
| Score options | positional levels |
| Sequence bound | 16 384 encoded tokens (reference `max_length`) |
| Calibration | none — softmax at temperature 1.0 |
| Generation | none |
| Vision | none exposed (text decision surface only) |

## GGUF execution notes

The bartowski GGUF checkpoints (llama.cpp `qwen3.5` layout) store the
linear-attention value/gate streams with value heads grouped by key-head
group and interleaved group-first, the DeltaNet decay as `A = -exp(A_log)`
instead of the log, and every RMSNorm weight pre-folded as `1 + w`. The
loader undoes all three; the forward kernels are the parity-verified shared
Qwen3.5 kernels.

## JevBench systems covered

Clef is not a JevBench system. Dataset evidence lives under
`benchmarks/2026-10-02-clef/` and `docs/BENCHMARKS.md`.

## What remains open

- `clef-flash-mlx-4bit`: the MLX 4-bit execution path (in tree, feature
  `mlx`) runs end-to-end but its DeltaNet state on quantized inputs diverges
  from the CPU oracle — the joint-head parity fixtures do not pass yet, so
  the profile is not loadable or catalog-installable. Promote it when the
  parity fixtures pass.
- CUDA and ONNX execution: not implemented; the shared
  [`FamilyExecution`](../../crates/openkind-backends/src/device.rs) seam
  accepts them when evidence work lands.
- The 27B BF16 checkpoint (54.7 GB) is host-infeasible locally; the GGUF
  profile covers the 27B model.

## What this page does not say

Anything about task quality or release promotion: no M2 reviewed-decision
gate has run for this family. The dataset evaluations recorded in
`docs/BENCHMARKS.md` are research evidence, not qualification.
