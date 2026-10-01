# Family: encoder-instruct-label

> Instruction-following label-marker encoder: all candidate label markers in
> one sequence, one forward pass per question regardless of candidate count.

## Status in openkind

**Implemented (unblocked 2026-09-26).** The family is served by the pinned
profile `knowledgator/gliclass-modern-base-v3.0` at revision
`ac369222ca4375ca66ebaf7fb5220f223514c035` (Apache-2.0), profile ID
`9fd68313a5606eca42f2`: a GLiClass-style uni-encoder — ModernBERT-base
backbone (22 layers, hidden 768, hybrid global/sliding attention, RoPE-only
positions) whose `<<LABEL>>`-marked class-token positions are pooled,
projected, and dot-scored against the pooled text representation. The
backbone is hand-implemented for the pinned candle 0.8.0 line in
[`crates/openkind-backends/src/families/encoder_instruct_label/arch.rs`](../../crates/openkind-backends/src/families/encoder_instruct_label/arch.rs)
(candle ships no ModernBERT); the Rust forward was validated against a
PyTorch reference of the same checkpoint to `<= 4.3e-6` maximum answer delta
over the golden fixture, and digest-checked parity fixtures live under
`crates/openkind-backends/tests/fixtures/encoder_instruct_label_9fd68313a5606eca42f2/`.

It is catalog-installable offline-first: `openkind pull encoder-instruct-label:9fd68313a5606eca42f2` downloads the pinned artifacts, verifies every SHA-256, and installs them for `--installed-models` (see [`../MODELS.md`](../MODELS.md)).

**MLX backend (2026-09-29).** The same pinned checkpoint also runs on the
MLX/Metal backend (feature `mlx`, macOS arm64): `openkindd
--encoder-instruct-label-backend mlx-fp32` and `openkind-bench --engine
encoder-instruct-label-mlx-fp32`. No MLX conversion of the checkpoint exists
on the Hub (surveyed 2026-09-29), so the backend reads the identical
digest-verified FP32 shard directly; the arithmetic identity is
`mlx-gpu-fp32-gliclass-modern-base`. The MLX body is the parity-proven
shared ModernBERT implementation
([`families/mlx_modernbert.rs`](../../crates/openkind-backends/src/families/mlx_modernbert.rs),
extracted from the laya backend) behind the GLiClass projector pair and
dot-product scorer; the candle CPU path remains the correctness oracle.

How the two former blockers were resolved:

1. **Backbone availability.** The original blocker named
   DeBERTa-v3 / ModernBERT / llm2vec-style backbones that candle 0.8.0
   cannot load. The resolution is the same one `qwen3guard` used for Qwen3:
   hand-implement the pinned checkpoint's architecture. GLiClass v3 moved to
   ModernBERT encoders (`gliclass-modern-*`, `gliclass-edge-v3.0`), whose
   semantics are compact: pre-norm residual blocks, fused bias-free `Wqkv`,
   NeoX rotate-half RoPE with per-layer-type thetas (local 10000, global
   160000), global attention every 3rd layer starting at layer 0, a
   symmetric sliding band of half-width `local_attention / 2` elsewhere, and
   gated GELU MLPs.
2. **Checkpoint selection on measured state-support behavior.** The 2026-09-26
   prototype's MLM fallback was rejected because it tracked claim
   plausibility rather than state support. Three GLiClass v3 checkpoints were
   probed on a 36-case support-ticket battery (shape777-style prose, bare
   criterion markers, 0.5 threshold) plus the 7 stated-fact single-fact
   frames the earlier prototype had failed:

   | Checkpoint | Ticket battery | Single-fact frames | Verdict |
   |---|---|---|---|
   | `gliclass-edge-v3.0` (32M) | 12/36 | 2/4 | rejected: floods on multi-topic prose |
   | `gliclass-base-v3.0` (DeBERTa-v3) | 16/36 | 2/4 | rejected: worse than edge on tickets |
   | `gliclass-modern-base-v3.0` (149M) | **32/36** | **7/7** | selected |

   The modern-base misses are all borderline true-cases (0.35–0.45 against a
   0.5 threshold), not inversions. The rejected checkpoints were deleted
   from the model cache after the probe; only the selected checkpoint's
   digest is pinned in the family module.

The instruction-verbatim `Noul` marker (the shape777 question form "Answer
yes if … Otherwise answer no.") scores 30/36 on the same battery; the
calibration `Noul` shapes all pass.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | GLiClass uni-encoder over ModernBERT-base (hand-implemented in candle) |
| Tokenization | One sequence per question: `[CLS]`, then `<<LABEL>>` immediately before every candidate criterion, then `<<SEP>>`, then the state document; template from the digest-verified fast `tokenizer.json` |
| Forward pattern | Single forward pass per question regardless of candidate count (up to the frozen 25-marker head envelope) |
| Readout | Dot product of the projected `[CLS]` representation with each projected class-marker representation. `Choice`: temperature-calibrated sigmoids renormalized over the offered set. `Score`: temperature softmax over level logits. `Noul`: the support sigmoid of one proposition marker |
| Continuation state | None — each question is an independent forward pass |
| Text generation | None |
| Calibration | Fitted temperature `0.44530092168688412` by mean NLL over the 15-case stated-fact workload with the exact wire readout (NLL 0.151 vs 0.194 at T=1); see [`../BENCHMARKS.md`](../BENCHMARKS.md) |
| Semantic none | The reserved `__none__` key is an ordinary offered marker; no mass is invented |

## Readout determinism contract

Marker scores shift with marker order (the encoder attends across the whole
sequence), so the profile owns a fixed order: `Choice` labels sort
lexicographically (the shared wire unpacking order), `Score` levels stay in
declared order. Probabilities are defined under that order and the golden
fixtures pin it.

## Noul contract

The proposition marker is the caller's `true` criterion when explicit Noul
criteria are supplied, and the question instruction verbatim otherwise. The
`false` criterion is never scored: the readout is the support sigmoid of the
proposition itself. Scoring the wire-default meta-criteria instead would
score how the sentence "The stated proposition is true…" is itself supported
— the exact self-support trap the earlier prototype documented — so the
family rejects nothing, it simply scores the caller's words, not the meta
frame.

## Limits

- `MAX_SEQUENCE_TOKENS = 1024` (reference pipeline budget); over-long
  sequences fail closed, truncation is forbidden.
- `MAX_CANDIDATES = 25` (the checkpoint's trained `max_num_classes`); more
  markers leave the head's training envelope and are rejected at the wire.
- The profile is zero-shot; the M2 useful-decision gate
  has not run for it. The 32/36 ticket
  battery above is checkpoint-selection evidence, not a recorded
  classification evaluation: no dataset revision, seed, or retained
  operating point is claimed here.

## Benchmark record

`openkind-bench score` over the standard shape777 workload (777 rows):
4.73 decisions/s, 1.27 GB peak RSS, 1.23 s model load (candle CPU,
2026-09-29 re-run) and 75.42 decisions/s, 1.02 GB peak RSS, 1.46 s model
load (MLX FP32); recorded in [`../BENCHMARKS.md`](../BENCHMARKS.md) and
[`../benchmarks/2026-09-29-mlx-counterparts/`](../benchmarks/2026-09-29-mlx-counterparts/).
One forward pass carries all 21 criteria of a grouped request. The backbone
still has roughly 8× the per-token compute of the encoder-nli DistilBERT,
hence the lower CPU throughput. Request-path timing only; no model-quality
claim.

MLX backend parity (2026-09-29): the golden fixtures replay through the MLX
engine (env-gated test in
`crates/openkind-backends/tests/encoder_instruct_label_parity.rs`, module
`mlx_replay`, enabled by `--features mlx` plus
`OPENKIND_ENCODER_INSTRUCT_LABEL_MODEL_ROOT`): 15 answers, max probability
drift 4.487e-6, zero selection flips — inside the workspace MLX gate of
0.005.

## What this page does not say

No accuracy numbers on any public dataset, no calibration-error metric, and
no claim about the DeBERTa-v3 GLiClass line beyond the recorded rejection
probe. The comparison rows in the vendor's model card belong to their
benchmarks, not to `openkind`.
