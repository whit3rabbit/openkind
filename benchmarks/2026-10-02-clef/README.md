# Clef family bring-up and cross-check (2026-10-02)

First records for the Cloudflare Clef family: three Rust-loadable profiles
(`clef-flash` BF16 CPU oracle, `clef-flash-gguf` Q4_K_M, `clef-27b-gguf`
Q4_K_M) with golden-fixture parity, plus dataset evaluations. Request-path
evidence only — no model-quality claim and no promotion; see
[`families/clef`](../../docs/families/clef.md).

## Profiles and pins

| Profile | Checkpoint sources | Pinned revision | Download |
|---|---|---|---|
| `clef-flash:dfe12a21a5c9dd5b2fb1` | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | `17f0b0ad64efb65d273590632833508766b2aae6` | 19.08 GB |
| `clef-flash-gguf:c330d9ee7e9cc658ad45` | [bartowski/Cloudflare_clef-flash-GGUF](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF) + official head | GGUF `d7f376ea88c05e7bb1014dd5351a93df9dd8029e`; head `19cdcec8…5ba0` | 6.13 GB |
| `clef-27b-gguf:48cb5634b4a258de5a6b` | [bartowski/Cloudflare_clef-GGUF](https://huggingface.co/bartowski/Cloudflare_clef-GGUF) + official head | GGUF `e306f00c…426a7`; head `a010ac04…4953` | 17.48 GB |

Every artifact digest was taken from the Hugging Face LFS OID (SHA-256) at
the pinned revision and re-verified locally after download. The 27B BF16
release (54.7 GB) is documented in the family page as host-infeasible
locally; the GGUF profile covers it.

## Golden-fixture parity

Three fixture cases generated per profile by
`gen_clef_fixture` (a joint multi-question record exercising Noul + Choice +
Score in one pass — the joint head's designed operating mode — plus a
stated-verbatim Choice and a text-passthrough Noul). The
`clef_parity` integration tests replay each profile's golden JSON against a
fresh engine load with a 0.02 cross-execution probability tolerance:

| Profile | Replay | Result |
|---|---|---|
| `clef-flash` | 3 cases, 6 questions | pass |
| `clef-flash-gguf` | same cases | pass (Q4_K_M within tolerance of the BF16 oracle on the same prompts) |
| `clef-27b-gguf` | same cases | pass |

Bring-up spot answers (fixture case 1, clef-flash BF16):
`billing_refund` noul 0.980 (state asserts true), `category` billing 0.785
(correct), `severity` 1.93; GGUF: noul 0.978 / billing 0.806 / 1.913; 27B
GGUF: noul 0.988 / billing 0.950 / 1.995. All selections agree with the
oracle on every fixture question.

The GGUF loaders reproduce llama.cpp's `qwen3.5` storage contract, undoing
three transforms (verified against the BF16 checkpoint numerically):
group-interleaved value heads in `attn_qkv`/`attn_gate`, the decay stored as
`A = -exp(A_log)`, and pre-folded `1 + w` RMSNorm weights. The forward
kernels are the parity-verified shared Qwen3.5 kernels over a
`Qwen35Geometry` parameterization (hidden 4096 / 32 layers flash, 5120 / 64
layers 27B); the pinned Qwen3.5-4B profiles' parity fixtures were re-run
after the geometry refactor and still pass.

## Open: MLX 4-bit

The `mlx-community/clef-flash-4bit` execution path (in tree, feature `mlx`)
loads and runs end-to-end but its DeltaNet state on the quantized inputs
diverges from the CPU oracle, so the joint-head parity fixtures do not pass
and the profile is excluded from the loadable set and catalog. Debug notes
and the reproduction live in the family page's open items.

## Dataset evaluations

Clef-Flash BF16 (CPU bf16w/fp32c), 50 rows per dataset, Apple M4 Max —
recorded under
[`benchmarks/2026-10-01-local-model-suite/quality/clef-flash/`](../2026-10-01-local-model-suite/):

| Dataset | Rows | Accuracy | Macro-F1 | Brier | ECE (10 bins) |
|---|---|---|---|---|---|
| banking77 | 50 | 0.980 | 0.990 | 0.030 | 0.076 |
| SST-2 | 50 | 0.960 | 0.958 | 0.073 | 0.054 |
| AG News | 50 | 0.900 | 0.886 | 0.170 | 0.041 |

| CLINC150 (GGUF Q4_K_M) | 25 | in flight | — | — | — |

For scale, the Clef-Flash model card reports CLINC150+OOS macro-F1 97.4 on
its own Decision Index harness; the numbers above are OpenKind's own
request-path measurements on different (smaller) splits and are not
comparable to card numbers. The GGUF Q4_K_M candle kernels cost roughly two
to three minutes per CLINC150 row on this host (per-token causal conv and
DeltaNet over dequantized K-quants), so its CLINC150 evaluation was still
running when this record was written; the GGUF path's correctness is
established by the golden-fixture parity above, not by this number.

## What this record does not claim

No task qualification, no reviewed-decision gate, no release promotion, and
no quality comparison against other families: the samples are small (50-500
rows), the workloads are the standard curated datasets, and the Clef model
card's numbers come from a different harness.
