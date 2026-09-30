# Family: decoder-logit-qwen35

> Frozen Qwen3.5 hybrid text backbone (24 gated-DeltaNet linear-attention
> layers, 8 grouped-query attention layers) prompting the question as a JSON
> decision payload and reading temperature-calibrated next-token logits over
> the 16 SemIf option letters. Implements the readout protocol of SemIf
> (`TheoLeeCJ/SemIf`, MIT) as served by the open `jevk5` runtime
> (`github.com/allebee/jevk5`, Apache-2.0).

## Status in openkind

**Rust-loadable (prototype profile).** The pinned profile
`415bcf4a064e6dadcf85` serves `alibiserikbay/JevK5` v0.3 (Apache-2.0) at
`c4f7fdb3aeab5582336406e78d3bef11bf98833d` — Qwen3.5-4B with the author's
distilled rank-16 LoRA already merged into the weights. The loader
verifies `config.json`, `tokenizer.json`, `jevk5_config.json`, and the
single-file BF16 checkpoint by SHA-256 in place, executes the shared native
FP32 CPU backbone
([`qwen35/backbone`](../../crates/openkind-backends/src/qwen35/backbone/)),
reads the tied output-embedding rows of the answer letters as the readout
projection, and applies the pinned calibration temperatures
(`jevk5_config.json`: letter temperature 1.22, knockout temperature 0.93).
It serves through `openkindd` via
`--decoder-logit-qwen35-aliases` / `--decoder-logit-qwen35-model-root` and is
benchmarked through `openkind-bench score --engine decoder-logit-qwen35`.

It is catalog-installable offline-first: `openkind pull decoder-logit-qwen35:415bcf4a064e6dadcf85` downloads the pinned artifacts, verifies every SHA-256, and installs them for `--installed-models` (see [`../MODELS.md`](../MODELS.md)).

**MLX backend (2026-09-29).** The same pinned artifacts also run on the
MLX/Metal backend (feature `mlx`, macOS arm64): `openkindd
--decoder-logit-qwen35-backend mlx-fp32` and `openkind-bench --engine
decoder-logit-qwen35-mlx-fp32`. No MLX conversion of JevK5 exists on the Hub
(surveyed 2026-09-29), so the backend reads the identical digest-verified
single-file BF16 checkpoint through the parity-verified Qwen3.5 MLX
backbone (survey-checkpoint descriptor; BF16 widened to FP32 exactly on
load, tied embedding kept host-resident for the letter readout). The
arithmetic identity is `mlx-gpu-fp32-jevk5`; the candle CPU path remains
the correctness oracle.

Profile semantics: `ConditionalOnOfferedOptions` (an offered `__none__` key
is scored as an ordinary option); no state is retained across questions or
requests — every pass is an independent full-sequence forward; no token is
ever sampled.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Qwen3.5 hybrid text decoder, 32 layers, hidden 2,560 (same frozen geometry as the pinned `encoder-state-first` base) |
| Tokenization | Pinned Qwen chat template with thinking off; the decision is one JSON user message (`evidence`, `criterion`, lettered `options`) |
| Forward pattern | One full-sequence forward per pass; up to 16 options per pass |
| Readout | Final-position hidden state dotted with the tied embedding rows of the option letters; softmax at the letter temperature |
| Wide questions | The reference knockout schedule: near-equal groups of ≤ 16, a 16-finalist final pass, and knockout-temperature sharpening |
| Continuation state | None — cache-free full forwards only |
| Text generation | None |
| Calibration | Pinned scalar temperatures fitted by the checkpoint author; loaders reject other values |
| Semantic none | None of its own; an offered `__none__` key is scored as an ordinary option |

## Renderer determinism

The profile pins the exact prompt bytes of the reference runtime. Two
declared determinism differences remain, both inherited from the wire: our
`criteria` maps are hash maps, so options and payload keys render in sorted
order where the reference preserves caller insertion order, and our Noul
default criterion text is the reference's "The proposition is
true/false." (identical) only when the caller supplies no criteria. Option
order does not change the answer semantics: the distribution stays aligned
to the sorted candidate order used by the shared wire mapping.

## JevBench systems covered

| JevBench rank | System | Checkpoint | Profile status |
|---|---|---|---|
| 5 | JevK5 v0.3 | `alibiserikbay/JevK5` (pinned revision) | Rust-loadable prototype |
| 2 | Plumb-4B | `crh225/plumb-4b` (JevK5 fine-tune, temperature 2.07) | surveyed — second profile pending |
| 17 | spark-s1-4b-v6 | `abhishek085/spark-s1-4b-v6` (same backbone, menu-style renderer instead of the JSON payload) | surveyed — renderer variant needs its own profile |
| 13 | SemIf | `TheoLeeCJ/openjev` | Blocked — gated checkpoint (Hub 401/404 with our credentials, 2026-09-27); the readout protocol itself is already implemented here |
| 14 | Jobe Qwen3.5-4B | `MantisShrimpdev/jobe` | Blocked — gated checkpoint |
| 15 | local-jev | `amithgc/local-jev` | Blocked — gated checkpoint |
| 23 | OpenSourceJev | `sabeel111/OpenSourceJev` (Qwen3.5-4B Q4_K_M GGUF) | Blocked — gated checkpoint |

The family is the reference implementation class for the letter-logit
Qwen3.5 rebuilds on the leaderboard.

## What remains open

- The Plumb-4B second profile (same runtime, own checkpoint and temperature)
  is surveyed but not loaded.
- No M2 reviewed-decision gate has run for this profile; its operating point
  is provisional and it carries no model-quality claim.
- Native FP32 CPU execution costs seconds per pass on the reference host;
  the MLX backend removes most of that cost for this family's shapes, but
  neither backend carries a model-quality claim.
- The `jevk5_config.json` digest pins the served temperatures; a new
  upstream revision that refits them is a new profile, not a silent update.

## Benchmark record

Shape777 MLX record (2026-09-29, single sample): 0.43 decisions/s, 117
input tok/s, 6.98 GB peak RSS, 20.2 s model load — roughly 2.7× the
family's recorded smoke-scale CPU row (0.16 decisions/s), with the gain
bounded by full-forward-per-pass compute and 1–2.5k-token JSON payloads:
[`../benchmarks/2026-09-29-mlx-counterparts/`](../benchmarks/2026-09-29-mlx-counterparts/).
MLX parity (2026-09-29): the golden fixtures replay through the MLX engine
(env-gated test in
`crates/openkind-backends/tests/decoder_logit_qwen35_parity.rs`, module
`mlx_replay`, enabled by `--features mlx` plus
`OPENKIND_DECODER_LOGIT_QWEN35_MODEL_ROOT`): 9 answers, max probability
drift 1.003e-6, zero selection flips — inside the workspace MLX gate of
0.005.

## What this page does not say

No accuracy, calibration, or leaderboard claims about JevK5 or Plumb-4B;
those belong to the checkpoint authors and, for our evidence, to
[`../BENCHMARKS.md`](../BENCHMARKS.md). The golden fixture records
distribution-level agreement, not decision quality.
