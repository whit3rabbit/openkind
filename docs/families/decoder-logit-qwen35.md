# Family: decoder-logit-qwen35

> Frozen Qwen3.5 hybrid text backbone (24 gated-DeltaNet linear-attention
> layers, 8 grouped-query attention layers) prompting the question as a JSON
> decision payload and reading temperature-calibrated next-token logits over
> the 16 SemIf option letters. Implements the readout protocol of SemIf
> (`TheoLeeCJ/SemIf`, MIT, now
> [`TheoLeeCJ/SemIf-OpenJev`](https://github.com/TheoLeeCJ/SemIf-OpenJev)) as
> served by the open `jevk5` runtime
> (`github.com/allebee/jevk5`, Apache-2.0).

## Status in openkind

**Rust-loadable (prototype profiles).** Two pinned profiles share this
module (the `laya` multi-profile precedent):

- **`decoder-logit-qwen35` `415bcf4a064e6dadcf85`** — `alibiserikbay/JevK5`
  v0.3 (Apache-2.0) at `c4f7fdb3aeab5582336406e78d3bef11bf98833d`,
  Qwen3.5-4B with the author's distilled rank-16 LoRA already merged into
  the weights; reference knockout runtime calibration (letter temperature
  1.22, knockout temperature 0.93 from `jevk5_config.json`).
- **`plumb-4b` `c1f080794d38e94a0bc2`** — `crh225/plumb-4b` (Apache-2.0)
  at `24f7bf77e7ee258a2d158c61ea2dce2b60321010`, a further JevK5 v0.2
  fine-tune merged to BF16; `jevk5` v0.2.0 **single-read** semantics (no
  knockout schedule, questions above 16 options fail closed) with the
  author server's per-type temperatures (choice/noul 2.07, score 1.2;
  `jevk5_config.json` pins 2.07). The reference server's JevBench
  `--noul-commit` band reporting is a scoring policy, not the model
  distribution, and is deliberately not reproduced.

The loader verifies `config.json`, `tokenizer.json`, the profile's runtime
config, and the single-file BF16 checkpoint by SHA-256 in place, executes
the shared native FP32 CPU backbone
([`qwen35/backbone`](../../crates/openkind-backends/src/qwen35/backbone/)),
reads the tied output-embedding rows of the answer letters as the readout
projection, and applies the profile's pinned calibration. Profiles serve
through `openkindd` via `--decoder-logit-qwen35-aliases` /
`--decoder-logit-qwen35-model-root` (JevK5) and `--plumb-4b-aliases` /
`--plumb-4b-model-root` (Plumb-4B), and are benchmarked through
`openkind-bench score --engine decoder-logit-qwen35|plumb-4b`.

Both are catalog-installable offline-first: `openkind pull
decoder-logit-qwen35:415bcf4a064e6dadcf85` or `openkind pull plumb-4b:c1f080794d38e94a0bc2`
downloads the pinned artifacts, verifies every SHA-256, and installs them
for `--installed-models` (see [`../MODELS.md`](../MODELS.md)). The catalog
also carries the ollaya-compatible alias `jevk5:4b` — ollaya serves the
same JevK5 v0.3 model as the author's Q8_0 GGUF, while this profile serves
the pinned unquantized BF16 checkpoint — and the board-name alias
`plumb:4b`; aliases resolve to the canonical pull name at pull time.

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

Plumb uses the same MLX adapter with its own explicit profile. Select
`plumb-4b-mlx-fp32` for benchmarks. In the daemon, configure Plumb aliases
and its model root, then use the shared
`--decoder-logit-qwen35-backend mlx-fp32` selector. The backend does not
substitute JevK5's calibration or limits for Plumb's.

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
| Wide questions | JevK5: the reference knockout schedule (near-equal groups of ≤ 16, a 16-finalist final pass, knockout-temperature sharpening). Plumb-4B: none — the reference `jevk5` v0.2.0 runtime has no knockout schedule and questions above 16 options fail closed |
| Work limits | JevK5: at most 32 options, 512 rendered tokens per pass, 1,536 aggregate per question. Plumb-4B: at most 16 options, 16,384 rendered tokens (the reference server's frozen admission) |
| Interruption | Caller cancellation and the queue-inclusive deadline are checked before and after every pass and decoder layer |
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
| 4 | JevK5 v0.3 | `alibiserikbay/JevK5` (pinned revision) | Rust-loadable prototype |
| 5 | Plumb-4B | `crh225/plumb-4b` (JevK5 v0.2 fine-tune, single read, temperature 2.07/1.2) | Rust-loadable prototype (`plumb-4b:c1f080794d38e94a0bc2`) |
| 12 | SemIf | `openjev/openjev` (the former `TheoLeeCJ/openjev`, moved to the `openjev` org; CC-BY-NC-4.0) | surveyed 2026-10-01, not host-feasible: the current checkpoint is 27B-class (64 layers, hidden 5,120, 54.7 GB BF16 — the shim serves `Qwen/Qwen3.8-27B`), beyond local fp32 CPU and 36 GB MLX memory, and its serving contract is env-configured (`READOUT_T`/`READOUT_NOUL_T`/`READOUT_PERMS`), not a pinned artifact. The earlier 4B board configuration this family's protocol mirrors no longer exists as a pinned checkpoint. Revisit if the authors ship a small pinned checkpoint |
| 13 | spark-s1-4b-v6 | `abhishek085/spark-s1-4b-v6` (same backbone, menu-style renderer instead of the JSON payload) | surveyed — renderer variant needs its own profile |
| 14 | Jobe Qwen3.5-4B | `MantisShrimpdev/jobe` | recipe over public `Qwen/Qwen3.5-4B` — no fine-tuned weights to pin (2026-10-01 re-survey) |
| 19 | local-jev | `amithgc/local-jev` | local server recipe over public base weights — nothing to pin (2026-10-01 re-survey) |
| 63 | OpenSourceJev | `sabeel111/OpenSourceJev` | llama.cpp recipe over public Qwen3.5-4B Q4_K_M — nothing to pin (2026-10-01 re-survey) |

An earlier revision of this table called SemIf, Jobe, local-jev, and
OpenSourceJev "gated checkpoints". That was wrong: the Hub 401s came from
repos that had moved or never held weights, not access gates. The 2026-10-01
re-survey corrected every row above; jqv (#22) is likewise a zero-shot
recipe over public `Qwen/Qwen3-32B`, and Cygnet (#1) is a frozen-Gemma-4
recipe with no fine-tune.

The family is the reference implementation class for the letter-logit
Qwen3.5 rebuilds on the leaderboard.

## What remains open

- No M2 reviewed-decision gate has run for either profile; their operating
  points are provisional and they carry no model-quality claim.
- Native FP32 CPU execution costs seconds per pass on the reference host;
  the MLX backend removes most of that cost for this family's shapes, but
  neither backend carries a model-quality claim.
- The `jevk5_config.json` digest pins the served temperatures; a new
  upstream revision that refits them is a new profile, not a silent update.

## Checkpoint replay

The integration tests always run offline contract checks. Golden checkpoint
replays return early when their model-root variable is unset or names a
nonexistent directory. Provide the matching pinned local artifacts to run
these gates:

```bash
# CPU replays for the two separate profiles
OPENKIND_DECODER_LOGIT_QWEN35_MODEL_ROOT="<pinned-jevk5-root>" \
  env -u RUST_LOG cargo test -p openkind-backends \
  --test decoder_logit_qwen35_parity \
  golden_replay_matches_the_pinned_checkpoint -- --exact --nocapture
OPENKIND_PLUMB_4B_MODEL_ROOT="<pinned-plumb-root>" \
  env -u RUST_LOG cargo test -p openkind-backends \
  --test decoder_logit_qwen35_parity \
  plumb_golden_replay_matches_the_pinned_checkpoint -- --exact --nocapture

# Plumb MLX replay, macOS arm64 with the mlx feature
SDKROOT=$(xcrun --show-sdk-path) \
  OPENKIND_PLUMB_4B_MODEL_ROOT="<pinned-plumb-root>" \
  env -u RUST_LOG cargo test -p openkind-backends --features mlx \
  --test decoder_logit_qwen35_parity \
  mlx_replay::plumb_mlx_golden_replay_matches_the_pinned_checkpoint \
  -- --exact --nocapture
```

For JevK5 MLX, set `OPENKIND_DECODER_LOGIT_QWEN35_MODEL_ROOT` and use test
`mlx_replay::mlx_golden_replay_matches_the_pinned_checkpoint` with the same
feature and SDK settings. Confirm that the replay prints an answer count,
not a skip message, and retain its output with the measured source state.

## Benchmark record

The [Plumb/Decider CPU record](../benchmarks/2026-09-30-jevbench-expansion/README.md)
covers the smoke workload and Plumb's 96-case choice diagnostic. It also
reports Plumb checkpoint replays, with missing raw evidence stated in the
record. These workloads differ from shape777 and establish no task quality.

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
