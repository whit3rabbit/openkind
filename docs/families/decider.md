# Family: decider

> Frozen Qwen3.5 hybrid text backbone prompting each question as a plain
> state-first decision block (`Context:` + `Question:`/`Options:` +
> `Answer: (`) and reading temperature-calibrated next-token logits at the
> answer slot, restricted to uppercase label tokens. Implements the serving
> contract of the `decider` package shipped with the checkpoint
> (`github.com/Mapika/decider`, Apache-2.0). No chat template wraps the
> prompt and no token is ever sampled.

## Status in openkind

**Rust-loadable (prototype profile).** The pinned profile
`decider-4b:0529bf6f2bed84641701` serves `Mapika/decider-4b` (4b-v2.1,
Apache-2.0) at `eb5fbdfc9448473ec25e399882912863afbdb70e` — Qwen3.5-4B-Base
plus one supervised pass and a rank-64 LoRA, merged into the BF16 weights.
The loader verifies `config.json`, `tokenizer.json`, `decider_config.json`,
and the single-file checkpoint by SHA-256 in place, executes the shared
native FP32 CPU backbone
([`qwen35/backbone`](../../crates/openkind-backends/src/qwen35/backbone/)),
reads the tied output-embedding rows of the answer labels at the final
`Answer: (` position, and applies the pinned per-type calibration
temperatures (`decider_config.json`: choice 1.11, noul 1.56, score 1.287;
the scalar 1.099 is the fallback the by-type map replaces). It serves
through `openkindd` via `--decider-4b-aliases` /
`--decider-4b-model-root` and is benchmarked through
`openkind-bench score --engine decider-4b`.

It is catalog-installable offline-first: `openkind pull
decider-4b:0529bf6f2bed84641701` downloads the pinned artifacts, verifies
every SHA-256, and installs them for `--installed-models` (see
[`../MODELS.md`](../MODELS.md)).

Profile semantics: `ConditionalOnOfferedOptions`; no state is retained
across questions or requests — every row is an independent full-sequence
forward; no token is ever sampled.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Qwen3.5 hybrid text decoder, 32 layers, hidden 2,560 (same frozen geometry as the `decoder-logit-qwen35` family) |
| Prompt layout | Plain state-first, no chat template: `Context:\n<state>` (keep-first truncated at 32,768 tokens), then `\n\nQuestion: <text>\nOptions:` and `\nAnswer: (` |
| Option rendering | Up to 10 options as `(A) text` lines tokenized as one string; wider sets build lines from token ids so every label stays one token (label head `A..Z` + single-token two-letter strings, 255 entries) |
| Readout | Final-position hidden state dotted with the tied embedding rows of the option labels; softmax at the type temperature |
| Score questions | Isolated levels (`isolated_levels: true`): one yes/no row per level (`…\nProposed answer: <level>\nDoes the proposed answer fit?`), per-level P(fits) normalized into the level distribution |
| Work limits | Up to 255 Choice options, 10 Score levels, 32,768-token state cap (keep-first), 65,536-token row cap |
| State serialization | Strings pass through; objects/arrays serialize as Python `json.dumps(..., ensure_ascii=False)` with `_index` annotations on arrays of ≥ 8 elements |
| Interruption | Caller cancellation and the queue-inclusive deadline are checked before and after every row and decoder layer |
| Continuation state | None — cache-free full forwards only |
| Text generation | None |
| Calibration | Pinned per-type temperatures from `decider_config.json`; loaders reject other values |
| Semantic none | None of its own; an offered `__none__` key is scored as an ordinary option |

## Renderer determinism

The profile pins the reference prompt bytes on the `_NoShuffle` serving
path: options render in the order given and the serving shuffle/subsample of
the training path never runs. Declared determinism differences, both
inherited from the wire: our Choice `criteria` map is a hash map, so options
render in the wire's sorted label order where the reference preserves caller
insertion order, and a null criteria description renders the bare label
exactly as the reference does. The reference's `neutralize_none` rewrite is
pinned off by `decider_config.json` for this checkpoint and is not
implemented.

## JevBench systems covered

| JevBench rank | System | Checkpoint | Profile status |
|---|---|---|---|
| 7 | decider-4b v2 | `Mapika/decider-4b` (pinned revision) | Rust-loadable prototype |

The decider-2b sibling (`Mapika/decider-2b`, Qwen3.5-2B) shares the runtime
contract and is a candidate future profile.

## What remains open

- No M2 reviewed-decision gate has run for this profile; its operating point
  is provisional and it carries no model-quality claim.
- Native FP32 CPU execution costs seconds per row on the reference host.
  No MLX backend exists for this family yet (the checkpoint layout matches
  the shared Qwen3.5 MLX backbone, so a `decider` MLX module is a
  candidate follow-up).
- The `decider_config.json` digest pins the served temperatures and flags;
  a new upstream revision that refits them is a new profile, not a silent
  update.
- The reference schema-cache path (`schema_first`, CUDA-graph reuse) is out
  of scope: `decider_config.json` pins `schema_first: false` for this
  checkpoint.

## What this page does not say

No accuracy, calibration, or leaderboard claims about decider-4b; those
belong to the checkpoint authors and, for our evidence, to
[`../BENCHMARKS.md`](../BENCHMARKS.md). The golden fixture records
distribution-level agreement, not decision quality.
