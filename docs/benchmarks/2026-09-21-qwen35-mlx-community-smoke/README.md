# 2026-09-21 — qwen35-MLX community-checkpoint smoke sweep

First `opendecision-bench` run against the MLX-community export:
`mlx-community/Qwen3.5-4B-MLX-bf16`, Hub revision
`475632ded9a95863da4e4b235ab9ccbc5d3cc6bf`, loaded through the digest-locked
community adapter (checkpoint identity `mlx-community-qwen35-4b-bf16`).
**Throughput evidence only.** The community export is a different source
model and conversion (`Qwen/Qwen3.5-4B` via an `mlx-vlm` fix branch, not the
pinned `Qwen3.5-4B-Base`): it already fails the frozen Phase 3B parity gates
(probability error `0.9999983`, four argmax changes — see the Phase 3M
working-tree verification note). Nothing in this record is a parity or
model-quality claim.

- Schema: `opendecision-bench/v1` (`summary-qwen35-mlx-fp32.json`)
- Engine: `qwen35-mlx-fp32` (`mlx-core-0.32.2/fp32/reference-ops`),
  bundle version `2ij.2.0`
- **Checkpoint caveat**: the summary's `model_revision` field
  (`1001bb4d826a52d1f399e183466143f4da7b741b`) is read from the head bundle
  manifest and describes the fitted head's base, **not** the community
  checkpoint this run actually loaded. The checkpoint identity is carried by
  this README and the adapter's state identity, not by that summary field.
- Fixture: `crates/opendecision-bench/fixtures/decisions_smoke.jsonl`,
  SHA-256 `3a673e843690b942658b4c9de6cc594770185756098d356e78dcd3efd5cffeeb`,
  12 rows grouped per state (4 groups)
- Host: Mac16,5 Apple M4 Max 36 GiB (named Mac), warm process, untimed warmup
- Commit: `ddbc5dd7e46bb1318bb98af8b0d924d78d4452ee` with the (then)
  uncommitted working-tree changes that added MLX engine/bench dispatch
  (dirty-tree record)
- Digest-locked tokenizer: vendored Phase 3B copy (same as the pinned-base
  record)

## Results (single sample per strategy)

| Strategy | Total (s) | Decisions/s | Pinned-base same cell |
|---|---:|---:|---:|
| `repeated_full` | 30.51 | 0.393 | 31.64 s |
| `nested_sequential` | 8.07 | 1.487 | 8.32 s |
| `nested_batched` | 8.12 | 1.478 | 8.31 s |
| `choose_strategy` | 8.54 | 1.405 | 8.32 s |

Throughput is statistically indistinguishable from the pinned-base FP32 run
on this smoke fixture (all cells within ~±4% single-sample noise, load
~22 s). This matches expectations: same architecture, same reference-ops
execution path, same tokenizer — only the weight values differ. It confirms
the community adapter executes the complete engine request path (including
nested branch topology) at full speed, but says nothing about the model's
decision quality. Peak process RSS at summary time: 11.64 GB.

`cross_strategy_answer_parity_clean: false`, with the same cause as the
pinned-base MLX record: exact cross-strategy wire equality is stricter than
the MLX execution shapes' numerical identity (see the pinned-base record for
the quantified bound).

## Interpretation boundaries

- Single-sample smoke cells: quote ratios, not absolutes.
- Community-checkpoint throughput is an implementation observation. Because
  the source model differs, it must not be used for head-quality,
  parity, or release claims, and the near-identical timings are not evidence
  of model equivalence.
- 4-bit/8-bit community exports are quantized profiles and are not supported
  by the current adapter.
