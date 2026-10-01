# Family: decoder-logit-qwen3

> The raw direct-logit controls of the JevBench board: unmodified dense
> Qwen3 checkpoints (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`,
> `Qwen/Qwen3-4B-Instruct-2507`, all Apache-2.0) prompted with the family's
> letter-pass decision renderer and read through the tied embedding rows of
> the option letters at the answer slot. No fine-tuned decision weights
> exist for these checkpoints — their board rows measure how far a stock
> model plus a constrained scoring readout goes.

## Status in openkind

**Rust-loadable (prototype profiles).** Three pinned profiles share this
module:

- **`decoder-logit-qwen3-06b`** `d900f4af57509fe02e62` —
  `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`
  (board #79 raw-logit control).
- **`decoder-logit-qwen3-17b`** `8119b9271f8d011e7d03` —
  `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
  (board #81).
- **`decoder-logit-qwen3-4b`** `9dfaf11792a8d061b6b8` —
  `Qwen/Qwen3-4B-Instruct-2507` at
  `cdbee75f17c01a7cc42f958dc650907174af0554` (board #21, the
  highest-ranked raw control, above several fine-tunes OpenKind serves).

The loaders verify `config.json`, `tokenizer.json`, and every checkpoint
shard by SHA-256 in place, execute the shared dense-Qwen3 candle forward
([`qwen3guard/arch.rs`](../../crates/openkind-backends/src/families/qwen3guard/arch.rs),
parameterized per profile), and read the tied output-embedding rows of the
answer letters at the final position. `Qwen/Qwen3-8B` (board #29) is
surveyed but not landed: fp32 CPU execution needs roughly 32 GB of weights
alone, which does not fit the reference host alongside the runtime.

They serve through `openkindd` via
`--decoder-logit-qwen3-aliases "06b=<alias>[,…];17b=<alias>[,…];4b=<alias>[,…]"`
and `--decoder-logit-qwen3-model-roots "06b=<path>;17b=<path>;4b=<path>"`,
and are benchmarked through `openkind-bench score --engine
decoder-logit-qwen3-06b|decoder-logit-qwen3-17b|decoder-logit-qwen3-4b`.

They are catalog-installable offline-first: `openkind pull
decoder-logit-qwen3-4b:9dfaf11792a8d061b6b8` (or the 06b/17b siblings)
downloads the pinned artifacts, verifies every SHA-256, and installs them
for `--installed-models` (see [`../MODELS.md`](../MODELS.md)). The 4B
profile carries the board-name alias `qwen3:4b`.

Profile semantics: `ConditionalOnOfferedOptions`; no state is retained
across questions or requests — every question is one full-sequence
forward; no token is ever sampled.

## Control contract (declared)

The upstream checkpoints ship no serving runtime, so the renderer is a
declared construction of this family:

| Aspect | Contract |
|---|---|
| Decision payload | The `jevk5` letter-pass bytes (SemIf-origin system instruction, `evidence`/`criterion`/lettered `options` JSON with Python `json.dumps` shapes), identical to the `decoder-logit-qwen35` family |
| Chat template | Each checkpoint's own pinned template with thinking off: 0.6B/1.7B close the assistant turn with the empty `<think>` block; the 2507-Instruct 4B ends at the bare `<\|im_start\|>assistant\n` prompt |
| Calibration | Temperature 1.0 — a raw control fits nothing; loaders reject any other value through the same profile constant |
| Pass schedule | One forward per question; questions above 16 options fail closed (no knockout schedule exists to mirror) |
| Readout | Final-position hidden state dotted with the tied embedding rows of the option letters |
| Work limits | At most 16 options, 512 rendered tokens per pass |
| Text generation | None |

## JevBench systems covered

| JevBench rank | System | Checkpoint | Profile status |
|---|---|---|---|
| 21 | Raw Qwen3 4B Instruct 2507 direct logits | `Qwen/Qwen3-4B-Instruct-2507` (pinned revision) | Rust-loadable prototype |
| 29 | Raw Qwen3 8B direct logits | `Qwen/Qwen3-8B` | surveyed — fp32 CPU infeasible on the reference host |
| 79 | Raw Qwen3 0.6B direct logits | `Qwen/Qwen3-0.6B` (pinned revision) | Rust-loadable prototype |
| 81 | Raw Qwen3 1.7B direct logits | `Qwen/Qwen3-1.7B` (pinned revision) | Rust-loadable prototype |

## What remains open

- No M2 reviewed-decision gate has run for these profiles; they are
  untrained controls and carry no deployment claim by construction.
- No MLX backend exists for the dense-Qwen3 controls; the shared MLX
  Qwen3.5 backbone serves the hybrid (DeltaNet) geometry, not dense Qwen3.
- The `Qwen/Qwen3-8B` row stays surveyed-only unless a quantized or MLX
  route is qualified for the family.

## What this page does not say

No accuracy, calibration, or leaderboard claims about the Qwen3 controls;
those belong to the benchmark operators and, for our evidence, to
[`../BENCHMARKS.md`](../BENCHMARKS.md). The golden fixtures record
distribution-level agreement, not decision quality.
