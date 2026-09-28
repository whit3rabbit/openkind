# Family: laya

> ModernBERT-family decision encoders with a shared typed-decision head,
> serving the published Laya (System One) contract.

## Status in openkind

**Implemented (2026-09-27).** The reference project
(`github.com/NandhaKishorM/laya`, Apache-2.0, by Convai Innovations) ships
three open checkpoints, the reference implementation, training configs, and
the evaluation protocol. Apache-2.0 grants the implementation and
redistribution rights, and the per-checkpoint `rl_agent_config.json`
(shipped temperatures, sequence budgets, encoder identity) plus the
reference source make the readout contract readable instead of invented.
The multilingual encoder derives from `jhu-clsp/mmBERT-base` (MIT).

One family module hosts three pinned profiles sharing the same
tokenization-forward-readout contract; they differ in checkpoint, sequence
budgets, temperature tables, and encoder dimensions:

| Profile | Base checkpoint | Encoder | max_len / head_max_len |
|---|---|---|---|
| `laya-english` (`c8ea29bf1e33a343c4b7`) | `convaiinnovations/laya` at `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | ModernBERT-large (28 layers, hidden 1024, 16 heads) | 512 / 192 |
| `laya-multilingual` (`f4064eb56fb7f7d325e1`) | `convaiinnovations/laya-multilingual` at `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` | mmBERT-base (22 layers, hidden 768, 12 heads, both RoPE bases 160000) | 1024 / 256 |
| `laya-typed-decisions` (`9d28cfa9567902801ed1`) | `convaiinnovations/laya-typed-decisions` at `1a793eb568e6718f15941d08f85432581df534e3` | ModernBERT-large (fine-tuned) | 1024 / 256 |

Each implements `DecisionEngine` behind the bounded family scaffold,
registers in `openkindd` via `--laya-english-aliases` /
`--laya-english-model-root` (and the per-profile equivalents), and is
benchmarked through `openkind-bench --engine laya-english` (and siblings).
All three are installable through `openkind pull` (registry manifests under
`registry/v1/manifests/laya-*.json`).

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | ModernBERT-shaped encoder (config-driven; shared with `encoder-instruct-label`'s hand-implemented backbone): weight-only norms, fused bias-free QKV, per-layer-type RoPE (global 160000 / sliding 10000 — both 160000 for mmBERT), global-every-3rd-layer vs sliding-window-128 attention, GEGLU MLP |
| Tokenization | Each checkpoint's shipped fast `tokenizer.json` used raw: ByteLevel BPE with NFC for the English profiles (`[CLS]` 50281 / `[SEP]` 50282 / `[PAD]` 50283 / `[MASK]` 50284), Metaspace BPE with byte fallback and a 256k vocabulary for multilingual (`<bos>` 2 / `<eos>` 1 / `<pad>` 0 / `<mask>` 4) |
| Forward pattern | One sequence per question: `[CLS] "<type> question: <instructions>" [SEP] ([MASK] + option span)… [SEP] <state> [SEP]`. Option spans cap at 48 text tokens; `head_max_len` overflow cuts every span (marker included) to `max(4, (head_max_len − 16) / options)` and the instruction to `max(8, budget)`. The state keeps its first `room` tokens (newest `room` for array states); literal mask-token text in user input is replaced with a space. One full forward per question; nothing generates text |
| Readout | Shared typed-decision head: type embedding (3 × hidden) added to every encoder position, two pre-norm `nn.TransformerEncoderLayer`s (packed-QKV with bias, nhead = hidden/64, ReLU FFN at 4×hidden), then `LayerNorm → Linear → GELU → Linear(1)` at each marker position — one logit per option span |
| Continuation state | None — every question is an independent full forward |
| Text generation | None |
| Calibration | The checkpoints ship fitted per-type and per-option-count temperatures in `rl_agent_config.json`; decode applies them under the reference's `[0.5, 5.0]` clamp, under which the shipped `choice:11+` sharpening bucket (0.1006) resolves to 0.5. No openkind-side fit |
| Semantic none | The family models no semantic-none mass; a reserved `__none__` key is an ordinary offered option (conditional probability space) |
| Excluded | The reference's `act_head` (act/escalate) — upstream documents its output as carrying no usable signal (issue #185) and the Jev wire has no field for it — and the language router and `predict_long` windowing, both deferred |

## Implementation notes

- The ModernBERT encoder is the shared
  [`families/modernbert.rs`](../../crates/openkind-backends/src/families/modernbert.rs)
  module (extracted from `encoder-instruct-label`, whose golden fixture
  gates the extraction); laya pins the large and mmBERT configurations.
  mmBERT's both-160000 RoPE is exactly the upstream mismatch the reference
  carries an explicit rope-config fixup for.
- The multilingual encoder config ships a stale `cls_token_id` (the eos id);
  the reference builds sequences from the tokenizer's ids, and so does the
  openkind renderer, which verifies the pinned special ids at load.
- `Choice` options are evaluated in the shared sorted-label order; the
  reference evaluates in caller dict order. Marker readouts are
  order-sensitive, so the profile owns the deterministic order.
- Object and array states serialize as compact JSON with Python
  `json.dumps` separators (`", "`, `": "`) and sorted keys, matching the
  reference's serialization byte for byte except that the wire parser's map
  order is not preserved.
- The shards store fp16 and are upcast to FP32 at load (mmap in place);
  execution arithmetic is `candle-cpu-fp32-laya`.
- The reference's Python loader rewrites `tokenizer/tokenizer_config.json`
  in place for older-transformers compatibility (the multilingual
  checkpoint's list-valued `extra_special_tokens`); openkind digest-pins the
  pristine hub bytes. `scripts/laya_reference.py` snapshots and restores
  that file so reference runs leave the model root reproducible.

## Validation

- Digest-checked golden fixtures over 15 stated-fact readout-surface cases
  (both noul criteria modes, the reserved `__none__` key, described and bare
  choice options, 12-way choices, 4- and 6-level rubrics, object/array/long
  states, head-budget overflow, mask-token sanitization, Latin and Cyrillic
  script), shared by every profile:
  `crates/openkind-backends/tests/fixtures/laya-<profile>_<profile-id>/`.
- Cross-implementation check against the Python reference (`laya` 0.3.21,
  torch CPU fp32) over the identical requests: probabilities and scores
  agree to `≤ 5.3e-5` across all 45 profile × case combinations with zero
  selection flips — at the reference's own 4-decimal output rounding floor.
  Regenerate and compare with
  `crates/openkind-backends/src/bin/gen_laya_fixture.rs` and
  [`scripts/laya_reference.py`](../../scripts/laya_reference.py).
- Task quality is **not** claimed: the upstream model card reports the base
  checkpoints are near chance zero-shot on typed decisions (0.362 against a
  0.318 random baseline) and describes Laya as "a fast base to specialise".
  M2 labeled-dataset qualification is a separate gate.

## Benchmark record

`openkind-bench score` over the standard shape777 workload (777 rows):
`laya-english` 2.20 decisions/s, 2.56 GB peak RSS, 1.83 s model load;
`laya-multilingual` 5.13 decisions/s, 2.61 GB, 1.81 s;
`laya-typed-decisions` 2.37 decisions/s, 2.57 GB, 1.80 s. Recorded in
[`../BENCHMARKS.md`](../BENCHMARKS.md) and
[`benchmarks/2026-09-27-laya/`](../benchmarks/2026-09-27-laya/).
Request-path timing only; the upstream model-card accuracies belong to the
reference's own evaluation suites and are not `openkind` measurements.

## Limits

- `MAX_SEQUENCE_TOKENS` = 512 (`laya-english`) / 1024 (others); heads that
  would drop a marker fail closed instead of truncating.
- `MAX_CANDIDATES = 100` per question, matching the reference server's
  per-question choice guard.
- `predict_long` (8192-token windowed documents), the language router, and
  batched multi-state forwards are out of scope.
