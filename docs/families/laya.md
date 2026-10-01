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
benchmark-eligible through `openkind-bench --engine laya-english` (and
siblings). All three are installable through `openkind pull` (registry
manifests under `registry/v1/manifests/laya-*.json`). Each also carries the
ollaya-compatible alias `laya:en` / `laya:multilingual` /
`laya:typed-decisions` in the catalog — the ollaya registry pins the same
byte-identical weights and tokenizers (see
[`../MODELS.md`](../MODELS.md) for the alias policy).

**MLX backend (2026-09-28).** The same pinned checkpoints also run on the
MLX/Metal backend (feature `mlx`, macOS arm64): `openkindd --laya-backend
mlx-fp32` and `openkind-bench --engine laya-english-mlx-fp32` (and
siblings). The MLX loader reads the identical digest-verified shard — the
third-party MLX conversions on the Hub (`aac6fef/laya-*-mlx`, created
2026-09-19) proved the weights transport unchanged (byte-identical tensors
under a small rename map) but openkind does not depend on them. Execution
arithmetic is `mlx-gpu-fp32-laya`; the candle CPU path remains the
correctness oracle.

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
- The MLX engine (`families/laya/mlx/`) mirrors the candle arithmetic op for
  op in mlx-rs FP32 arrays: weight-only norms, NeoX rotate-half RoPE with
  per-layer-type tables, symmetric sliding-window band masks, gated GELU
  MLP, and the bias-carrying pre-norm head. Weights are pre-transposed at
  load so every forward matmul consumes `(input, output)` weights. All MLX
  work runs under the process-wide `MlxRuntime` serialized stream, per
  [`../MLX.md`](../MLX.md).
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
- **MLX backend parity (2026-09-28).** The golden fixtures replay through
  the MLX engine (env-gated tests in
  `crates/openkind-backends/tests/laya_parity.rs`, module `mlx_replay`,
  enabled by `--features mlx` plus the `OPENKIND_LAYA_<PROFILE>_MODEL_ROOT`
  variables) with zero selection flips and maximum calibrated-probability
  drift **7.2e-6** (`laya-multilingual`), **6.5e-6** (`laya-english`), and
  **2.5e-6** (`laya-typed-decisions`) against the committed fixtures — far
  inside the workspace MLX gate of 0.005 and consistent with the
  independent conversion evidence published with the `aac6fef/laya-*-mlx`
  Hub repos (FP32 max error 2.7e-6, 63/63 argmax agreements, on M3 Max).

## Upstream optimization survey (unsloth Studio PRs, 2026-09-28)

Unsloth's Studio backend serves the same three checkpoints; their
merged/open PRs were surveyed for portable techniques:

- **[#12202](https://github.com/unslothai/unsloth/pull/12202) (merged) —
  vendor laya + fp16 weight storage.** openkind already stores the pinned
  shards fp16 and upcasts to FP32 at load, so the storage half is already
  the openkind shape. Their fp16 *compute* path stays a CUDA/x86 serving
  trade; on their own CPU numbers fp16 was slower than fp32 (77 ms vs
  65 ms, AVX512-FP16), and openkind keeps the FP32 CPU oracle.
- **[#12210](https://github.com/unslothai/unsloth/pull/12210) (merged) —
  skip the random vocab-embedding init at load** (peak host RAM 4.49 →
  2.38 GB). Not applicable: candle loads tensors straight from the
  digest-verified mmap, so no random initialization ever exists to skip.
- **[#12224](https://github.com/unslothai/unsloth/pull/12224) (open) —
  marker-only head + CUDA graphs.** The marker-only head (queries, residual,
  FFN only at option markers; K/V span all tokens; matches laya's forward to
  2e-6) applies only to the *last* head layer — layer 1 output is still
  needed at every position as layer 2's K/V input. In openkind's
  single-row request path the two head layers are ~3% of encoder cost
  (2 layers × ≤256 head tokens vs 22–28 layers × 512–1024 encoder tokens),
  so the portable win is ~1%, below the divergence risk it adds. CUDA
  graphs, the numpy collate, and their batched-serving wins are
  torch/CUDA-specific. Their int8 dynamic-quantization experiment was
  rejected by unsloth itself (30/36 answers flipped) and is not reconsidered
  here; BF16 serving fails openkind's frozen parity gates.

## Benchmark record

`openkind-bench score` over the standard shape777 workload (777 rows):

| Engine | Decisions/s | Input tokens/s | Peak RSS | Model load |
|---|---:|---:|---:|---:|
| `laya-english` (CPU) | 2.20 | — | 2.56 GB | 1.83 s |
| `laya-multilingual` (CPU) | 5.13 | — | 2.61 GB | 1.81 s |
| `laya-typed-decisions` (CPU) | 2.37 | — | 2.57 GB | 1.80 s |
| `laya-english-mlx-fp32` | 24.34 | 4,964 | 2.12 GB | 2.77 s |
| `laya-multilingual-mlx-fp32` | 56.37 | 11,431 | 2.94 GB | 3.54 s |
| `laya-typed-decisions-mlx-fp32` | 23.77 | 4,849 | 2.13 GB | 2.87 s |

CPU rows are the 2026-09-27 record
([`../benchmarks/2026-09-27-laya/`](../benchmarks/2026-09-27-laya/)); MLX
rows (and same-schema CPU re-runs measuring ~2.2–5.3 dec/s and 455–1,075
tok/s) are the 2026-09-28 MLX campaign
([`../benchmarks/2026-09-28-laya-mlx-campaign/`](../benchmarks/2026-09-28-laya-mlx-campaign/README.md)),
which measures the MLX backend at **~10.2–10.9×** the CPU path with equal or
lower peak RSS. Also in [`../BENCHMARKS.md`](../BENCHMARKS.md). Request-path
timing only; the upstream model-card accuracies belong to the reference's
own evaluation suites and are not `openkind` measurements.

## Limits

- `MAX_SEQUENCE_TOKENS` = 512 (`laya-english`) / 1024 (others); heads that
  would drop a marker fail closed instead of truncating.
- `MAX_CANDIDATES = 100` per question, matching the reference server's
  per-question choice guard.
- `predict_long` (8192-token windowed documents), the language router, and
  batched multi-state forwards are out of scope.
