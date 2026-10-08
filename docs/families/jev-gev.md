# Family: jev-gev (JEV: loader implemented, unverified on real weights; GEV: readout only)

> JEV-27B-VL and GEV-26B-Decide answer typed `bool`, `score`, and `choice`
> questions from one forward pass over a fixed prompt. Both ship as 8-bit
> MLX conversions. OpenKind has pinned both. For JEV it also has a quantized
> MLX backbone, a loader, and an engine that pass offline contract tests but
> have not run on the real weights. For GEV it has only the readout.

## Pinned sources

Metadata is in
[`registry/v1/jev-gev-mlx-models.json`](../../registry/v1/jev-gev-mlx-models.json),
checked 2026-10-08. It records per-file sizes and SHA-256 digests for the six
weight shards, `config.json`, the safetensors index, and the tokenizer files.
Weights are Apache-2.0.

| | JEV-27B-VL | GEV-26B-Decide |
|---|---|---|
| MLX repository | [`nativ-community/JEV-27B-VL-MLX-8bit`](https://huggingface.co/nativ-community/JEV-27B-VL-MLX-8bit) | [`nativ-community/GEV-26B-Decide-MLX-8bit`](https://huggingface.co/nativ-community/GEV-26B-Decide-MLX-8bit) |
| Hub revision | `a871d5f8787b3d8ce9d618e7260e959393030c7b` | `8f04bf3f3204f067625a2716c54dd68de7200ec6` |
| Source model | `autotrust/JEV-27B-VL` at `4000d2393be6718e604f8e7dca563a780ab78e78` | `autotrust/GEV-26B-Decide` at `7c89590ead085bf77630b4bf68264ea30b6ddc78` |
| Adapted from | `Qwen/Qwen3.8-27B` | `google/gemma-4-26B-A4B-it` |
| Weights | 30.7 GB, six shards | 28.0 GB, six shards |
| Quantization | affine 8-bit, group size 64; `lm_head` and vision tower unquantized | affine 8-bit, group size 64; `head` in FP32 |
| mlx-vlm reference | `Lazarus-931/mlx-vlm` `feat/jev` at `6ef5c0d13b847ef2a3c3586276af9c4b75da4686` | `Lazarus-931/mlx-vlm` `feat/gev` at `227940979130cbeb3518d07182d4fa2b6ea68de6` |

The System 1 LoRA is already merged in both conversions: neither weight index
has a `base_model.model.*` tensor. The reference's load-time `merge_lora`
matters only for raw adapter checkpoints. Neither reference is in an mlx-vlm
release.

## Protocol

Both models read each question in a single pass and generate nothing, which
fits OpenKind's no-autoregression invariant. The prompt is:

```text
[kind] {noul|score|choice}
[state] {state text}
[question] {instruction}
[options]
{one option per line}
[decision]:
```

GEV prepends `<bos>`. Choice lines are `{label}) {key}: {criterion}`, or just
the key when the criterion is absent. Score lines are the bare levels `0` to
`5`; the rubric text is never shown to the model and is only response
metadata. Bool lines are `false` and `true`. Probabilities are a softmax over
the option scores divided by a per-kind temperature from `decision_config`.

| | JEV | GEV |
|---|---|---|
| Backbone | Qwen3.5 hybrid: 64 layers, 3 DeltaNet to 1 full attention, hidden 5120 | Gemma 4 MoE: 30 layers, hidden 2816, 128 experts with top 8, dense MLP in parallel |
| Option score | Vocabulary logit at the final position plus a per-slot bias | 24-wide linear head over the final hidden state, FP32, `30 * tanh(x / 30)` |
| bool | tokens 3721, 1802 | head slots 0 to 1 |
| score | tokens for `0` to `5` | head slots 2 to 7 |
| choice | single-token letters `A`..`Z`, then `AA`..`ZZ`, up to 256 options in one pass | slots 8 to 23, labels `A`..`P`; more than 16 options use a grouped tournament |

Label tokens are derived from the checkpoint tokenizer, not hard-coded: a label
qualifies when it encodes to one token that also appears in the encoding of
`x\n{label}) y`. Against the pinned JEV `tokenizer.json`
(`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`), 256
labels derive, and the first 16 equal `verbalizer_ids[8:24]` (tokens 32 to 47).

The GEV tournament splits more than 16 options into `ceil(n / 16)` balanced
groups, scores each, scores the group winners plus the best remaining options
(16 in total) together, then scales each in-group probability by its group's
share of that final pass.

## What is implemented

[`families::jev_protocol`](../../crates/openkind-backends/src/families/jev_protocol/mod.rs)
holds everything below. None of it is registered with the daemon, CLI,
benchmark harness, or catalog.

**Readout, shared by JEV and GEV** (runs on any host):

- `DecisionConfig`: validates the `decision_config` object and fails closed on
  the other protocol's fields, unequal slot widths, or a non-positive
  temperature.
- `render_prompt`, `ChoiceLabels`, and `choice_option_text`.
- Slot readout: `jev_token_ids` and `jev_probabilities` (bias, per-kind
  temperature), and `gev_window`, `gev_softcap`, and `gev_probabilities`.
- `gev_choice_tournament`, with the reference's tie-breaks.

Its tests use the two pinned `decision_config` objects as fixtures, kept equal
to the registry index by a test. Expected values come from a line-for-line
Python port of the mlx-vlm functions run in f64; the tournament is checked at
17, 40, and 256 options. Prompts are checked against the reference format.

**JEV engine** (runs on any host with a mock backbone; the MLX parts need
`--features mlx` on macOS arm64):

- `JevConfig` parses the pinned root `config.json` and rejects any other
  geometry or quantization. `expected_tensors` lists every tensor the loader
  needs, and a test proves the list equals the real checkpoint's 2,178 tensor
  headers, apart from the vision tower: no missing or extra names, with the
  same dtypes and shapes. The headers are vendored in
  `tests/fixtures/jev_gev/jev_tensor_manifest.json`.
- `JevEngine` renders one prompt per question, tokenizes it with the pinned
  tokenizer, reads the option logits, and builds the wire answer. It rejects
  booleans with criteria, score questions without six levels, more than 256
  choice options, and prompts over 8,192 tokens. The backbone sits behind the
  `JevForward` trait.
- `qwen35_quantized::QuantizedQwen35` runs the Qwen3.5 hybrid decoder over
  affine 8-bit triples with `quantized_matmul`, at any geometry, with an
  explicit causal mask. It is not the shared 4B MLX layer stack, which fixes
  2,560-wide dimensions. It matches the Candle CPU oracle (the shared
  `TextBackbone`) to about 1e-5 per layer on a tiny random model with three
  DeltaNet layers, one full-attention layer, and one more DeltaNet layer. The
  test runs MLX's CPU backend; the repo's Mac gates cover the same code.
- `mlx_engine::load` verifies the size and SHA-256 of every pinned file in
  place, builds the engine, and checks every loaded tensor's shape.
- `jev_mlx_smoke` and `scripts/jev-mlx-compare.py` run the engine and the
  mlx-vlm reference on the same requests (see below).

## What is not done

- **No run on the real weights.** The loader has never read the 30.7 GB
  checkpoint, and no forward has been compared with mlx-vlm. The model card's
  agreement claims (largest probability gap 0.0006) are upstream evidence, not
  OpenKind parity. Speed and memory are unmeasured: the DeltaNet recurrence
  advances one token at a time, activations are FP32, and each question
  re-reads its whole prompt.
- **Not registered.** There is no daemon alias, `openkind pull` entry,
  catalog manifest, or benchmark engine for JEV, and no golden fixture from a
  real forward.
- **The GEV backbone.** There is no Gemma 4 MoE port. The in-tree `gemma4`
  family is fail-closed to E4B GGUF geometry. The port needs the 128-expert
  top-8 router with its `per_expert_scale`, the parallel dense MLP,
  `attention_k_eq_v` on global layers, proportional partial RoPE, per-layer
  `layer_scalar`, and sliding-window masks.

## Verify JEV on a Mac

Operator steps only; tests never download. This needs Apple silicon, about
31 GB of free disk, and enough unified memory for the weights.

```bash
hf download nativ-community/JEV-27B-VL-MLX-8bit \
  --revision a871d5f8787b3d8ce9d618e7260e959393030c7b \
  --local-dir "$HOME/.cache/openkind/jev-27b-vl-mlx-8bit"
pip install "git+https://github.com/Lazarus-931/mlx-vlm.git@6ef5c0d13b847ef2a3c3586276af9c4b75da4686"
cargo run -p openkind-backends --release --features mlx --bin jev_mlx_smoke -- \
  --model-root "$HOME/.cache/openkind/jev-27b-vl-mlx-8bit" > rust.json
python3 scripts/jev-mlx-compare.py \
  --model-root "$HOME/.cache/openkind/jev-27b-vl-mlx-8bit" --rust rust.json
```

The script prints mlx-vlm's probabilities and the largest difference per
question. The reference computes in BF16 and this engine in FP32, so expect a
small gap. A gap near 0.1, or a different argmax, is a bug. The first load
hashes all 30.7 GB. If the numbers agree, the next steps are golden fixtures
from this run, a throughput and memory record, and daemon registration per
[`NEW_FAMILY.md`](NEW_FAMILY.md).

## Wire differences to resolve in an adapter

- Choice labels sort lexicographically in OpenKind's `unpack_question`, while
  the reference keeps the caller's dictionary order. Order changes what the
  model sees, so fixtures must use one order throughout.
- `unpack_question` fills a missing criterion with the label, so the readout
  treats a criterion equal to its key as absent. A caller who sends
  `"keep": "keep"` is rendered as `keep`, not `keep: keep`.
- Score needs exactly six levels; the wire accepts any count of two or more.
  Other counts must be rejected.
- Every native Choice question must include a non-empty `__none__` option and
  report its probability mass. Both models score it as an ordinary option.
- The reference accepts images in the state. The Jev wire here is text, so an
  adapter can skip the vision tower.
- Memory admission must account for 28 to 31 GB of resident weights. The
  JEV engine does not yet enforce an admission estimate.

## Reproduce the pins

Operator step only; tests never download. Hub file listings give shard sizes
and SHA-256 digests for LFS files, and `git`-tracked JSON files were hashed
after download.

```bash
hf download nativ-community/JEV-27B-VL-MLX-8bit \
  --revision a871d5f8787b3d8ce9d618e7260e959393030c7b \
  --local-dir "$HOME/.cache/openkind/jev-27b-vl-mlx-8bit"
hf download nativ-community/GEV-26B-Decide-MLX-8bit \
  --revision 8f04bf3f3204f067625a2716c54dd68de7200ec6 \
  --local-dir "$HOME/.cache/openkind/gev-26b-decide-mlx-8bit"
pip install "git+https://github.com/Lazarus-931/mlx-vlm.git@feat/jev"   # or @feat/gev
```
