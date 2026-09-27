# Family survey: parallel constrained Qwen2.5 decisions

## Status in OpenKind

**Surveyed only.** The Hugging Face repository
[`harshatheg/Qwen-2.5-1B-RLCD`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/tree/2af86848be75847ccb3553b0941cc51d6ef7e4e9)
at `2af86848be75847ccb3553b0941cc51d6ef7e4e9` contains Python inference
code and presets, but no checkpoint, tokenizer, or model config. Its
[`core/engine_mlx.py`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/2af86848be75847ccb3553b0941cc51d6ef7e4e9/core/engine_mlx.py)
selects the separate
[`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit/tree/8b403126fc14f14cfc99bb4cfa72ecbc129ea677)
checkpoint at `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`. The repository
name's `1B` and `RLCD` labels are not evidence of a 1B fine-tune or of
TypeSafe's RLCD training procedure.

Neither repository is an installable OpenKind profile. The current Rust MLX
backend implements the pinned Qwen3.5 hybrid backbone only. It does not load
Qwen2.5's pure attention graph or the MLX-community 4-bit weight format. The
[`decoder-logit-letter`](./decoder-logit-letter.md) family loads a different
Qwen2.5-0.5B checkpoint on CPU with Candle; that loader does not establish
support for this checkpoint or MLX arithmetic.

## Inference pattern and fit

The Python path prefills one schema-first prompt, repeats its KV cache across
field lanes, right-pads field suffixes, evaluates them in one batched MLX
forward, and reads next-token logits at each field's last real suffix token.
It assembles the result in host code. That pattern overlaps OpenKind's existing
shared-root and batched continuation work, but its schema-first prompt and
token-logit readout are different from the selected state-first, candidate-
feature Qwen3.5 profile. The existing
[`candidate-pooling diagnostic`](../benchmarks/2026-09-27-candidate-pooling/README.md)
already groups Qwen3.5 candidate lanes across questions. Its pinned FP32 MLX
measurements did not support scheduler promotion.

The Python implementation's reported `sequential_forward_passes: 1` counts
the batched suffix forward but omits the prefill. On a first-token collision it
greedily continues for at most four tokens, then may choose the first option
and assign it a probability of at least `0.75`; this is neither an exact
candidate probability nor empirical calibration. A Rust port must instead
score the full offered options under an explicit probability space and fail
closed when it cannot distinguish them.

## Gate for a Rust MLX profile

1. Pin and verify the actual checkpoint, tokenizer, quantization config, and
   license as a new profile. Downloads remain explicit operator actions.
2. Implement Qwen2.5 4-bit MLX loading and complete KV branch isolation.
   Compare logits and decisions with an identified offline reference on
   equal and unequal suffix lengths.
3. Define a Jev mapping for `Noul`, `Choice`, `Score`, and `__none__`; add a
   `DecisionEngine` adapter, daemon alias, admission and cancellation checks,
   and offline parity fixtures before marking the profile Rust-loadable.
4. Measure matched full requests with and without field pooling in fresh
   processes. Report prefill, suffix, readout, forward calls, padded work,
   peak memory, selection parity, and probability differences. The source
   repository's autoregressive-JSON comparison is a different workload.

No MLX parity, OpenKind request-path benchmark, or quality result is claimed
for this surveyed family.
