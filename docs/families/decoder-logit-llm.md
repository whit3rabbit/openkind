# Family: decoder-logit-llm

> Decoder backbone (any chat/instruct GGUF), prompt with lettered options,
> label-logit readout via llama.cpp. The "common core" of the
> `llm-logits-v1` family in ollaya's nomenclature.

## Status in openkind

**Rust-loadable (prototype profile).** The pinned profile
`465963d705b6f35d6208` loads `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (q8_0) at
`9217f5db79a29953eb74d5343926648285ec7e67` (Apache-2.0) through the candle
quantized-runner binding, FP32 dequantization on CPU. It implements
`DecisionEngine` behind the bounded family scaffold in
[`families/decoder_logit_llm/`](../../crates/openkind-backends/src/families/decoder_logit_llm/mod.rs),
registers in `openkindd` via `--decoder-llm-aliases` /
`--decoder-llm-model-root`, and is benchmarked through
`openkind-bench --engine decoder-llm`.

**Binding decision.** The surveyed page left the binding open
(`llama-cpp-2` versus a thin `libllama` wrapper). This profile resolves the
open question by using candle's quantized GGUF runner: the GGUF checkpoint
format is honored, the build surface stays inside the existing Rust
dependency set, and the dequantization arithmetic is deterministic per
checkpoint. The profile carries its own fitted calibration temperature; the
q8_0 quantization shifts logits relative to the unquantized
[`decoder-logit-letter`](decoder-logit-letter.md) profile, so the two
profiles are not interchangeable.

Probability space: `ConditionalOnOfferedOptions`; the readout never samples
an output token; continuation state is KV cache only, cleared after every
question. Parity fixtures replay the shared letter-family calibration
workload within the frozen probability tolerance.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Any chat/instruct decoder GGUF; the family is model-agnostic above the contract layer |
| Tokenization | One prompt per question, identical layout to [`decoder-logit-letter.md`](decoder-logit-letter.md); label-letter set is the chat template's letter vocabulary |
| Forward pattern | Single forward pass per question via `llama-cpp-2`; next-token logits masked to letter tokens |
| Readout | Softmax over letter logits at the answer slot; identical to decoder-logit-letter |
| Continuation state | KV cache only; no DeltaNet or convolution state |
| Text generation | None — answer is the argmax over label logits |
| Calibration | Per-profile temperature fitted offline; the temperature is the only family-owned parameter |
| Semantic none | Same two-option `Noul` contract as decoder-logit-letter |

## Why this family is a candidate

- The family is **not** tied to a specific backbone. Any chat-tuned
  decoder that places answer-slot mass on the prompt's letter tokens
  can serve as a profile.
- The runtime backend is `llama.cpp`, which already ships CPU and Metal
  implementations; this would address the Apple Silicon acceleration
  gap left by the current encoder-state-first MLX path
  (see [`../../MLX.md`](../../MLX.md)).
- License variety is large. Permissively licensed chat/instruct models
  are widely available and would let `openkind` carry multiple model
  lines without depending on a single backbone.

## What blocks implementation

- `openkind` currently has no llama.cpp linking decisions. Adding the
  dependency changes the build surface across all supported platforms;
  the rules in [`../../crates/openkind-backends/AGENTS.md`](../../crates/openkind-backends/AGENTS.md)
  require an explicit profile + offline parity story for any new
  backend.
- llama.cpp's own numerics are not always bit-identical to the reference
  PyTorch. The parity tolerance, hidden-vector diagnostics, and
  acceptance criteria are owned by
  [`../../BENCHMARKS.md`](../../BENCHMARKS.md); the family would need a
  new tolerance study on each new backbone.
- BF16 widening is documented as a compatibility path that fails the
  frozen probability gate for the current Qwen 3.5 family. Decoder
  backends in this family have the same risk and need their own
  fixtures.
- The runtime surface is currently CPU-only on Apple Silicon for the
  implemented encoder family. The Metal story for llama.cpp-backed
  profiles would need its own verification.

## Open questions

- Which backbones would qualify as profiles under the M0 supported
  workload definition? The current milestone sequence in
  [`../../ROADMAP.md`](../../ROADMAP.md) has not selected any.
- Is `llama-cpp-2` the right binding, or does the family contract
  require a thin Rust wrapper over `libllama`? Both shapes are
  mentioned in the surveyed literature; neither has been prototyped
  inside `openkind`.
- Does the chat-template letter vocabulary differ across backbones,
  and how would the family contract document that variation?

## What this page does not say

No timings, no accuracy numbers, no calibration error. The cited
comparison values belong to ollaya's family page on `llm-logits-v1`, not
to `openkind`. `openkind` has no measurements to report.
