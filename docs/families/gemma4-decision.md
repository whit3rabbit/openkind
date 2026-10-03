# Family: gemma4-decision (Rust-loadable)

> Gemma 4 text backbones fine-tuned for typed decisions, executed through
> the openkind Gemma 4 backbone port over pinned GGUF checkpoints. The
> founding profile is Winnow-E4B, the JevBench #8 Winnow lineage's
> E4B sibling.

## Status in openkind

**Rust-loadable prototype.** One pinned profile loads local artifacts
offline, passes offline contract checks, implements `DecisionEngine`, and is
registered by the daemon and `openkind-bench`. No M2 reviewed-decision gate
has run: task quality remains the checkpoint authors' claim, not an
openkind measurement.

| Profile | Checkpoint | Reference runtime | Calibration |
|---|---|---|---|
| `winnow-e4b` (`winnow:e4b`) | [EldanRing/Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B) Q8_0 GGUF @ `1b257e8` | winnow-inference @ `77d1458` (llama.cpp; MIT) | Q8 letter softmax `1.2574172017327816` (authors' 778-question fit) |

## Architecture

The Gemma 4 E4B text decoder, as implemented by HF transformers, llama.cpp,
and mistral.rs (the three agreeing references this port follows):

- 42 layers in a five-sliding/one-full repeating pattern; sliding layers
  use head dim 256 with full-dimension RoPE (base `1e4`), full-attention
  layers use head dim 512 with proportional partial RoPE (base `1e6`,
  rotary factor `0.25`).
- Attention softmax scale is the constant `1.0` — no `1/sqrt(head_dim)`;
  the learned Q/K RMSNorms carry the magnitude.
- RMSNorm applies the stored weight directly: Gemma 4 dropped the
  historical Gemma `weight + 1` shift (llama.cpp's converter pins
  `norm_shift = 0`).
- Values receive a pure, unweighted RMS norm.
- The E-series per-layer embeddings (PLE) inject a per-layer signal after
  each FFN block: a shared `(vocab, 42 x 256)` embedding table and a
  model-level projection combine with fixed scalars
  (`sqrt(256)`, `1/sqrt(2560)`, `1/sqrt(2)`), then each layer gates,
  multiplies, projects, and post-norms its slice.
- The trailing 18 KV-shared layers (24..41) have no K/V projections in the
  reference runtimes; they reuse the post-rope K/V of their same-type
  donors (layer 22 for sliding, 23 for full). The Winnow GGUF carries
  dead K/V rows for those layers; the loader skips them.
- Per-layer output scalars (`layer_output_scale`) rescale the residual
  stream; the tied LM head reuses the token embedding; final logits pass a
  `30.0` tanh softcap.

Where candle's own `gemma4` example (candle `main`) disagrees with the
three references — the `+1` norm shift and the `1/sqrt(head_dim)`
attention scale — the references win; both divergences are visible in the
HF source (`Gemma4RMSNorm` has no shift; `Gemma4TextAttention.scaling =
1.0`).

## Execution

CPU reference execution runs candle's quantized q8_0 `QMatMul` kernels —
the binding the `decoder-logit-llm` family established for GGUF
checkpoints — with two memory-bounded deviations on this 36 GB-class
measurement host:

- The two giant embedding tables (token embedding and the PLE table) are
  never fully resident: a file-backed row reader dequantizes only the
  rows a prompt references, arithmetic identical to candle's own q8_0
  block dequantization (`d * q` per 34-byte block).
- The attention projections run as dequantized F32 linears (~2.9 GB for
  the profile): candle's quantized kernels dequantize blocks to f16, and
  Gemma 4's scale-1.0 attention amplifies that f16 noise into a reshuffled
  softmax argmax, which destroys the readout. The residual-stream paths
  (FFN, PLE, LM head) tolerate the quantized kernels.

Every question is an independent full-sequence forward; no KV state is
retained across questions (numerically the reference's cold-prefix path).

Attention is banded, never dense: queries attend in fixed 256-token blocks
over only the keys the mask would admit — the causal prefix on the seven
full-attention layers, a `window + block` (512 + 256) key band on sliding
layers — with a compact block-local mask applied before the softmax. No
`heads x seq x seq` score tensor and no `seq x seq` mask is ever
materialized, so attention scratch grows linearly with prompt length: at
the frozen 8,192-token maximum the largest single score tensor is 64 MiB
(the final full-attention block), where a dense implementation would
allocate a 2 GiB score plus two 256 MiB masks per layer. Offline tests
assert the banded path matches the dense masked reference.

Admission is cost-aware, not only count-based. Loading derives the
effective accepted context length by capping the frozen 8,192-token
maximum at the largest sequence whose conservative linear scratch estimate
(banded attention blocks, expanded K/V, FFN intermediates, per-layer
inputs, retained donor K/V) fits the scratch budget — 4 GiB by default,
`OPENKIND_GEMMA4_SCRATCH_BUDGET_MB` to override. The forward re-checks
that estimate per question and fails closed before allocating anything,
and only the two donor layers (22 sliding, 23 full) retain their K/V for
the KV-shared tail instead of every layer's K/V.

Long forwards are cooperatively cancellable: the pass re-checks the
request control before the first layer, between every decoder layer, and
before the LM head, so a disconnect or an expired queue-inclusive deadline
releases the execution permit within one layer of the 42-layer pass
instead of only between questions.

Peak RSS in bring-up was 8.5 GB with the pinned Q8_0 checkpoint (short
prompts, dense attention).

## Readout

The Winnow letter protocol, reproduced from the reference
`winnow-inference` `native/protocol.h` and `native/engine.h`:

- Fixed system turn declaring the classification contract; state
  serialized as compact JSON with `<` escaped as `\u003c`; one
  letter-labelled option block per question; the model turn ends at
  `Answer:\n`, whose next-token logits are read.
- Letter labels `A`..`Z` then `AA`..`ZZ` (cap 64), each verified to
  tokenize to exactly one token that decodes back to the label; the
  discovered ids are cross-checked against the GGUF-embedded vocabulary at
  load and fail closed on drift.
- The pinned turn boundary has no thought channel: the reference's
  template search for an immediately-closed `<|channel>thought` marker
  does not match this checkpoint's canonical Gemma 4 template (the markers
  always surround `thinking_text`), so the compiled boundary is the plain
  `<turn|>\n<|turn>model\n`.
- Probability space `ConditionalOnOfferedOptions`; confidence is the
  normalized inverse entropy, matching the wire contract.

Declared determinism differences from the reference server: Choice
options render in canonical byte-lexicographic label order (the
repository `state_first` invariant), and a `Noul` question without caller
criteria renders the repository's default true/false descriptions rather
than the bare keys.

## Artifacts and verification

`openkind pull winnow-e4b:656ac636ce450cf79c7d` installs the pinned
`Winnow-E4B-Q8_0.gguf` (8,005,437,472 bytes,
`840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2`) and
the Gemma 4 E4B `tokenizer.json` (32,169,626 bytes,
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`,
from `mistralrs-community/gemma-4-E4B-it-UQFF` @ `a1789f4`; the GGUF
embeds the same vocabulary and the loader cross-checks the letter tokens).
The daemon serves flag-configured aliases with
`--winnow-e4b-model-root`/`--winnow-e4b-aliases`.

Offline tests cover the profile identity, pinned GGUF metadata contract,
letter-label discovery order, tensor-name mapping, KV-donor rule, banded
attention block masks and their equivalence to the dense masked reference,
the linear scratch estimate with its budget-derived context cap, and the
scratch budget override parsing. The operator-gated smoke test
(`OPENKIND_GEMMA4_MODEL_ROOT=<dir> cargo test -p openkind-backends --lib --
--ignored --nocapture gemma4::tests::pinned_checkpoint_smoke`) loads the
real checkpoint and asserts the argmax of four obvious ground-truth
questions across noul, choice, and score; bring-up passed all four at
0.99+ calibrated probability.

## What remains open

- Task qualification through the M2 gates is separate and has not run.
- The 12B sibling (`Winnow-12B`, JevBench #8) and the other surveyed
  Gemma 4 systems (Jev-Omni's safetensors export, the gated Cygnet and
  system-one-open) reuse this backbone once their own profiles pin them.
- An MLX backend for the Gemma 4 backbone would need its own parity
  qualification; the candle CUDA path follows the `cuda` feature like
  other families.

## What this page does not say

No accuracy or leaderboard claims beyond the checkpoint authors' own
published numbers; measured openkind results belong in
[`../BENCHMARKS.md`](../BENCHMARKS.md) once a qualified run lands.
