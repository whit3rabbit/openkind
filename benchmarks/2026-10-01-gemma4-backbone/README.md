# Gemma 4 backbone and winnow-e4b bring-up (2026-10-01)

First load of the `gemma4-decision` family: an in-tree Gemma 4 E4B text
backbone executing the pinned `EldanRing/Winnow-E4B` Q8_0 GGUF, with the
Winnow letter-logit decision protocol. This is a bring-up and
sanity record, **not** a task-quality or benchmark result: four
hand-written ground-truth questions and one chat probe were evaluated on
one machine, and no JevBench or M2 gate has run.

## What landed

- `crates/openkind-backends/src/families/gemma4/`: the Gemma 4 E4B
  backbone port (hybrid 5-sliding/1-full attention, proportional partial
  RoPE, per-layer embeddings, KV-shared trailing layers, softcap 30),
  the GGUF loader with a file-backed q8_0 row reader, the Winnow protocol
  renderer, and the `DecisionEngine` adapter.
- Profile `winnow-e4b:656ac636ce450cf79c7d` in `registry/v1` with the
  pinned digests; daemon flags (`--winnow-e4b-model-root`,
  `--winnow-e4b-aliases`), catalog install, and the `winnow-e4b` bench
  engine.

## Architecture sources

The port follows the three implementations that agree with each other —
HF `transformers` `modeling_gemma4.py`, llama.cpp `src/models/gemma4.cpp`
plus the `Gemma4Model` converter (`norm_shift = 0`), and mistral.rs
`vision_models/gemma4` — and deliberately diverges from candle `main`'s
`gemma4` example, which carries a Gemma-3-style `weight + 1` RMSNorm shift
and a `1/sqrt(head_dim)` attention scale that Gemma 4 does not use
(`Gemma4TextAttention.scaling = 1.0`).

## Bring-up findings

1. **K/V projections must consume the input-layernorm output.** The
   original port projected K/V from the raw residual while Q used the
   normed hidden; the result was a content-blind model (bitwise-identical
   readouts for prompts differing in one word). Projecting K/V from the
   normed hidden — as HF and llama.cpp do — fixed it.
2. **candle's quantized kernels dequantize q8_0 blocks to f16**, which is
   fine for the residual-stream paths but reshuffles the softmax argmax in
   Gemma 4's scale-1.0 attention. The attention projections therefore run
   as dequantized F32 linears (~2.9 GB); everything else runs `QMatMul`.
3. **GGUF q8_0 blocks are 34 bytes** (2-byte f16 scale + 32 int8), and the
   data section is `general.alignment`-aligned — both cost a debugging
   round before the row reader matched candle's dequantization exactly.
4. **The E4B turn boundary carries no thought channel.** The reference's
   template search for an immediately-closed `<|channel>thought` marker
   does not match this checkpoint's canonical Gemma 4 template (ollaya's
   12B converter pins the channel because the 12B template does match).
   Verified against `winnow-inference` at both the model-card pin
   (`77d1458`) and main.
5. **KV sharing ignores the checkpoint's dead K/V rows.** The Winnow GGUF
   carries K/V weights for layers 24..41 that differ from their donors';
   llama.cpp, mistral.rs, and HF all ignore them, so the loader skips
   them.

## Smoke evidence

`OPENKIND_GEMMA4_MODEL_ROOT=<dir> cargo test -p openkind-backends --release
--lib -- --nocapture --ignored gemma4::tests::pinned_checkpoint_smoke` on
`openkind-mac-arm64-local` (Apple M4 Max, 14 logical cores, 36 GB):

- checkpoint `sha256
  840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2`
  (matches the author's `SHA256SUMS`), tokenizer `sha256
  cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`;
  64 verified letter tokens cross-checked against the GGUF vocabulary.
- chat probe `The capital of France is` ranks `Paris` in the top two
  (llama.cpp with the same GGUF answers `Paris`; the pre-fix port answered
  generic prose openers).
- four ground-truth questions (state: an order to Reykjavik with one
  cancelled departure), one per wire primitive:

| question | read | calibrated probability |
|---|---|---:|
| noul: does the record still list an active departure? | true | 0.989 |
| noul: was the 09:30 departure cancelled? | true | 0.993 |
| choice: lisbon or reykjavik? | reykjavik | 0.999 |
| score: how many departures cancelled (0/1/2)? | 1 | 0.995 |

Temperature is the authors' published Q8 fit (`1.2574172017327816`),
not an openkind calibration.

- peak RSS 8.5 GB for the whole test process (weights ≈ the 7.46 GB
  on-disk artifact plus F32 attention copies and activations); whole run
  37 s for five forwards plus load.

## Reproduction

```bash
hf download EldanRing/Winnow-E4B gguf/Winnow-E4B-Q8_0.gguf \
    --revision 1b257e8fa80b270a62338362a8b35e37f7890273 --local-dir models/Winnow-E4B/gguf
hf download mistralrs-community/gemma-4-E4B-it-UQFF tokenizer.json \
    --revision a1789f4cb2d036e0e15c5cdb10317536e4a2d847 --local-dir models/Winnow-E4B
mv models/Winnow-E4B/gguf/Winnow-E4B-Q8_0.gguf models/Winnow-E4B/
OPENKIND_GEMMA4_MODEL_ROOT=models/Winnow-E4B cargo test -p openkind-backends --release --lib \
    -- --nocapture --ignored gemma4::tests::pinned_checkpoint_smoke
```

A measured `openkind-bench score --engine winnow-e4b` run on the seeded
shape777 workload is pending; the MODELS.md cells stay "measured run
pending" until it lands.
