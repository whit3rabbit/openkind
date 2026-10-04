# encoder-embedding — frozen sentence encoder for the proxy-cache student

> Status: **Rust-loadable** (prototype profile). Loads local artifacts
> offline and serves through the daemon. It carries no model-quality claim:
> quality evidence lives in the proxy-cache disagreement budget
> ([`../PROXY_CACHE.md`](../PROXY_CACHE.md)).

## Architecture

`BAAI/bge-small-en-v1.5` is a BERT-base-shaped sentence encoder (12 layers,
384 hidden, 12 heads, 33M parameters) trained by BAAI with the
sentence-transformers recipe. OpenKind does not use its classification head;
the profile exports the encoder body only:

- tokenization: the pinned `tokenizer.json`, truncation at 256 tokens;
- pooling: **CLS** (position 0), the BGE convention;
- output: L2-normalized FP32 vectors, 384 dimensions;
- id: `candle-bert:BAAI/bge-small-en-v1.5:cls:256:<8-hex content hash>` on
  the candle CPU backend,
  `mlx-bert:...:cls:256:<hash>` on the MLX backend.

The embedding feeds the proxy-cache distilling student. Request states are
serialized canonically (strings pass through, objects and arrays become
sorted-key compact JSON) and truncated to 32,768 characters. The encoder id
is recorded on every training row, so changing it starts a new training
lineage for every task.

## Pinned profile

| Field | Value |
|---|---|
| Profile ID | `8d9498269ef05d95d93c` (derived from `family:encoder-embedding`, backbone `bge-small-en-v1.5`, revision below) |
| Pull name | `encoder-embedding:8d9498269ef05d95d93c` |
| Loader ID | `encoder-embedding` |
| Repository | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| License | MIT |
| Question types | choice (as the proxy-cache embedder; it answers no questions itself) |
| Artifacts | `model.safetensors` (133,466,304 bytes), `config.json`, `tokenizer.json` — digest-pinned in [`registry/v1/manifests/encoder-embedding-8d9498269ef05d95d93c.json`](../../registry/v1/manifests/encoder-embedding-8d9498269ef05d95d93c.json) |

## Backends

- **candle CPU (default)**: FP32 weights are memory-mapped in place; the
  forward runs through the hand-pinned `candle-transformers` BERT with an
  additive attention mask over padding. The embedding is a per-batch
  matmul plus 12 encoder layers — same arithmetic as the upstream PyTorch
  export in FP32.
- **MLX (feature `mlx`, macOS arm64)**: the same FP32 arithmetic expressed
  as MLX arrays under the process-wide serialized GPU stream
  ([`mlx_bert_encoder.rs`](../../crates/openkind-backends/src/proxy_cache/mlx_bert_encoder.rs)).
  The candle CPU path remains the correctness oracle.

Both backends implement the same `TextEmbedder` contract
([`encoder.rs`](../../crates/openkind-backends/src/proxy_cache/encoder.rs)):
`id()`, `dim()`, and `encode(texts) -> L2-normalized f32 vectors`.

The dependency-free `hash` embedder (`--proxy-cache-encoder hash`) is not
part of this profile: it needs no weights and exists for tests and for
running the proxy cache without a model download.

## Load

```bash
openkind pull encoder-embedding:8d9498269ef05d95d93c
openkindd --models mock --proxy-cache-upstream https://api.typesafe.ai \
  --proxy-cache-encoder encoder-embedding:8d9498269ef05d95d93c
```

The daemon verifies every artifact digest against the pinned manifest before
use. See [`../PROXY_CACHE.md`](../PROXY_CACHE.md) for the full flag surface
and the distillation lifecycle.

## Parity

Unit-level contract checks always run (shapes, normalization, encoder-id
stability). The golden-path replay
(`proxy_cache::bert_encoder::tests::embeds_text_with_pinned_checkpoint_when_env_gated`)
runs only when `OPENKIND_ENCODER_EMBEDDING_MODEL_ROOT` points at a verified
installation; tests never download model assets. The MLX replay compares three
checkpoint-backed embeddings with Candle CPU and passes with maximum absolute
element delta `2.980e-7` and minimum cosine `0.999999881`. It also covers the
Safetensors `__metadata__` header field.

The measured component benchmark and exact host/checkpoint details are in the
[BGE benchmark record](../benchmarks/2026-09-30-encoder-embedding/README.md).
This establishes numerical parity and local execution performance, not
embedding or proxy-cache quality.
