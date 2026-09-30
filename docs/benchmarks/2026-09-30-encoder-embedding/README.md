# BGE proxy-cache embedding benchmark

This is a component benchmark for registry profile
`encoder-embedding:8d9498269ef05d95d93c`. BGE produces proxy-cache state
embeddings; it is not a decision engine, so these numbers do not belong in the
777-decision request-path table.

## Identity and setup

| Field | Value |
|---|---|
| Checkpoint | `BAAI/bge-small-en-v1.5` |
| Source revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Weights SHA-256 | `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad` |
| Input SHA-256 | `086c9e1df940316a5935968950f614bab3f2f4f0a860e9029d5e0c9f63ee5b97` |
| Inputs | Three fixed support-ticket state strings; one text per call |
| Host | Apple M4 Max, 14 logical CPUs, 36 GiB, macOS 26.6.2, arm64 |
| Source | `5fe4f1b9ecbfc6cb9fa4bde4c4ca4cc67af4c015`, dirty working tree |
| Calls | 5 warmups and 30 timed samples per backend |

The timed path includes tokenization and one `TextEmbedder::encode` call; it
excludes model loading. Peak RSS below includes the process high-water mark,
including model loading and mapped weights. These runs were taken from a dirty
working tree and are local component evidence, not a release measurement.

## Results

| Backend | p50 | p95 | Embeddings/s | Peak RSS |
|---|---:|---:|---:|---:|
| Candle CPU FP32 | 26.06 ms | 29.49 ms | 40.52 | 273,891,328 bytes (261.2 MiB) |
| MLX FP32 | 4.30 ms | 5.69 ms | 220.36 | 273,907,712 bytes (261.2 MiB) |

On this host and input set, MLX measured about 5.4x the CPU throughput. The
sample is small and single-host; it does not establish performance on other
Apple silicon systems or under concurrent proxy-cache load.

## CPU/MLX parity

The checkpoint-gated parity test compared all three embeddings against Candle
CPU. It passed with maximum absolute element delta `2.980e-7` and minimum
cosine similarity `0.999999881`. The test acceptance bounds are max absolute
delta `1e-5` and minimum cosine `0.99999`.

```bash
SDKROOT="$(xcrun --show-sdk-path)" \
OPENKIND_ENCODER_EMBEDDING_MODEL_ROOT=/path/to/verified-local-checkpoint \
cargo test -p openkind-backends --features mlx --lib \
  proxy_cache::mlx_bert_encoder::tests -- --nocapture
```

The test also checks that the Safetensors reader skips free-form
`__metadata__` before decoding tensor entries. Tests use local model files and
do not download assets.

## Reproduction

Run each backend in a fresh process with the same pinned local checkpoint and
attribution values:

```bash
/usr/bin/time -l cargo run --release -p openkind-backends \
  --example encoder_embedding_bench -- \
  --model-root /path/to/verified-local-checkpoint --backend cpu \
  --warmup-calls 5 --sample-calls 30 \
  --host "Apple M4 Max, 14 logical CPUs, 36 GiB, macOS 26.6.2" \
  --commit 5fe4f1b9ecbfc6cb9fa4bde4c4ca4cc67af4c015 \
  --working-tree-dirty true

SDKROOT="$(xcrun --show-sdk-path)" /usr/bin/time -l \
cargo run --release -p openkind-backends --features mlx \
  --example encoder_embedding_bench -- \
  --model-root /path/to/verified-local-checkpoint --backend mlx \
  --warmup-calls 5 --sample-calls 30 \
  --host "Apple M4 Max, 14 logical CPUs, 36 GiB, macOS 26.6.2" \
  --commit 5fe4f1b9ecbfc6cb9fa4bde4c4ca4cc67af4c015 \
  --working-tree-dirty true
```

This benchmark and numerical parity do not measure embedding usefulness,
proxy-cache disagreement, student quality, or decision-task accuracy.
