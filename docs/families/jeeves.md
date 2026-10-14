# Family: jeeves

## Availability

Rust-loadable prototype for no-thinking decisions on macOS arm64 with
`--features mlx`. Profile `15fb3b95801f2f039636` loads the 4-bit
[cowWhySo/jeeves-mlx package](https://huggingface.co/cowWhySo/jeeves-mlx/tree/6f4220a267a94c5279b3fcbc5499de734b9a55b0)
in Rust. Catalog name `jeeves:15fb3b95801f2f039636` is installable; the
explicit artifact-root path registers `jeeves-native`. There is no native CPU
path. The external CUDA comparator remains separate.
Reviewed 2026-09-29 against
[source commit f04ec556](https://github.com/PostHog/jeeves/tree/f04ec5567301450dcaae0210dd54deeb4f647f87)
and [released weights 8622b7d1](https://huggingface.co/PostHog/jeeves/tree/8622b7d1652a9dcb8629486b84dce9e8d690c5cd).
The source is MIT; the weight repository declares Apache-2.0.

The native [loader and engine](../../crates/openkind-backends/src/families/jeeves/mod.rs)
keep affine 4-bit weights packed, widen BF16 vectors and quantization scales
to FP32, and execute the 4096-hidden, 12288-MLP hybrid backbone through the
shared serialized MLX runtime. The converted norms already include their
offset. Rust applies the exported FP32 pointer head and original temperature.
It neither loads Python nor generates reasoning tokens.

## Run the native engine

Build the CLI and daemon with MLX, then pull and serve the catalog profile
with `openkindd` on `PATH`:

```bash
SDKROOT="$(xcrun --show-sdk-path)" cargo build --release \
  -p openkind-cli -p openkind-server --features mlx
target/release/openkind pull jeeves:15fb3b95801f2f039636
PATH="$PWD/target/release:$PATH" target/release/openkind serve \
  --installed-models jeeves:15fb3b95801f2f039636
```

Requests to that daemon use the immutable catalog name as `model`.
The following artifact-root setup uses the shorter `jeeves-native` alias.

Acquire the pinned package explicitly into persistent storage. Builds and
tests never download assets; startup verifies the eight execution files'
sizes and SHA-256 digests in place.

```bash
JEEVES_ROOT="$HOME/.cache/openkind-comparators/jeeves-mlx/4bit-6f4220a267a9"
hf download cowWhySo/jeeves-mlx \
  --revision 6f4220a267a94c5279b3fcbc5499de734b9a55b0 \
  --local-dir "$JEEVES_ROOT"

SDKROOT="$(xcrun --show-sdk-path)" cargo build -p openkind-server --features mlx
target/debug/openkindd --http-addr 127.0.0.1:8009 --grpc-addr 0 \
  --models jeeves-native --jeeves-model-root "$JEEVES_ROOT"
```

Send an ordinary OpenKind request:

```bash
curl --fail-with-body http://127.0.0.1:8009/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"model":"jeeves-native","state":"The invoice was paid.","questions":{"paid":{"type":"noul","instructions":"Was the invoice paid?"}}}'
```

`--jeeves-aliases` changes the alias set. The matching environment variables
are `OPENKIND_JEEVES_MODEL_ROOT` and `OPENKIND_JEEVES_ALIASES`.
Shared `--family-concurrency`,
`--family-queue`, and `--family-timeout-ms` bound admission and deadlines.
Forward execution checks cancellation between layers.

Every question gets fresh KV, DeltaNet, and convolution state. Choice labels
sort lexicographically and must include a described `__none__` candidate.
Its probability is conditional on the supplied options, without an added
rejection head. Score levels retain caller order. Requests exceeding 64
questions, 255 candidates per question, or 4096 rendered tokens per question
fail without truncation.

Replay the four frozen reference rows against an already-downloaded model:

```bash
SDKROOT="$(xcrun --show-sdk-path)" cargo run -p openkind-backends \
  --features mlx --example jeeves_native_parity -- "$JEEVES_ROOT"
```

The [native evidence report](../verification/jeeves-native/2026-09-29/REPORT.md)
separates reference parity, HTTP behavior, and remaining qualification gates.
The [benchmark campaign](../benchmarks/2026-09-29-jeeves-native/REPORT.md)
records release-build request timing, process memory, and catalog installation.

## Download and run the reference

The helper pins both revisions and verifies each weight file's byte size and
SHA-256, including the pointer head, tokenizer, five fused backbone shards,
and block-4 drafter. Its checked-in
[artifact lock](../../scripts/jeeves-artifacts.json) covers 21,026,039,326 bytes
(about 19.6 GiB). Files stay in the operator's persistent directory. Download
is explicit; verification and serving use local files only.

Inspect the commands without downloading:

```bash
python3 scripts/jeeves.py plan --root "$HOME/.cache/openkind-comparators/jeeves"
```

Download on any host with Git and `huggingface_hub` installed:

```bash
python3 scripts/jeeves.py download --root "$HOME/.cache/openkind-comparators/jeeves"
python3 scripts/jeeves.py verify --root "$HOME/.cache/openkind-comparators/jeeves"
```

Serving requires Python 3.12 and a CUDA GPU, as specified upstream. Prepare a
separate environment and install the pinned source's dependencies:

```bash
python3.12 -m venv .venv-jeeves
source .venv-jeeves/bin/activate
pip install -r "$HOME/.cache/openkind-comparators/jeeves/source/requirements.txt"
python scripts/jeeves.py serve --root "$HOME/.cache/openkind-comparators/jeeves" \
  --port 8009 --max-think 768
```

The helper binds localhost and defaults to upstream BF16 (`--no-fp8`). Pass
`--fp8` to opt into the upstream kernels; this does not qualify another GPU
or toolchain. Startup still loads the drafter and performs a thinking warmup,
even if callers disable thinking. CUDA execution and dependency installation
have not been verified on the development Mac.

The external endpoint accepts Jeeves' own request contract:

```bash
curl --fail-with-body http://127.0.0.1:8009/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state":"The invoice was paid.","questions":{"paid":{"type":"noul","instructions":"Was the invoice paid?"}},"options":{"think":false}}'
```

Set `options.think` to `true` for the reasoning comparator. This endpoint is
independent of `openkindd`; its extra options and usage fields are not an
OpenKind wire extension.

## Inference pattern

Jeeves trains rank-16 LoRA and a learned pointer head over post-trained
`Qwen/Qwen3.5-9B`, rather than OpenKind's frozen 4B Base profile. The released
export merges LoRA into the backbone. Requests render text through the Qwen
chat template. The repository does not establish image decision support.

The [encoder](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/loader/dataloader.py)
places state, question and option markers before `<think>`. After reasoning,
it appends `</think>`, repeats the option block, and ends with `<decide>`.
The actual `remainder_text` does not repeat the question instruction. Without
thinking, the engine uses an empty chain. The
[head](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/model/head.py)
projects option-end and decide hidden states into 256 dimensions, computes
their dot product scaled by `1/16`, divides by the export's fitted temperature
(`1.8589280843734741`), and softmaxes over supplied options. The pinned export
declares renderer format `markers-v3-plainchains`.

The [engine](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/inference/engine.py)
shares state prefixes and isolates question branches across attention KV,
DeltaNet recurrent state, and convolution state. The diffusion drafter
accelerates autoregressive reasoning tokens. The server serializes requests
on one GPU. Choice supports up to 255 supplied labels, with probabilities
conditional on those labels. It supplies no separately learned semantic-none
mass corresponding to OpenKind's selected probability space.

## Remaining qualification gates

The native profile implements the empty-chain path. Generated reasoning and
the diffusion drafter stay in the external comparator under the repository's
no-generation invariant. Prefix sharing, a native CPU backend, other
quantizations are not implemented.

The four bundled examples establish a small reference-parity gate. Broader
source-aligned fixtures, reviewed task quality, semantic-none calibration,
matched cross-engine throughput and service soak remain
separate gates before release promotion.

## Evidence boundary

The [author's results](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/README.md#results)
report improved accuracy with reasoning and substantial tail latency.
The Kev/Jev comparisons outside JevBench use different items; they are not
paired wins. The 231-item public JevBench result excludes the sealed tier
and does not establish the current board's composite score. These are
upstream results, not OpenKind measurements.

Compare no-thinking and fixed chain caps on untouched document groups at
fixed question counts. Measure task accuracy, calibration, semantic-none
behavior, full-request latency and process memory separately. Acquisition
checks establish artifact identity, not inference parity or model quality.
