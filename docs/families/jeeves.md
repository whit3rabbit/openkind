# Family: jeeves

## Availability

Surveyed, with a pinned download and launch helper for the external CUDA
reference. No OpenKind Rust loader, `openkind pull` entry, daemon alias, or
CPU/MLX qualification exists. Reviewed 2026-09-29 against
[source commit f04ec556](https://github.com/PostHog/jeeves/tree/f04ec5567301450dcaae0210dd54deeb4f647f87)
and [released weights 8622b7d1](https://huggingface.co/PostHog/jeeves/tree/8622b7d1652a9dcb8629486b84dce9e8d690c5cd).
The source is MIT; the weight repository declares Apache-2.0.

CUDA is a requirement of the upstream server and its kernels, not of the
Jeeves checkpoint or pointer-head math. A native CPU/MLX port is feasible in
principle. OpenKind's current Qwen3.5 executor pins the 4B dimensions (hidden
2560, MLP 9216); Jeeves needs hidden 4096 and MLP 12288. Sharing the family
name does not make the existing loader compatible. See the adaptation gates
below for the missing implementation and parity checks.

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

## Native adaptation gates

The no-thinking mode is the candidate for OpenKind integration. Thinking
conflicts with the repository's no-generation invariant, so this helper keeps
it in the external comparator. A native implementation needs:

1. A separate profile identity for the fused 9B config, tokenizer, exact
   chat-template renderer, marker sanitization, head and temperature.
2. A config-driven backbone compatibility check against the native Qwen3.5
   executor. The existing Kev loader is Qwen3-0.6B and cannot load these
   hybrid weights. Do not substitute the frozen 4B model or its readout.
3. A digest-pinned safetensors conversion of upstream `head.pt`, avoiding a
   pickle loader in Rust, and offline reference fixtures for the empty-chain
   forward, option-end vectors, logits and final distributions.
4. Request-bound Noul, Choice and Score mapping, including an explicit
   `__none__` policy and evidence for its semantics. Offering that label as
   another candidate does not establish calibrated rejection behavior.
5. Admission, cancellation and branch-isolation checks, then a concrete
   `DecisionEngine` adapter and daemon registration. Only then add an
   installable catalog manifest and update [the model index](../MODELS.md).

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
