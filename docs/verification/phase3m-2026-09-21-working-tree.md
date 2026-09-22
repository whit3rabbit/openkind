# Phase 3M working-tree recheck and MLX-community adapter: 21 September 2026

This is a dirty-working-tree real-checkpoint recheck. It is diagnostic
evidence for the current MLX implementation, not a commit-stamped landing or
release record. The historical first-pass reports in
[`phase3m-2026-09-20/`](phase3m-2026-09-20/) must not be used as the current
BF16 status because the current rerun found a BF16 continuation defect.

## Environment and inputs

- Host: named macOS arm64 development Mac.
- Xcode: `27.0` (`27A266a`).
- Metal toolchain: `32023.921`.
- MLX: core `0.32.2`, `mlx-rs 0.32.0`, vendored mlx-c
  `v0.6.0-7-gc74db53`.
- Pinned base model: `Qwen/Qwen3.5-4B-Base`, revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Base shard SHA-256:
  `model.safetensors-00001-of-00002.safetensors`:
  `df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712`;
  `model.safetensors-00002-of-00002.safetensors`:
  `590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef`.
- MLX-community comparison: `mlx-community/Qwen3.5-4B-MLX-bf16`, Hub revision
  `475632ded9a95863da4e4b235ab9ccbc5d3cc6bf`.
- Community artifact digests:
  `model.safetensors.index.json`:
  `0d718f2e2c433b1913fdeb7e4ebcdaab291e49137ec806980aecb1e88cf94d9d`;
  `model-00001-of-00002.safetensors`:
  `bbf84cddf80989b5a34317031a766816e4b5752468c498b60076cc4380b4aa9e`;
  `model-00002-of-00002.safetensors`:
  `b022545897033ebc19523a3a949e0ebe1837f3e76abe8a991aa2f71e4babe07a`;
  `tokenizer.json`:
  `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`.
- Phase 3B reference root:
  [`OpenKind_Phase3B_BackboneParity_20260920T152206Z/`](../../research/OpenKind_Phase3B_BackboneParity_20260920T152206Z/).
- Head bundle:
  [`qwen35_statefirst_a047d6802c3f06f085b8/`](../../crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8/).

The checkpoint was downloaded outside the repository. No repository files were
changed by the download or parity commands.

## Results

| Gate | Result | Evidence |
|---|---|---|
| 3M.0 runtime qualification | PASS | FP32 and BF16 primitive gates, including the issue-115 pipeline, passed. |
| FP32 embedding/layer trace | PASS | Embedding max error `0`; layer-00 max absolute error `3.22e-06`; final norm max absolute error `3.86e-05`. |
| FP32 full decision parity | PASS | 10 candidates, maximum probability delta `1.5148e-06`, zero argmax changes, zero policy changes. |
| FP32 nested continuation | PASS | Maximum probability delta `1.3787e-05`; root immutability, position, storage, and sibling isolation passed; cached/full delta `7.25e-05` within `1e-4`. |
| BF16 full decision parity | FAIL | Maximum probability delta `5.4572e-03` exceeded the frozen `0.005` tolerance; argmax and policy changes stayed at zero. |
| BF16 nested continuation | FAIL | Question continuation stopped at layer 4 with `invalid linear cache shape or dtype`. |
| MLX-community adapter load and execution | PASS | The adapter verified the community artifacts, normalized key/layout differences, loaded the model, and completed the full FP32 harness. This is format compatibility, not parity qualification. |
| MLX-community embedding comparison | FAIL | Maximum embedding error `8.5449e-04` against the pinned-base Phase 3B trace. |
| MLX-community full decision parity | FAIL | Maximum feature error `45.64`, probability error `0.9999983`, four argmax changes, and three policy changes. |
| MLX-community nested FP32 continuation | FAIL | Position, root storage, root immutability, and sibling isolation held; maximum probability error `0.9999982301`, cached/full feature delta `0.2339146631`, four argmax changes, and three policy changes. |
| MLX-community native BF16 decision parity | FAIL | The native BF16 run completed, but maximum feature error was `38.42`, probability error `0.99055`, with one argmax change and two policy changes. |

Load-inclusive timing from `/usr/bin/time -l` is bring-up evidence, not a
warm-process benchmark:

| Run | Wall | Load | Peak MLX allocation | Peak RSS |
|---|---:|---:|---:|---:|
| FP32 full parity | `44.7 s` | `4.7 s` | `16.10 GB` | `6.40 GB` |
| FP32 nested parity | `59.3 s` | included | not emitted | `9.85 GB` |
| BF16 full parity | `37.9 s` | `3.7 s` | `8.05 GB` | `8.35 GB` |
| MLX-community full FP32 parity attempt | `38.2 s` | `4.4 s` | `15.61 GB` | `8.70 GB` |
| MLX-community nested FP32 parity attempt | `69.69 s` | included | not emitted | `6.62 GB` |
| MLX-community full BF16 parity attempt | `55.3 s` | `3.7 s` | `7.81 GB` | `7.66 GB` |

The BF16 nested error occurs immediately after the first full-attention block,
at the following linear layer. The causal attention mask is currently built as
an FP32 array before it is added to BF16 scores. That is the leading suspect
for promotion of the following hidden/state path to FP32 while continuation
validation requires BF16 caches. This is an implementation diagnosis, not yet
a fix.

## MLX-community comparison

The closest unquantized community artifact was downloaded for comparison:
`mlx-community/Qwen3.5-4B-MLX-bf16`, Hub revision
`475632ded9a95863da4e4b235ab9ccbc5d3cc6bf`.
The [community model page](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-bf16/tree/main)
labels it as an MLX BF16 export.
Its model card identifies the source as `Qwen/Qwen3.5-4B`, converted through
an `mlx-vlm` Qwen3.5 fix branch, while the frozen OpenKind target is
`Qwen/Qwen3.5-4B-Base`. That source-model distinction is an important
qualification boundary before comparing decision probabilities.

Its config describes the same Qwen3.5 text architecture, and after mapping
the export prefixes the language-model and vision tensor sets correspond. It
is nevertheless not a drop-in artifact for this loader:

- text keys use `language_model.model.*`, not `model.language_model.*`;
- vision keys use `vision_tower.*`, not `model.visual.*`;
- the community index has 723 tensors versus 738 in the base index, with the
  15 omitted tensors belonging to the base checkpoint's `mtp.*` module;
- shards are `model-00001-of-00002.safetensors` and
  `model-00002-of-00002.safetensors`, not the pinned base names;
- `linear_attn.norm.weight` is BF16 in the community export, while it is FP32
  in the pinned base checkpoint. `A_log` remains FP32 in both;
- community safetensors headers include the standard `__metadata__` object;
- the community convolution weights use `[C, K, 1]`, while the pinned export
  uses `[C, 1, K]`. The singleton-axis layouts have the same contiguous values
  and the adapter accepts both.

The adapter now verifies those exact community artifacts, normalizes the text
and vision prefixes, skips the unused MTP difference, and records the
checkpoint identity as `mlx-community-qwen35-4b-bf16`. The real FP32 run
completed, but it is not a frozen-reference parity result: the community
embedding already differs by `8.5449e-04`, and the full run reached feature
error `45.64`, probability error `0.9999983`, four argmax changes, and three
policy changes. This is evidence that the downloaded community export is a
different source model and conversion, not merely a different file layout.
The 4-bit and 8-bit community exports are quantized profiles and were not
tested against this non-quantized adapter.

## Addendum (same day): `openkind-bench` MLX dispatch

The benchmark harness now dispatches the MLX backends. This is working-tree
evidence on top of the same dirty tree; nothing here changes the parity-gate
status above.

### Changes

- `Qwen35EngineConfig` selects the execution backend through a new
  `Qwen35Backend` enum (`NativeCpu`, `MlxFp32`, `MlxBf16`); the engine's
  request pipeline is unchanged and the CPU default is byte-for-byte the
  previous behavior (`backend_id` stays `qwen35-native-cpu`).
- An MLX backbone re-derives the scheduler's continuation-state size
  constants from the loaded model (BF16 states are half the FP32 bytes)
  and clamps backend capabilities to per-lane forward.
- `openkind-bench` gains an `mlx` cargo feature (forwarding to
  `openkind-backends/mlx`) and `--engine qwen35-mlx-fp32` /
  `qwen35-mlx-bf16` choices with distinct summary ids and output slugs.
- The daemon stays CPU-only; engine-side selection is in-process only
  (daemon alias registration remains 3M.7).

### Defect found and fixed: cross-thread MLX evaluation

The first bench run failed on the first request with
`There is no Stream(gpu, 0) in current thread` raised at the executor
boundary `eval`. Root cause: the engine evaluates on `spawn_blocking`
threads, but MLX graphs record the stream of the thread that created them
(mlx-c keeps per-thread stream/encoder registries), and load-time weight
transforms — the conv-weight reshape — left unevaluated lazy graphs
referencing the loading thread's stream. Worker threads cannot resolve that
stream's encoder.

Fix: weights now leave the loading thread fully materialized
(`MlxDecoderLayer::materialize` evaluates every weight array inside the
load-time `runtime.execute` closure; the final norm is evaluated too). This
extends the module's existing boundary discipline — continuation state was
already evaluated at executor boundaries — to weights. An offline regression
test (`materialized_arrays_evaluate_on_worker_threads`) pins the discipline
without a checkpoint. `MlxQwen35Backbone` also gained an `unsafe impl Sync`
with the same soundness argument as `MlxBackboneState` (all evaluation under
the process-wide execution mutex), which is what lets the engine share one
loaded backbone across blocking-pool threads.

### Results

Recorded in [`docs/benchmarks/2026-09-21-qwen35-mlx-smoke/`](../benchmarks/2026-09-21-qwen35-mlx-smoke/)
(pinned base) and
[`docs/benchmarks/2026-09-21-qwen35-mlx-community-smoke/`](../benchmarks/2026-09-21-qwen35-mlx-community-smoke/)
(community export): warm-process, untimed-warmup, single-sample-per-strategy
smoke cells on the named Mac, directly comparable to the CPU smoke record.
Headline: pinned-base FP32 shared-state strategies completed in ~8.3 s
versus ~70–74 s on CPU for the same fixture. The harness's exact
cross-strategy answer-equality flag is `false` for the MLX run: measured
maximum probability delta `1.68e-05`, confidence delta `1.46e-05`, zero
selection changes — the cached-versus-full numerical divergence already
measured by the 3M.4 gate, far inside the frozen `0.005` tolerance, but not
bit-identical as the CPU engine produces.

### Verification

- `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`
  (default), and `SDKROOT=$(xcrun --show-sdk-path) cargo clippy
  -p openkind-backends -p openkind-bench --features mlx --all-targets
  -- -D warnings` pass.
- `env -u RUST_LOG cargo test --workspace` (exit 0) and
  `cargo test -p openkind-backends --features mlx` (103 tests incl. the
  new regression test) and `cargo test -p openkind-bench --features mlx`
  pass.
- `cargo run -p openkind-gen-schemas -- --write` produced no schema
  changes (no wire types were touched); `git diff --check` is clean.
