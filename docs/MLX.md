# MLX backend

> Implementation guide for the optional Qwen 3.5 MLX/Metal parity backend.
> Benchmark methodology and recorded numbers remain canonical in
> [`BENCHMARKS.md`](BENCHMARKS.md).

## Status

Standard macOS arm64 releases include this backend and its colocated
`mlx.metallib`. Source builds enable it with `--features mlx`. Daemon
selectors default to `auto`, preferring MLX FP32 for supported loaders
after an isolated runtime probe. Explicit `native-cpu` and `mlx-fp32`
overrides remain available. See [backend releases](BACKENDS.md). It executes
the pinned `Qwen/Qwen3.5-4B-Base` profile through MLX 0.32.2 while preserving
the same renderer, continuation-state, readout, calibration, and policy
contracts as the Candle CPU oracle.

The current production configuration is deliberately conservative:

- FP32 with `ReferenceOps` is the default and passes the frozen full and
  sequential-nested parity gates on the named M4 Max development host.
- GPU work is serialized process-wide on one explicit cross-thread stream.
- A forced `nested_batched` FP32 `ReferenceOps` run can use a real vectorized
  forward for 2–8 equal-start-position lanes, with right padding and each
  lane's true suffix length. The pinned-base variable-length model gate now
  passes for unequal question and candidate lengths. Automatic scheduling
  remains per-lane. The measured candidate-pooling and flat-field diagnostics
  did not justify changing that choice.
- The packed FP32 Metal reduction-tree kernel passes parity but remains opt-in
  because it is slower than `ReferenceOps` on the current smoke workload.
- Native BF16 and the downloaded MLX-community checkpoint are unpromoted
  compatibility or research paths. Neither inherits the FP32 parity claim.
- The daemon exposes `--qwen35-backend mlx-fp32` behind its `mlx` feature.
  A forced vectorized two-question request returned HTTP 200 and logged the
  expected physical forwarding mode. Bounded unified-memory admission and
  recovery runs are recorded. Service load/soak and a clean-commit promotion
  record remain open.

## External MLX decision profile

[`Jev-Style-2B-Decision-v3-MLX`](./families/jev-style.md) is a separate
surveyed profile at Hub revision
`b86e4cbc6f420f5d7d1691299d62db272934ebf4`. It includes BF16 and affine 8-bit
weights plus a custom Python scoring runtime pinned to `mlx-lm==0.31.3`. That
runtime implements block-causal attention, a yes/no logit-difference readout,
and exact FP32 norm sidecar weights. The OpenKind Qwen3.5-4B loader does not
load this 2B profile. Both upstream precision folders were hash-verified and
ran a local decision smoke on the M4 Max; same-host timing comparisons are in
[`BENCHMARKS.md`](BENCHMARKS.md#jev-style-2b-mlx-survey).

## Encoder family backend: laya (2026-09-28)

The laya decision-encoder family has a second execution backend:
`families/laya/mlx/` runs the same digest-locked pinned checkpoints through
mlx-rs FP32 arrays mirroring the candle encoder arithmetic (weight-only
norms, per-layer-type RoPE, sliding-window bands, gated GELU, pre-norm head).
It is the first MLX path for an encoder family and follows the same
discipline as the Qwen3.5 backend: one process-wide serialized stream via
`MlxRuntime::execute`, weights verified in place and read without temporary
copies, and the candle CPU path as the correctness oracle.

Parity gates are frozen in `tests/laya_parity.rs` (module `mlx_replay`):
golden-fixture replay per profile with zero selection flips and a 0.005
probability-drift budget; measured drift is 2.5–7.2e-6 across the three
profiles. Unlike the Qwen3.5 continuation backend there is no recurrent
state, so the qualification surface is the single-forward contract. BF16
execution is not offered for this family. Evidence and request-path
benchmarks live in
[`benchmarks/2026-09-28-laya-mlx-campaign/`](benchmarks/2026-09-28-laya-mlx-campaign/README.md).

## Survey-family MLX backends: gliclass and jevk5 (2026-09-29)

Two surveyed families gained MLX execution backends behind their frozen
family contracts, both following the laya discipline (candle CPU oracle,
golden-fixture replay gates, one process-wide serialized stream):

- **`encoder-instruct-label/mlx-fp32`** — the GLiClass uni-encoder. The
  ModernBERT body moved into a shared implementation
  ([`families/mlx_modernbert.rs`](../crates/openkind-backends/src/families/mlx_modernbert.rs))
  extracted from the laya backend, so every ModernBERT-shaped encoder family
  executes one numerical path; the family module adds only the projector
  pair and dot-product scorer. Parity: max probability drift `4.487e-6`,
  zero selection flips over the 15 golden answers.
- **`decoder-logit-qwen35/mlx-fp32`** — the JevK5 letter-logit decoder runs
  the pinned single-file BF16 checkpoint through the parity-verified
  Qwen3.5 MLX backbone via a survey-checkpoint descriptor
  ([`MlxSurveyCheckpoint`](../crates/openkind-backends/src/qwen35/mlx/weights/checkpoint.rs)):
  the family verifies its own pinned digests, the backbone streams the shard
  in place and widens BF16 to FP32 exactly, and the letter readout dots the
  final-token feature vector with host-read tied-embedding rows — the same
  arithmetic as the CPU family code. Parity: max probability drift
  `1.003e-6`, zero selection flips over the 9 golden answers.

The same decoder backend also serves the pinned Plumb-4B profile as
`plumb-4b/mlx-fp32`. It retains Plumb's tokenizer, single-read limit, and
per-type calibration. Select `plumb-4b-mlx-fp32` in the benchmark harness,
or configure Plumb aliases and model root with the daemon's shared
`--decoder-logit-qwen35-backend mlx-fp32` selector. The
[family page](families/decoder-logit-qwen35.md#checkpoint-replay)
documents checkpoint-gated CPU and MLX replays. The
[Plumb/Decider record](benchmarks/2026-09-30-jevbench-expansion/README.md)
contains CPU timing and a reported Plumb MLX replay, with its evidence gaps
identified. No Plumb MLX throughput record is archived there. Decider has
no MLX adapter yet.

Backend selection: `openkindd --encoder-instruct-backend mlx-fp32` /
`--decoder-logit-qwen35-backend mlx-fp32`, and
`openkind-bench --engine encoder-instruct-label-mlx-fp32` /
`decoder-logit-qwen35-mlx-fp32`. The conversion leads below cover catalog
profiles that still lack a qualified MLX result. A Hub conversion, even when
it uses the same backbone, does not by itself provide an OpenKind execution
path.

The proxy-cache BGE sentence encoder also has a macOS arm64 MLX FP32 path. Its
checkpoint-backed CPU/MLX replay passed with max element delta `2.980e-7` and
minimum cosine `0.999999881`; its single-host component timings are recorded
in [`benchmarks/2026-09-30-encoder-embedding/`](benchmarks/2026-09-30-encoder-embedding/README.md).
This profile embeds proxy-cache state and is not a `DecisionEngine`.

## Survey-family MLX coverage expansion (2026-11-08)

The surveyed decision profiles below have MLX paths behind their
existing family contracts. The implementations reuse several shared MLX
backbones:

- The pinned Qwen3.5 hybrid backbone with the six fused Metal kernels
  (`decider-4b` reuses it directly through the survey-checkpoint descriptor;
  `sieve-4b` additionally applies the PEFT `base + (B @ A) · alpha/r` merge
  host-side in FP32 before the values reach MLX).
- `families/qwen35_survey_mlx.rs` — a geometry-parameterized causal hybrid
  forward for non-4B widths, used by `strands-decider-2b` (Qwen3.5-2B +
  adapter + pointer head) and `drex-v15` (9B hybrid + pickle pointer head).
- `families/qwen_dense_mlx.rs` — a dense Qwen3/Qwen2 decoder body (per-head
  q/k RMSNorm flag, full-dimension NeoX RoPE, HF `repeat_kv` interleave,
  tied-embedding readout, optional PEFT merge): `kev`,
  `decoder-logit-qwen3-06b/17b/4b`, `qwen3guard`, `decoder-logit-letter`.
  `winnow` (lowercase-key LoRA) and `decoder-logit-llm` (GGUF q8_0
  dequantized to FP32 at load) carry the same forward in their own modules
  because their artifact containers differ.
- `families/mlx_modernbert.rs` / `families/bert_mlx.rs` — encoder bodies:
  `von` (ModernBERT-large over the author's torch pickle, widened to FP32
  exactly), `encoder-nli` (DistilBERT), and `schema-scorer` (MiniLM BERT
  cross-encoder). `gemma4-decision` runs a dedicated FP32-dequantized GGUF
  port (`gemma4-decision/mlx-fp32-deq`) mirroring the scale-1.0 attention,
  partial RoPE, KV sharing, PLE, and softcapping numerics of the CPU oracle.

Backend identifiers and arithmetic identities (all FP32 on the Metal GPU
through the process-wide serialized stream):

| Family | Backend id | Arithmetic id | Loader |
|---|---|---|---|
| decider-4b | `decider-4b/mlx-fp32` | `mlx-gpu-fp32-decider` | `DeciderMlxEngine::load` |
| sieve-4b | `sieve-4b/mlx-fp32` | `mlx-gpu-fp32-sieve` | `SieveMlxEngine::load` |
| strands-decider-2b | `strands-decider-2b/mlx-fp32` | `mlx-gpu-fp32-strands` | `StrandsMlxEngine::load` |
| drex-v15 | `drex-v15/mlx-fp32` | `mlx-gpu-fp32-drex` | `DrexMlxEngine::load` |
| von | `von/mlx-fp32` | `mlx-gpu-fp32-von` | `VonEngine::load_mlx` |
| kev | `kev/mlx-fp32` | `mlx-gpu-fp32-kev` | `KevMlxEngine::load` |
| qwen3guard | `qwen3guard/mlx-fp32` | `mlx-gpu-fp32-qwen3guard` | `Qwen3GuardEngine::load_mlx` |
| decoder-logit-qwen3-06b | `decoder-logit-qwen3-06b/mlx-fp32` | `mlx-gpu-fp32-decoder-logit-qwen3` | `DecoderLogitQwen3Engine::load_mlx` |
| decoder-logit-qwen3-17b | `decoder-logit-qwen3-17b/mlx-fp32` | `mlx-gpu-fp32-decoder-logit-qwen3` | `DecoderLogitQwen3Engine::load_mlx` |
| decoder-logit-qwen3-4b | `decoder-logit-qwen3-4b/mlx-fp32` | `mlx-gpu-fp32-decoder-logit-qwen3` | `DecoderLogitQwen3Engine::load_mlx` |
| decoder-logit-letter | `decoder-logit-letter/mlx-fp32` | `mlx-gpu-fp32-decoder-logit-letter` | `DecoderLetterMlxEngine::load` |
| decoder-logit-llm | `decoder-logit-llm/mlx-fp32` | `mlx-gpu-fp32-decoder-logit-llm` | `DecoderLlmMlxEngine::load` |
| winnow | `winnow/mlx-fp32` | `mlx-gpu-fp32-winnow` | `WinnowEngine::load_mlx` |
| encoder-nli | `encoder-nli/mlx-fp32` | `mlx-gpu-fp32-encoder-nli` | `EncoderNliMlxEngine::load` |
| schema-scorer | `schema-scorer/mlx-fp32` | `mlx-gpu-fp32-schema-scorer` | `SchemaScorerMlxEngine::load` |
| gemma4-decision | `gemma4-decision/mlx-fp32-deq` | `mlx-gpu-fp32-deq-gemma4` | `Gemma4MlxEngine::load_mlx` |

On macOS arm64, the daemon's `auto` setting prefers MLX for these profiles
when the `mlx` feature is built and the runtime is ready. Use
`--<family>-backend native-cpu` to select the CPU reference. For benchmark
runs, use the family engine listed by
`openkind-bench score --help`; Gemma 4 uses
`--engine gemma4-decision-mlx-fp32-deq`.

Evidence gates: every backend carries an env-gated replay
(`tests/<family>_parity.rs`, `mod mlx_replay`) that compares the MLX engine
against the committed fixture or the CPU oracle with zero selection flips and
a 0.005 probability budget for probability-bearing answers. None of these
backends has a recorded checkpoint-backed parity run yet: the gates skip
without the pinned checkpoints, and a promoted parity claim still requires a
replay record per the [benchmark guide](BENCHMARKS.md). Load and replay them
by pointing the documented env vars at the pinned artifact roots on a macOS
arm64 host.

## Hugging Face MLX conversion survey

The catalog has 18 profiles without a qualified MLX result. The supplemental
[`registry/v1/mlx-alternatives.json`](../registry/v1/mlx-alternatives.json)
records every catalog profile's current MLX status and pins each discovered
Hugging Face lead to an immutable revision. These leads are research metadata,
not pullable model entries. Most conversions cover only a backbone; a matching
model size leaves profile fine-tuning, readout head, renderer, calibration,
and artifact format unverified. A new executable `registry/v1` profile needs
a Rust loader for its pinned artifacts and the offline parity and
daemon-registration gates in
[`families/NEW_FAMILY.md`](families/NEW_FAMILY.md).

| OpenKind profile(s) | Hugging Face MLX lead | Identity and integration boundary |
|---|---|---|
| `decoder-logit-qwen3-06b`, `decoder-logit-qwen3-17b`, `decoder-logit-qwen3-4b` | [`Qwen3-0.6B-4bit`](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit), [`Qwen3-1.7B-bf16`](https://huggingface.co/mlx-community/Qwen3-1.7B-bf16), [`Qwen3-4B-Instruct-2507-bf16`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-bf16), [`Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) | Same upstream checkpoints. OpenKind's FP32 paths load the pinned original weights. These conversions need separate loader and parity evidence. |
| `kev` | [`Qwen3-0.6B-Base-bf16`](https://huggingface.co/mlx-community/Qwen3-0.6B-Base-bf16) | Base-backbone lead only. Kev's LoRA and pointer head are separate profile artifacts. |
| `qwen3guard` | [`Qwen3Guard-Stream-0.6B-mxfp4-mlx`](https://huggingface.co/abnormalmapstudio/Qwen3Guard-Stream-0.6B-mxfp4-mlx/tree/7d27a40cfe7800748c0d68b9ed56c02a6017b81a) | Conversion of the exact upstream profile checkpoint. The Hub card uses generic text generation; token-level classifier semantics and parity remain unverified in OpenKind. |
| `decoder-logit-letter`, `decoder-logit-llm`, `winnow` | [`Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit) | Shared Qwen2.5 backbone conversion. OpenKind's MLX paths use the pinned safetensors or q8_0 GGUF, and Winnow merges its pinned LoRA at load. |
| `plumb-4b` | No additional conversion needed | The profile already has a `plumb-4b/mlx-fp32` execution path through the Qwen3.5 MLX backend. Its reported replay lacks archived evidence for a qualified result; see the [Plumb and Decider record](benchmarks/2026-09-30-jevbench-expansion/README.md). |
| `decider-4b` | [`Qwen3.5-4B-MLX-bf16`](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-bf16) | Related Qwen3.5 conversion. The MLX path uses Mapika's pinned merged Decider checkpoint. |
| `strands-decider-2b` | [`Qwen3.5-2B-MLX-bf16`](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-bf16), [`Jev-Style-2B-Decision-v3-MLX`](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX) | The Qwen conversion is a base-model lead. Jev-Style is an independent task-trained decision model with its own MLX runtime, readout, and calibration, not a Strands conversion or an OpenKind parity result. |
| `drex-v15` | None found (checked 2026-11-07) | Nace.AI ships only a Q8_0 GGUF of this checkpoint, and an affine-Q4 MLX checkpoint for Drex Max. The license counts conversions as derivatives, so OpenKind mirrors none. |
| `clef-flash`, `clef-flash-gguf`, `clef-27b-gguf` | [`clef-flash-8bit`](https://huggingface.co/mlx-community/clef-flash-8bit), [`clef-8bit`](https://huggingface.co/mlx-community/clef-8bit) | Exact Cloudflare checkpoints with a separate Python scoring runtime. Its 8-bit spot-check is not an OpenKind parity result; the in-tree 4-bit path remains unqualified. |
| `winnow-e4b` | [`gemma-4-e4b-it-bf16`](https://huggingface.co/mlx-community/gemma-4-e4b-it-bf16) | Gemma 4 base-family conversion. The Winnow fine-tune and letter-logit profile require separate weights and integration. |
| `von` | [`answerdotai-ModernBERT-Large-Instruct-bf16`](https://huggingface.co/mlx-community/answerdotai-ModernBERT-Large-Instruct-bf16/tree/3a31b85359b2419b39e8addd4db696218a82c0f9) (also 8-bit, 6-bit, and 4-bit) and [`tasksource-ModernBERT-base-embed-bf16`](https://huggingface.co/mlx-community/tasksource-ModernBERT-base-embed-bf16/tree/d1ba4cc2280fe323abd82d731d2e178d60313175) | The Instruct conversions are separate masked-language-model classifiers, not Von's fine-tuned option-marker checkpoint or an OpenKind parity result. The embedding conversion has a different task and no declared license. |
| `encoder-nli` | [`splade-v3-distilbert-mlx`](https://huggingface.co/NomaDamas/splade-v3-distilbert-mlx/tree/b639afadb020ad5041c9fc78495dd8c0fea656f1) | Related DistilBERT sparse-retrieval encoder under CC BY-NC-SA 4.0, not the Typeform MNLI fine-tune or its sequence-classification head. No matching MLX conversion of [`typeform/distilbert-base-uncased-mnli`](https://huggingface.co/typeform/distilbert-base-uncased-mnli) was found. |
| `schema-scorer` | [`all-MiniLM-L6-v2-bf16`](https://huggingface.co/mlx-community/all-MiniLM-L6-v2-bf16) | Related MiniLM embedding model, not a sentence-pair cross-encoder. OpenKind uses the MS MARCO MiniLM cross-encoder and its single-logit readout. |

Hub repositories and metadata were checked on 2026-10-03. Their commit SHAs,
reported licenses, and fit assessments are pinned in the supplemental index.
Recheck those details and qualify the model before using a lead in OpenKind.

### Jeeves packed-weight profile

[`Jeeves`](families/jeeves.md) has a separate native Rust loader for the
pinned affine4/g64 Qwen3.5-9B conversion. It uses FP32 arithmetic and the
exported pointer head through the same serialized runtime. The no-thinking
path supports `jeeves-native` with `--jeeves-model-root` and installed
`jeeves:15fb3b95801f2f039636`; it has no CPU backend.
Its four-row reference parity and HTTP
smoke are recorded in the [native evidence report](verification/jeeves-native/2026-09-29/REPORT.md).
Request-path measurements use `--engine jeeves-mlx-affine4` and are recorded
in the [benchmark campaign](benchmarks/2026-09-29-jeeves-native/REPORT.md),
without reviewed task-quality or release-promotion evidence.

## Pinned runtime and model identity

| Component | Identity |
|---|---|
| Rust binding | `mlx-rs 0.32.0` |
| Vendored mlx-c | `v0.6.0-7-gc74db53` |
| MLX core | `0.32.2` |
| Backbone | `Qwen/Qwen3.5-4B-Base` |
| Backbone revision | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| Profile | `a047d6802c3f06f085b8` |
| Production arithmetic family | `mlx-core-0.32.2/fp32/metal-simd-sequence-dk128-v1` |
| Baseline reference arithmetic family | `mlx-core-0.32.2/fp32/reference-ops` |
| FP32 kernel candidate | `mlx-core-0.32.2/fp32/metal-tree-packed-dk128-v1` |
| Default inactive cache limit | 256 MiB |

The complete arithmetic identity also includes checkpoint format and the
Xcode/Metal toolchain identity. Rebuilding mlx-c under another toolchain is a
new runtime identity and requires runtime qualification plus the model gates.

## Execution flow

```text
Qwen35DecisionEngine
  -> Qwen35Backend::MlxFp32 or MlxBf16
  -> MlxRuntime and verified checkpoint load
  -> host-resident token embedding rows
  -> 24 DeltaNet layers + 8 grouped-query attention layers
  -> final RMSNorm, final token only
  -> score-summary head, calibration, rejection, and typed answer
```

The backend is an execution replacement, not a different decision model.
[`engine/backbone.rs`](../crates/openkind-backends/src/qwen35/engine/backbone.rs)
selects MLX behind the same backend-neutral executor contract used by Candle.
[`model/mod.rs`](../crates/openkind-backends/src/qwen35/mlx/model/mod.rs) owns load,
prefill, continuation, final normalization, and executor-boundary
materialization.

Production forward returns only the final token's FP32 feature vector to the
host. The 34-stage host trace is isolated in `prefill_trace` so ordinary
serving and benchmarks do not pay for 32 layer readbacks.

## Stream and threading contract

MLX streams and lazy graphs are normally thread-sensitive. The engine can
load on one thread and execute on a blocking-pool worker, so relying on each
caller's default stream is unsafe.

[`MlxRuntime`](../crates/openkind-backends/src/qwen35/mlx/runtime.rs)
therefore enforces one process-wide contract:

1. Construct one GPU stream with `mlx_stream_new_thread_unsafe`.
2. Guard that stream with one process-wide mutex.
3. Enter `mlx_rs::with_stream` for every graph construction and evaluation
   scope.
4. Use the same explicit handle for synchronization, and serialize allocator
   telemetry under the same process-wide lock.
5. Fully materialize weights at load and continuation tensors before they
   leave an executor boundary.
6. Convert a panic inside `execute` into `MlxError` while releasing the lock
   normally.

Backend code must not call MLX operations outside `MlxRuntime::execute`.
Custom kernels fail closed when no runtime-scoped stream is installed. More
than one `MlxRuntime` may exist, but every instance shares the same execution
lock, GPU stream, and process-wide allocator-cache setting.

This intentionally permits only one active MLX model operation per process.
It is the correct first-parity design. Concurrency should be revisited only
with an explicit worker or stream ownership model and evidence that parallel
execution does not break state isolation, memory admission, or numerical
repeatability.

## Checkpoint loading and memory

[`weights/`](../crates/openkind-backends/src/qwen35/mlx/weights/) verifies
checkpoint configuration, tokenizer, index, shard length, and shard digests
before execution. Multi-gigabyte shards remain in their read-only checkpoint
directory and are read in place. They are never copied to temporary storage.

The loader:

- keeps the tied embedding table host-resident and reads only required token
  rows;
- filters vision, MTP, embedding, and output-head payloads before allocating
  decoder arrays;
- streams one required tensor at a time so host staging buffers drop between
  tensors;
- widens checkpoint BF16 exactly for FP32 execution;
- decodes native BF16 directly when the BF16 candidate is requested;
- folds offset layer-normalization weights where required, while leaving the
  DeltaNet normalization weight raw;
- records MLX peak allocation, process RSS high-water mark, and load time.

MLX `Array` cloning copies an immutable reference-counted handle. Branch
isolation still covers all Qwen 3.5 state families: attention KV, DeltaNet
recurrent matrices, convolution windows, logical position, and execution
identity. Tensor-payload accounting is separate from MLX allocator cache,
mapped weights, graph scratch, and process RSS. Admission logic uses
saturating arithmetic and must retain a separately measured process envelope.

## Continuation and batching

[`branch_state.rs`](../crates/openkind-backends/src/qwen35/mlx/branch_state.rs)
implements `BranchableState` and `BranchBatch`. A root state can be forked,
selected, and gathered without mutating the original. Strict fingerprints and
tests cover root immutability and sibling isolation.

`fork_batch` creates isolated lane states. For FP32 `ReferenceOps`, the
explicitly forced `nested_batched` runner advances 2–8 lanes through a real
vectorized suffix forward. It right-pads variable lengths and restores each
lane's true position and state. The pinned fixture pair covers unequal
question and candidate lengths. Automatic scheduling still advertises
per-lane execution. The candidate-pooling and flat-field comparisons did not
meet their promotion threshold.
Consequently:

- `nested_sequential` remains the automatic shared-prefix plan;
- forcing `nested_batched` exercises the vectorized FP32 path for diagnostics;
- parity does not establish that vectorized execution is faster;
- the [candidate-pooling diagnostic](benchmarks/2026-09-27-candidate-pooling/)
  found no latency win over current batching at Q2/K2 or Q8/K4;
- the [flat-field diagnostic](benchmarks/2026-09-27-python-flat-field/)
  ports the Python shared-root field schedule to Qwen3.5 with its unchanged
  readout. Paired native compute runs find it 6% to 12% slower at Q2/K2 and
  about 64% slower at Q8/K4 despite fewer physical forwards.

An opt-in [candidate pooling diagnostic](BENCHMARKS.md#candidate-pooling-diagnostic)
can combine candidate lanes from different questions when their continuation
positions match. It preserves question/candidate result order and falls back
to one-lane execution where pooling is impossible. The service and automatic
scheduler do not select this graph pending matched performance and parity
evidence.

[`run_flat_batched_candidates`](../crates/openkind-backends/src/qwen35/backbone/batched/flat.rs)
is another opt-in diagnostic. It concatenates each question and candidate
suffix into one lane from the state root and batches up to eight lanes. This
repeats question tokens per candidate; the pinned paired record above rejects
it for the tested shapes. It is absent from daemon dispatch and automatic
scheduling. The measurements exclude rendering, tokenization, validation,
transport and queueing, so they are not full-request throughput evidence.

## Gated DeltaNet implementations

The linear-attention recurrence has two implementations:

### `ReferenceOps`

The production default uses ordinary MLX array operations for the recurrent
update. It processes suffix tokens sequentially and remains the independent
differential comparator for any fused kernel.

For FP32 the row-independent work no longer runs once per token. The causal
convolution window, the q/k/v split and L2 norms, the beta and gate math, and
the output RMSNorm and SiLU gate read no recurrent state, so
[`forward_hoisted`](../crates/openkind-backends/src/qwen35/mlx/layers/linear_attention.rs)
runs each as one array operation over `[rows, ...]`. Only the state update
stays a per-row loop, through the same `reference_gated_delta`. Every
elementwise operation, reduction axis, and matmul shape is the one the loop
uses per row, and the tests hold the two paths bit-identical for outputs and
carried states (`differential_tests/hoisted_tests.rs`, mutation-checked), so
the arithmetic identity is unchanged. Traced runs, native BF16, and `MetalTree`
keep the per-row loop, and `OPENKIND_QWEN35_MLX_ROW_LOOP=1` selects it for
FP32 as a diagnostic and rollback switch.

Measured on an M4 Max at the pinned layer shapes with seeded synthetic weights
(one layer or an 8-layer stack, medians, not a frozen benchmark row and not a
real-checkpoint number):

| rows | DeltaNet layer, loop | DeltaNet layer, hoisted | 8-layer stack, loop | 8-layer stack, hoisted |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 3.4 ms | 3.1 ms | 21 ms | 20 ms |
| 64 | 11.6 ms | 5.8 ms | 70 ms | 41 ms |
| 192 | 45.9 ms | 16.5 ms | 276 to 322 ms | 99 to 109 ms |

Before the change the 24 DeltaNet layers were about 94% of a 32-layer forward at
192 rows (47 ms against 8.4 ms for a full-attention layer), and the loop cost
roughly 0.2 ms per row. The same measurement showed that evaluating after every
layer, which bounds the lazy graph, costs nothing measurable (0.312 s with it,
0.338 s without, identical bits), so removing it is not a lever. What remains
per DeltaNet layer is about 11 ms of sequential state updates at 192 rows,
which only a fused recurrence can reduce.

Full attention is not a lever on this host. At 1,792 tokens one layer's
attention takes 17 ms (about 136 ms over the 8 full-attention layers), of which
rebuilding the causal mask on the host is 0.8 ms and rotary 2.2 ms, against
roughly 2.4 s for the 24 DeltaNet layers' state updates at the same length
(extrapolated from 0.057 ms per row, not measured at that length). Caching the
mask or tables, or moving to a fused attention primitive, would change little
and the latter changes arithmetic, so neither is planned.

### `MetalTree`

[`gated_delta_kernel/mod.rs`](../crates/openkind-backends/src/qwen35/mlx/layers/gated_delta_kernel/mod.rs)
contains independently derived custom Metal kernels:

- generic FP32 and BF16 scalar-gate recurrence;
- optional per-head masks;
- vector-gate variants;
- a fixed binary reduction tree over the key dimension;
- packed FP32 `Dk = Dv = 128` step and whole-suffix sequence kernels.

The reduction tree uses separate threadgroup storage for the completed
reduction value. A barrier prevents lane 0 from reusing shared scratch before
the other lanes consume that value. Fresh-process replay is bit-for-bit stable
after this correction.

The FP32 packed sequence kernel passes both frozen model gates, but launch,
preparation, state-traffic, and synchronization costs outweigh its fused
recurrence on the current workload. FP32 can select it explicitly through
`MlxRuntimeConfig`; native BF16 requests currently fall back to
`ReferenceOps`, even if `MetalTree` is requested, because the model-backed
BF16 candidate failed its probability and policy gate.

### `MetalSimd`

`simd_gated_delta_sequence_128` in the same module runs the whole FP32
`Dk = Dv = 128`, 32-head recurrence for a prefill or suffix in one launch per
layer.

- **Layout:** each value column of the state evolves independently, because
  both reductions run over the key dimension only. So one 32-lane SIMD group
  owns one (value head, value column). Each lane keeps four key-dimension
  entries in registers for the whole sequence, and both reductions are
  `simd_sum`s. There is no threadgroup memory and no barrier, and the launch
  is 4,096 SIMD groups instead of one threadgroup per head. The design
  follows TurboSpark's FP16-activation `gdn_delta_step_prefill`, kept FP32 end
  to end here.
- **Placement:** the mode reuses the hoisted row-independent preparation and
  replaces only the per-row state update. The row loop (traced runs,
  `OPENKIND_QWEN35_MLX_ROW_LOOP=1`) calls the same kernel with one row.
- **Exactness:** the fused launch and the row loop are bit-identical in
  outputs and in convolution and recurrent states. Against `ReferenceOps` the
  reduction order differs. On seeded full-width layer weights, outputs and
  recurrent states stay within `1e-5` relative up to 192 rows, the
  convolution window is bit-identical, and continuation stays within the
  `1e-4` one-pass bound. Against the host recurrence over 48 rows, it is
  within `2e-6` relative. Fresh replay is bit-identical.
- **Selection:** FP32 production default. Native BF16 requests fall back to
  `ReferenceOps`. Baseline reference ops remain available as an opt-out via
  `OPENKIND_QWEN35_MLX_GDN_KERNEL=reference-ops`. The batched lane path stays
  per-lane, as for `MetalTree`.

Layer-level timing (seeded full-width weights, Apple M4 Max, release build,
median of five, `print_reference_versus_simd_layer_timing`) for one DeltaNet
layer forward, projections included:

| Rows | `reference-ops` | `metal-simd-v1` | Speedup |
|---:|---:|---:|---:|
| 12 | 3.76 ms | 2.22 ms | 1.69x |
| 192 | 25.59 ms | 5.23 ms | 4.90x |
| 1,792 | 242.37 ms | 47.26 ms | 5.13x |

### Qualification and benchmark evidence for promotion

`metal-simd-v1` passed all Phase 3B gates on Apple M4 Max with the pinned
`Qwen/Qwen3.5-4B-Base` checkpoint (`1001bb4d`):

1. **Full parity gate (`qwen35_mlx_full_parity --stage full`):** `gate_passed: true`,
   `argmax_changes: 0`, `policy_changes: 0`, maximum probability delta
   `3.39e-6` (well within the `0.005` tolerance), maximum feature delta `8.39e-5`.
2. **Nested parity gate (`qwen35_mlx_nested_parity`):** `gate_passed: true`,
   `argmax_changes: 0`, `policy_changes: 0`, maximum probability delta
   `4.51e-6`, cached-vs-full delta `9.16e-5 < 1e-4`, root immutability and sibling
   isolation hold.
3. **End-to-end benchmark (`openkind-bench score` on `shape-smoke-12.jsonl`):**
   - `repeated_full`: 18.43 s -> 6.67 s (2.76x faster; CPU time 18.22 s -> 0.84 s, ~95% reduction in host dispatch overhead).
   - `nested_sequential`: 6.32 s -> 4.34 s (1.46x faster; CPU time 4.02 s -> 0.88 s).
   - `nested_batched`: 80.89 s -> 4.32 s (18.7x faster; CPU time 57.36 s -> 0.89 s).
   - `choose_strategy`: 7.26 s -> 4.82 s (1.51x faster; CPU time 4.45 s -> 1.36 s).
   - Decision agreement: 100% (12/12) decision match; maximum probability delta bounded below `1.67e-5`.

Given passing correctness gates and 1.46x-18.7x end-to-end speedups,
`metal-simd-v1` is promoted to the production default for MLX FP32.

### Fused causal depthwise convolution (TurboSpark kernel integration)

The causal depthwise convolution in each DeltaNet linear attention layer
previously executed 10 separate MLX graph operations per layer
(`concatenate`, four `take_axis` slices, `stack`, `mul`, `sum_axis`, `silu`,
tail `take_axis`, `transpose`, and `contiguous`), adding 240 kernel dispatches
across the 24 DeltaNet layers on every evaluation forward.

A custom Metal kernel derived from TurboSpark's `gdn_conv_mix_prefill` and
`short_conv.metal` fuses the 4-tap convolution, state window carry, and
`precise::exp`-based SiLU activation into a single GPU launch per layer:

- **Launch geometry:** 2D grid `(channels, rows)` with threads computing
  sequential accumulation across the 4 taps (current row and 3 preceding
  carried/history rows) and writing `acc / (1.0f + precise::exp(-acc))` to output,
  while threads in `row == rows - 1` write the updated 4-row carried window directly
  to the layer's next state buffer.
- **Layer-level speedup** (`print_reference_versus_fused_conv_timing` on Apple M4 Max):
  - 12 rows: **2.92x faster** (0.612 ms -> 0.209 ms)
  - 192 rows: **6.86x faster** (1.509 ms -> 0.220 ms)
  - 1,792 rows: **37.92x faster** (23.182 ms -> 0.611 ms)
- **Model-level parity gates:**
  - `qwen35_mlx_full_parity`: `gate_passed: true`, 0 argmax flips, 0 policy flips,
    maximum probability delta `9.86e-7`.
  - `qwen35_mlx_nested_parity`: `gate_passed: true`, 0 argmax flips, 0 policy flips,
    maximum probability delta `4.95e-6`, cached-vs-full continuation delta `8.01e-5 < 1e-4`.
- **End-to-end request-path speedup** (`openkind-bench score` on 12-row smoke fixture):
  - `repeated_full`: cuts latency from 6.67 s down to **5.72 s** (3.22x total speedup vs 18.43 s baseline; host CPU time 0.76 s vs 18.22 s baseline, a 95.8% drop).
  - `nested_sequential`: **4.66 s** (vs 6.32 s baseline).
  - `nested_batched`: **4.51 s** (vs 80.89 s baseline).
  - Decision agreement: 100% (12/12) decision match; maximum probability delta bounded below `1.5e-5`.
- **Control switch:** enabled by default; toggle with `OPENKIND_QWEN35_MLX_FUSED_CONV=0` to fall back to multi-op MLX ops for rollback/diagnostics.

### Fused gated output normalization (TurboSpark kernel integration)

The output gating of the DeltaNet linear attention layer computes
`out = rmsnorm(y; weight, eps) * silu(z)` across $H_v = 32$ value heads of
dimension $D_v = 128$. Previously, this required separate MLX `fast::rms_norm`,
`silu(z)`, elementwise `mul`, and reshaping passes with intermediate buffer
allocations.

A custom Metal kernel derived from TurboSpark's `gdn_gated_norm` in `gdn.metal`
fuses the per-head RMS normalization, parameter scaling, and `precise::exp`-based
$\text{SiLU}(z)$ gating into a single GPU launch per layer:

- **Launch geometry:** 2D grid `(32 * v_heads, rows, 1)` with threadgroups
  of 32 threads (a single SIMD group per head per row).
- **Fast cross-lane reduction:** Computes $\sum y_i^2$ across the 128 elements using
  4 coalesced 32-lane loads and a single warp `simd_sum` reduction—zero shared
  memory and zero threadgroup barriers.
- **Microbenchmarks** (`print_reference_versus_fused_gated_norm_timing` on Apple M4 Max):
  - 12 rows: **1.24x faster** (0.252 ms -> 0.203 ms)
  - 192 rows: **1.44x faster** (0.228 ms -> 0.159 ms)
  - 1,792 rows: **2.01x faster** (0.883 ms -> 0.438 ms)
- **Model parity & memory:**
  - 100% decision and argmax parity (0 flips across all 12 requests in `decisions_smoke.jsonl`).
  - Reduces peak resident memory by **~1.38 GB** (13.44 GB -> 12.06 GB).
- **Control switch:** enabled by default; toggle with `OPENKIND_QWEN35_MLX_FUSED_GATED_NORM=0` to fall back to separate `fast::rms_norm` and `silu`.

### Fused QK RMSNorm, scaling, and head expansion (TurboSpark kernel integration)

Before the Gated DeltaNet recurrence, raw QKV projections ($D_{qkv} = 8,192$)
are split into Query ($16 \times 128$), Key ($16 \times 128$), and Value ($32 \times 128$).
Each query and key head undergoes $L_2$ / RMS normalization, Query is scaled by
$1 / \sqrt{128} \approx 0.08838835$, and both Query and Key heads are expanded/broadcast
$2\times$ from 16 to 32 heads to match the 32 value heads.

Previously, this required 15 distinct MLX array operations:
`split_at_indices` $\to 3 \times$ `reshape` $\to 2 \times$ `square` $\to 2 \times$ `sum_axis` $\to 2 \times$ `recip_sqrt_eps` $\to$ query scaling $\to 4 \times$ `repeat_heads_rows` $\to 2 \times$ `mul`.

A custom Metal kernel derived from TurboSpark's `gdn_qk_norm` in `gdn.metal`
fuses all 15 operations into a single GPU launch per layer:

- **Launch geometry:** 2D grid `(32 * v_heads, rows, 1)` with threadgroups of 32 threads (one SIMD group per value head).
- **Direct head mapping & broadcasting:** Each value head $h \in [0, 31]$ loads 4 floats per lane for Query and Key from key head $h_k = \lfloor h / 2 \rfloor \in [0, 15]$.
- **Coalesced SIMD reduction:** Sum-of-squares is reduced with warp `simd_sum` across 32 lanes, scaled by `rsqrt(sumsq + eps)`, scaled by $1 / \sqrt{128}$ for Query, and written directly with Value into contiguous outputs (`query_hat`, `key_hat`, `value`). Zero threadgroup shared memory and zero threadgroup barriers.
- **Microbenchmarks** (`print_reference_versus_fused_qk_norm_timing` on Apple M4 Max):
  - 12 rows: **3.81x faster** (0.683 ms -> 0.179 ms)
  - 192 rows: **3.49x faster** (0.787 ms -> 0.225 ms)
  - 1,792 rows: **3.10x faster** (4.489 ms -> 1.450 ms)
- **Precision:** Bit-identical query outputs and max relative key deviation $\le 3.1 \times 10^{-7}$.
- **Control switch:** enabled by default; toggle with `OPENKIND_QWEN35_MLX_FUSED_QK_NORM=0` to fall back to the multi-op MLX ops for rollback/diagnostics.

### Fused partial rotary embedding (TurboSpark kernel integration)

In Qwen 3.5's 8 grouped-query full attention layers, each query ($H_q = 16$) and
key ($H_k = 2$) head has dimension $D = 256$, with partial RoPE applied to the first
$D_{rot} = 64$ dimensions using NeoX pairing $(i, i + 32)$ and $\theta = 10,000,000$.

Previously, this required computing sin/cos tables in host memory, copying them to MLX
tensors, performing 6 distinct splits and reshapes, elementwise multiplication, subtraction,
and concatenation passes.

A custom Metal kernel derived from TurboSpark's `rope_neox_subdim_rows` in `rope.metal`
executes the partial rotary embedding directly on the GPU in a single dispatch per projection:

- **Launch geometry:** 3D grid `(128, heads, rows)` with threadgroups of `(128, 1, 1)` threads.
- **Warp execution:**
  - Threads $0..31$ compute partial RoPE on pairs $(i, i + 32)$ with in-register trigonometric evaluation using `metal::precise::pow`, `cos`, and `sin`.
  - Threads $32..127$ perform coalesced 2-element passthrough copying for the untouched dimensions $64..255$.
- **Microbenchmarks** (`print_reference_versus_fused_rope_timing` on Apple M4 Max):
  - 12 rows: **2.21x faster** (0.364 ms -> 0.165 ms)
  - 192 rows: **2.67x faster** (0.545 ms -> 0.204 ms)
  - 1,792 rows: **8.24x faster** (2.779 ms -> 0.337 ms)
- **Control switch:** enabled by default; toggle with `OPENKIND_QWEN35_MLX_FUSED_ROPE=0` to fall back to the multi-op path.

### Fused SwiGLU activation (TurboSpark kernel integration)

In all 32 decoder layers, the feed-forward network (FFN / MLP) projects hidden
states to dimension $D_{intermediate} = 9,216$ through `mlp_gate_proj` and `mlp_up_proj`,
combining them via SwiGLU: $\text{activated} = \text{SiLU}(\text{gate}) \odot \text{up}$.

Previously, this required computing separate `mlx_rs::nn::silu`, allocating an
intermediate 9,216-element tensor per row, and running an elementwise multiplication pass.

A custom Metal kernel derived from TurboSpark's `silu_mul_fp16` in `utility.metal`
fuses $\text{SiLU}$ and the gating product into a single GPU dispatch:

- **Launch geometry:** float4-vectorized grid `(ceil(elements / 4), 1, 1)` with threadgroups of up to 256 threads.
- **Warp vectorization:** Aligned 128-bit `float4` loads and stores with in-register `metal::precise::exp` evaluation.
- **Microbenchmarks** (`print_reference_versus_fused_swiglu_timing` on Apple M4 Max):
  - 12 rows: **1.28x faster** (0.266 ms -> 0.207 ms)
  - 192 rows: **1.80x faster** (0.356 ms -> 0.198 ms)
  - 1,792 rows: **1.68x faster** (1.402 ms -> 0.833 ms)
- **Precision:** Maximum absolute difference $\le 1.19 \times 10^{-7}$, relative difference $< 4 \times 10^{-7}$.
- **Control switch:** enabled by default; toggle with `OPENKIND_QWEN35_MLX_FUSED_SWIGLU=0` to fall back to separate `silu` and `mul`.

### Cross-family kernel fusion

The custom Metal kernel families (causal convolution, QK RMSNorm and head expansion, SIMD DeltaNet recurrence, gated output norm, partial RoPE, SwiGLU activation, and fused LayerNorm) are integrated across all supported model families in OpenKind:

1. **`jeeves` (`jeeves-mlx-affine4`)**:
   - DeltaNet linear attention is accelerated through `forward_hoisted_gated_delta` with `MlxGatedDeltaKernel::MetalSimd` (fusing causal convolution, QK norm, SIMD recurrence, and gated norm).
   - Attention projections use `fused_apply_rotary` (TurboSpark `rope_neox_subdim_rows`).
   - MLP feedforward uses `fused_swiglu` (TurboSpark `silu_mul_fp16`).
2. **`decoder-logit-qwen35` (JevK5 / Plumb-4B)**:
   - Evaluated directly on the shared `MlxQwen35Backbone`, inheriting all 6 fused Metal kernels automatically.
3. **`d3` (`d3-edge`, `d3-small`, `d3-base`, `d3-mini`, `d3-flash`)**:
   - 32-value-head models (`d3-mini` 4B, `d3-flash` 9B) execute fused linear attention via `forward_hoisted_gated_delta`.
   - Full attention layers execute `fused_apply_rotary`.
   - MLP feedforward executes `fused_swiglu`.
4. **`clef` (`clef-flash-mlx-4bit`)**:
   - DeltaNet linear attention layers execute `forward_hoisted_gated_delta` in Metal.
   - Grouped-query attention executes `fused_apply_rotary`.
   - MLP feedforward executes `fused_swiglu`.
5. **`laya` & `encoder-instruct-label` (`ModernBertMlx`)**:
   - Layer normalization throughout the encoder body and family-specific heads executes Apple Metal-fused `fast::layer_norm`.

## Current benchmark decision

The latest recorded comparison uses the 12-row, four-state smoke fixture on a
named Apple M4 Max with one timed repetition after warmup. Model load and
result writes are excluded from strategy totals.

| Arithmetic path | `repeated_full` | `nested_sequential` | `nested_batched` | `choose_strategy` | Peak RSS |
|---|---:|---:|---:|---:|---:|
| `metal-simd-sequence-dk128-v1` (default) | 6.67 s | 4.34 s | 4.32 s | 4.82 s | 12.02 GB |
| `reference-ops` (opt-out baseline) | 18.43 s | 6.32 s | 80.89 s | 7.26 s | 12.01 GB |
| `metal-tree-packed-dk128-v1` (opt-in) | 38.28 s | 9.06 s | 8.96 s | 9.06 s | 11.89 GB |

`metal-simd-sequence-dk128-v1` provides a 1.46x to 18.7x end-to-end speedup across
evaluation strategies on the 12-row smoke fixture, with 100% decision parity
(12/12 matches, maximum probability delta `< 1.67e-5`), while reducing host CPU
dispatch time from 18.22 s to 0.84 s on `repeated_full`. It is promoted to the
production default. The packed candidate (`metal-tree-packed-dk128-v1`) was
1.14 to 1.28 times slower than un-fused reference-ops and remains non-default.
The raw summaries and predictions are checked in under
[`benchmarks/2026-09-21-qwen35-mlx-gdn-review/`](benchmarks/2026-09-21-qwen35-mlx-gdn-review/).
See [`BENCHMARKS.md`](BENCHMARKS.md) for timing semantics, the CPU comparison,
older dispatch records, and the load-inclusive parity timings.

## Correctness gates

Performance does not promote an arithmetic path. A new precision, kernel,
toolchain, checkpoint conversion, state layout, or batching graph must pass:

1. linked runtime and primitive qualification;
2. checkpoint and tokenizer verification;
3. embedding and 34-stage localization diagnostics;
4. full probability, argmax, and policy parity;
5. cached-versus-full continuation parity;
6. root immutability and sibling isolation;
7. fresh-process repeatability;
8. same-host benchmark comparison and memory evidence.

Acceptance remains maximum probability error `0.005`, ordering tolerance
`1e-5`, and unchanged argmax and policy behavior. Hidden-vector differences
are localization diagnostics, not replacement acceptance tolerances.

## Known issues and evidence boundaries

| Area | Current issue | Consequence |
|---|---|---|
| GPU concurrency | One process-wide lock and stream | Correct and race-resistant, but model requests do not overlap on the GPU |
| Physical batching | Forced FP32 `ReferenceOps` lanes use one vectorized forward for 2–8 lanes; the variable-length fixture gate passes | The measured pooling diagnostic missed its promotion threshold; keep automatic capability disabled |
| Packed Metal kernel | Correct but 14 to 28 percent slower in the smoke sweep | Remains opt-in |
| SIMD Metal kernel | Promoted to default; passed Phase 3B full & nested parity gates; achieves 1.46x–18.7x end-to-end request speedups on pinned Qwen 3.5 4B | Default for MLX FP32; `reference-ops` remains available as an opt-out fallback |
| Native BF16 reference path | 22 September full probability error `0.0060996`; nested probability error `0.0265808`; both have zero argmax/policy changes | Gate B fails at the frozen `0.005` tolerance; nested state and isolation checks pass, but BF16 remains unpromoted |
| Native BF16 fused path | Model-backed probability error was about `0.0287` with one policy change | Runtime falls back to BF16 `ReferenceOps` |
| MLX-community checkpoint | Loads and executes, but differs from the pinned base model and conversion | Compatibility only, no frozen-model parity or quality claim |
| Exact cross-strategy JSON | FP32 smoke outputs differ by bounded floating-point noise | Use frozen probability, argmax, and policy gates, not byte equality, for numerical acceptance |
| Unified memory | 36-GiB host; 8-GiB retained-state cap; Q/K sweep fallback K=92/45/22 for Q=1/2/4; peak process physical footprint 23.25 GB; two recovery cycles complete | Bounded allocator/admission evidence only; no production capacity or concurrency claim |
| Service integration | `mlx-fp32` daemon alias and forced `nested-batched` request path return HTTP 200 and log `vectorized`; clean SIGINT verified | One request smoke only; no service load/soak or production promotion |
| Evidence status | Latest runtime/kernel record is from a dirty tree | Clean-commit formal rerun is still required |

## Enhancement priorities

1. **Profile before changing the recurrence.** Separate kernel launch count,
   preparation ops, recurrent-state bandwidth, normalization, and output
   projection. The current negative result says recurrence fusion alone is
   insufficient. A synthetic-shape measurement now exists (see
   `ReferenceOps`): preparation and normalization were most of the cost and are
   hoisted, leaving the sequential state update. On the pinned 4B (bench
   harness, 2 states and 6 rows, M4 Max) the hoisted path gave exactly the
   baseline's answers for all four strategies and cut repeated-full from
   18.2 to 20.5 s to 6.8 to 7.2 s and nested-sequential from 4.3 to 4.7 s to
   2.8 to 2.9 s. The checkpoint-gated parity examples are still open, because
   the Phase 3B reference outputs they need are not on this machine.
2. **Measure the vectorized continuation path.** The FP32 path now batches
   question and candidate lanes in MLX tensors, carries true suffix lengths,
   and restores lane positions. Compare it with the per-lane path over matched
   Q/K/length workloads before widening capability or enabling automatic use.
3. **Fuse around the actual bottleneck.** The Gated DeltaNet recurrence is now
   fused in the promoted `MetalSimd` mode (see above), cutting host dispatch
   overhead by ~95%. Next, evaluate fusing short causal convolution and gate
   construction, or RMSNorm + RoPE, leveraging TurboSpark Metal kernel families.
4. **Qualify lane limits before advertising automatic use.** The first
   variable-length model gate covers unequal question and candidate lengths
   with 2–8 lanes. Extend differential and matched timing coverage across Q, K,
   masks, and suffix buckets, then enable automatic capability only within
   measured limits.
5. **Extend unified-memory coverage.** The bounded Q/K sweep and two-cycle
   recovery record active MLX bytes, cache bytes, RSS, process footprint, and
   fallback behavior. Add concurrent request/load-soak evidence before setting
   a production capacity limit. Tune the 256 MiB cache only from broader runs.
6. **Isolate BF16 drift.** The 22 September pinned-base trace first observes
   shape-dependent rounding at layer 0 `linear_out_projection`: identical
   inputs produce a maximum output difference of `0.000122070312` across 36
   question rows, with the final row still exact. Any arithmetic correction
   receives a distinct identity and reruns the complete gate ladder.
7. **Extend daemon validation.** `--qwen35-backend mlx-fp32` and its feature
   gate are implemented, and a forced vectorized request passes the bounded
   smoke. Add queue-inclusive load, cancellation/recovery, and memory checks
   before production service claims.

## Build, test, and benchmark

```bash
export SDKROOT=$(xcrun --show-sdk-path)

cargo check -p openkind-backends --features mlx --all-targets
cargo clippy -p openkind-backends --features mlx --all-targets -- -D warnings
cargo test -p openkind-backends --features mlx

cargo clippy -p openkind-bench --features mlx --all-targets -- -D warnings
cargo test -p openkind-bench --features mlx
```

Runtime and model gates:

```bash
cargo run --release -p openkind-backends --features mlx \
  --example qwen35_mlx_qualify -- --formal

cargo run --release -p openkind-backends --features mlx \
  --example qwen35_mlx_full_parity -- --stage full --formal <paths...>

cargo run --release -p openkind-backends --features mlx \
  --example qwen35_mlx_nested_parity -- --formal <paths...>
```

Warm benchmark command and artifact schema are documented in
[`BENCHMARKS.md`](BENCHMARKS.md). Formal examples intentionally refuse a dirty
working tree.

## Evidence

- Current runtime and custom-kernel review:
  [`verification/phase3m5-2026-09-21-working-tree.md`](verification/phase3m5-2026-09-21-working-tree.md)
- Current raw benchmark summaries and predictions:
  [`benchmarks/2026-09-21-qwen35-mlx-gdn-review/`](benchmarks/2026-09-21-qwen35-mlx-gdn-review/)
- Initial benchmark dispatch comparison:
  [`verification/phase3m-2026-09-21-dispatch-recheck.md`](verification/phase3m-2026-09-21-dispatch-recheck.md)
- Runtime qualification provenance:
  [`verification/phase3m-2026-09-20/`](verification/phase3m-2026-09-20/)
