# Models

> The operator-facing index of every profile `openkind` can load: model type,
> source weights, catalog pull names, execution backends, and measured size,
> memory, and throughput.

## Ownership and sync

[`registry/v1/catalog.json`](../registry/v1/catalog.json) is the machine
source for catalog names, profile IDs, and manifest digests; this page is its
human-readable mirror. [`families/README.md`](families/README.md) owns the
family-to-loader catalogue and the surveyed-family list.
[`MODEL_REGISTRY.md`](MODEL_REGISTRY.md) owns pull, publication, and mirror
verification commands. [`BENCHMARKS.md`](BENCHMARKS.md) owns measurement
methodology. When a change edits `registry/v1` or lands a new loadable
profile, update the tables here in the same change.

## Pull a catalog model

Four profiles are catalog-installable. `openkind pull NAME` downloads pinned
artifacts, verifies every digest, and installs them for the daemon:

```bash
openkind catalog
openkind pull qwen35-state-first:a047d6802c3f06f085b8
openkind list
openkind serve --installed-models qwen35-state-first:a047d6802c3f06f085b8
```

The other Rust-loadable profiles are not in the catalog: download their
checkpoints from the linked repositories as an explicit operator step and
configure them with the artifact-path flags documented on their family pages.

## Models by type

| Model | Type | Backbone weights | Params | Profile ID | How to get it |
|---|---|---|---|---|---|
| [`encoder-state-first`](families/encoder-state-first.md) `qwen35-state-first` | Qwen3.5 hybrid decoder (24 DeltaNet + 8 attention blocks) | [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) + [OpenKind bundle](https://huggingface.co/cowWhySo/OpenKind-Qwen3.5-4B-StateFirst) | 4B | `a047d6802c3f06f085b8` | `openkind pull qwen35-state-first:a047d6802c3f06f085b8` |
| [`decoder-logit-qwen35`](families/decoder-logit-qwen35.md) | Qwen3.5 hybrid decoder (merged LoRA, letter-logit readout) | [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5) (merged Qwen3.5-4B) | 4B | `415bcf4a064e6dadcf85` | [family page](families/decoder-logit-qwen35.md) |
| [`kev`](families/kev.md) | Qwen3 decoder (LoRA + pointer head) | [jaredpalmer/kev-0.6b](https://huggingface.co/jaredpalmer/kev-0.6b) over [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 0.6B | `39d88c11faeb4ac165fa` | [family page](families/kev.md) |
| [`qwen3guard`](families/qwen3guard.md) (Stream) | Qwen3 decoder (safety-verdict classification head) | [Qwen/Qwen3Guard-Stream-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) | 0.6B | `0fcf416cab16d94f933d` | [family page](families/qwen3guard.md) |
| [`decoder-logit-letter`](families/decoder-logit-letter.md) | Qwen2.5 decoder (option-letter logit readout) | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.5B | `5492c97dfcdaf3fe9439` | [family page](families/decoder-logit-letter.md) |
| [`decoder-logit-llm`](families/decoder-logit-llm.md) | Qwen2.5 decoder, GGUF q8_0 (label-logit readout) | [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | 0.5B | `465963d705b6f35d6208` | [family page](families/decoder-logit-llm.md) |
| [`winnow`](families/winnow.md) | Qwen2.5 decoder + in-house LoRA (script-aware router) | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) + vendored adapter | 0.5B | `4dff8c5b03cfbf680db6` | [family page](families/winnow.md) |
| [`laya-english`](families/laya.md) | ModernBERT-large encoder, typed-decision marker head | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | — | `c8ea29bf1e33a343c4b7` | `openkind pull laya-english:c8ea29bf1e33a343c4b7` |
| [`laya-multilingual`](families/laya.md) | mmBERT-base encoder, typed-decision marker head | [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) (encoder derives from [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)) | — | `f4064eb56fb7f7d325e1` | `openkind pull laya-multilingual:f4064eb56fb7f7d325e1` |
| [`laya-typed-decisions`](families/laya.md) | ModernBERT-large encoder (fine-tuned), typed-decision marker head | [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | — | `9d28cfa9567902801ed1` | `openkind pull laya-typed-decisions:9d28cfa9567902801ed1` |
| [`encoder-nli`](families/encoder-nli.md) | DistilBERT encoder (entailment-probability readout) | [typeform/distilbert-base-uncased-mnli](https://huggingface.co/typeform/distilbert-base-uncased-mnli) | — | `1041a4c362338a61b820` | [family page](families/encoder-nli.md) |
| [`encoder-instruct-label`](families/encoder-instruct-label.md) | GLiClass uni-encoder over ModernBERT-base (label markers) | [knowledgator/gliclass-modern-base-v3.0](https://huggingface.co/knowledgator/gliclass-modern-base-v3.0) | 149M | `9fd68313a5606eca42f2` | [family page](families/encoder-instruct-label.md) |
| [`schema-scorer`](families/schema-scorer.md) | MiniLM-L-6 cross-encoder (single-logit schema score) | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | — | `5a7350af556f0ee66566` | [family page](families/schema-scorer.md) |
| [`router-script`](families/router-script.md) | Unicode script detector over registered siblings | none — fixed rule table | — | — | no artifacts |

All profiles are prototype status: they load pinned local artifacts offline
and pass parity fixtures, but none carries reviewed task-quality evidence. A
checkpoint's own model-card accuracies belong to its authors, not to
`openkind` measurements.

## Execution backends, size, memory, and measured runs

CPU fp32 (candle) runs everywhere and is the correctness oracle. The MLX
backend runs the Metal GPU on Apple silicon (macOS arm64, `--features mlx`;
see [`MLX.md`](MLX.md)); it is parity-qualified for the profiles below and
does not exist for the others. There is no CUDA or ROCm backend.

| Profile | Backend | Download | Peak RSS | Decisions/s | Input tok/s | Model load | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| `qwen35-state-first` | CPU fp32 | 9.34 GB | 18.62 GB | 0.22 | — | 16.5 s | [registry campaign](benchmarks/2026-09-28-registry-mlx-campaign/README.md) |
| `qwen35-state-first` | MLX fp32 (qualified) | same | 12.09 GB | 2.46 | — | ~21 s | [registry campaign](benchmarks/2026-09-28-registry-mlx-campaign/README.md) |
| `qwen35-state-first` | MLX bf16 (unqualified) | same | — | — | — | — | throughput-only probe; fails the frozen probability tolerance, never decision-safe |
| `decoder-logit-qwen35` | CPU fp32 | — | 8.04 GB | 0.16 | — | 15.2 s | [summary](benchmarks/2026-09-27-decoder-logit-qwen35/summary-decoder-logit-qwen35.json) |
| `kev` | CPU fp32 | — | 4.22 GB | 4.84 | — | 5.6 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-kev.json) |
| `qwen3guard` | CPU fp32 | — | 4.20 GB | 4.97 | — | 2.4 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-qwen3guard.json) |
| `decoder-logit-letter` | CPU fp32 | — | 3.33 GB | 7.22 | — | 2.0 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-decoder-letter.json) |
| `decoder-logit-llm` | CPU fp32 (GGUF q8_0 weights) | — | 1.55 GB | 0.42 | — | 1.5 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-decoder-llm.json) |
| `winnow` | CPU fp32 (router pass) | — | 3.32 GB | 194.6 | — | 2.4 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-winnow.json) |
| `laya-english` | CPU fp32 | 0.85 GB | 2.75 GB | 2.23 | 455 | 1.8 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-english` | MLX fp32 (qualified) | same | 2.12 GB | 24.34 | 4,964 | 2.8 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-multilingual` | CPU fp32 | 0.68 GB | 2.80 GB | 5.30 | 1,075 | 2.0 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-multilingual` | MLX fp32 (qualified) | same | 2.94 GB | 56.37 | 11,431 | 3.5 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-typed-decisions` | CPU fp32 | 0.85 GB | 2.76 GB | 2.33 | 475 | 1.8 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-typed-decisions` | MLX fp32 (qualified) | same | 2.13 GB | 23.77 | 4,849 | 2.9 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `encoder-nli` | CPU fp32 | — | 0.56 GB | 35.14 | — | 0.5 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-encoder-nli.json) |
| `encoder-instruct-label` | CPU fp32 | — | 1.27 GB | 4.00 | — | 1.2 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-encoder-instruct-label.json) |
| `schema-scorer` | CPU fp32 | — | 0.26 GB | 10.25 | — | 0.2 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-schema-scorer.json) |
| `router-script` | CPU (rule table) | — | 0.01 GB | 704,336 | — | ~0 | [summary](benchmarks/2026-09-26-surveyed-families/summary-router-script.json) |

Reading notes:

- Download is the manifest-pinned artifact total for catalog models; a dash
  means the checkpoint downloads directly from the linked repository and no
  size is pinned here.
- All measured cells are single-sample `--reps 1` runs of the seeded shape777
  workload (`be397bfc…`, 777 decisions, grouped by state, warm process) on
  `openkind-mac-arm64-local` — an Apple M4 Max, 14 logical cores, fp32,
  request-path timing with model load reported separately. Peak RSS is the
  process high-water mark including mapped weights and runtime scratch, so it
  is not per-request memory.
- `qwen35-state-first` CPU and MLX rows are `choose_strategy` (the scheduler
  picks a nested plan); fresh `repeated_full` scoring on MLX costs 0.32
  decisions/s. Peak RSS is shared across strategies of one process.
- `winnow` and `router-script` measure the routing pass over mock siblings,
  not end-to-end decisions.
- The `qwen35-mlx-bf16` engine is an unqualified candidate: it fails the
  frozen probability tolerance (see [`MLX.md`](MLX.md)) and exists as a
  throughput probe only.
- Engines run through `openkind-bench score --engine NAME` (`qwen35`,
  `qwen35-mlx-fp32`, `decoder-letter`, `encoder-nli`, `encoder-instruct-label`,
  `decoder-llm`, `schema-scorer`, `qwen3-guard`, `kev`,
  `decoder-logit-qwen35`, `laya-english`, `laya-multilingual`,
  `laya-typed-decisions`, `winnow`, `router-script`, and the `*-mlx-fp32`
  variants) or through the daemon (`--installed-models` for catalog profiles,
  `--qwen35-backend mlx-fp32` and `--laya-backend mlx-fp32` for the MLX
  paths). Family pages document per-family artifact-path flags.
