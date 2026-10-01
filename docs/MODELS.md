# Models

> The operator-facing index of every profile `openkind` can load: model type,
> source weights, catalog pull names (plus any compatible short aliases),
> execution backends, and measured size, memory, and throughput.

## All models

[`registry/v1/catalog.json`](../registry/v1/catalog.json) is the machine
source for the pull names and profile IDs below, and
[`crates/openkind-model-store/tests/models_doc.rs`](../crates/openkind-model-store/tests/models_doc.rs)
fails when this table and the catalog drift apart. Pull names are OpenKind
profile identities, not upstream checkpoint names; each row's HuggingFace
source repository has its own column. Choose by task shape,
download size, and measured speed: CPU fp32 (candle) runs everywhere and is
the correctness oracle. The MLX fp32 backend runs the Metal GPU on Apple
silicon (macOS arm64, `--features mlx`; see [`MLX.md`](MLX.md)) and is listed
only where parity-qualified. There is no CUDA or ROCm backend. Measured cells
read `peak RSS · throughput` on the seeded shape777 workload, except where
marked below (single-sample warm runs on an Apple M4 Max; model-load times,
input token rates, and strategy caveats are in the per-backend table below).

| Model | Type | Source weights (HuggingFace) | Pull name | Download | CPU fp32 (candle) | MLX fp32 | Evidence |
|---|---|---|---|---:|---|---|---|
| [`qwen35-state-first`](families/encoder-state-first.md) | Qwen3.5 hybrid decoder (24 DeltaNet + 8 attention blocks), 4B | [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) + [OpenKind bundle](https://huggingface.co/cowWhySo/OpenKind-Qwen3.5-4B-StateFirst) | `qwen35-state-first:a047d6802c3f06f085b8` | 9.34 GB | 18.62 GB · 0.22/s | 12.09 GB · 2.46/s | [registry campaign](benchmarks/2026-09-28-registry-mlx-campaign/README.md) |
| [`decoder-logit-qwen35`](families/decoder-logit-qwen35.md) | Qwen3.5 hybrid decoder (merged LoRA, letter-logit readout), 4B | [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5) (merged Qwen3.5-4B) | `decoder-logit-qwen35:415bcf4a064e6dadcf85` (`jevk5:4b`) | 8.43 GB | 8.04 GB · 0.16/s | 6.98 GB · 0.43/s | [CPU summary](benchmarks/2026-09-27-decoder-logit-qwen35/summary-decoder-logit-qwen35.json) · [MLX](benchmarks/2026-09-29-mlx-counterparts/README.md) |
| [`kev`](families/kev.md) | Qwen3 decoder (LoRA + pointer head), 0.6B | [jaredpalmer/kev-0.6b](https://huggingface.co/jaredpalmer/kev-0.6b) over [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `kev:39d88c11faeb4ac165fa` | 1.25 GB | 4.22 GB · 4.84/s | — | [summary](benchmarks/2026-09-26-surveyed-families/summary-kev.json) |
| [`qwen3guard`](families/qwen3guard.md) (Stream) | Qwen3 decoder (safety-verdict classification head), 0.6B | [Qwen/Qwen3Guard-Stream-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) | `qwen3guard:0fcf416cab16d94f933d` | 1.21 GB | 4.20 GB · 4.97/s | — | [summary](benchmarks/2026-09-26-surveyed-families/summary-qwen3guard.json) |
| [`decoder-logit-letter`](families/decoder-logit-letter.md) | Qwen2.5 decoder (option-letter logit readout), 0.5B | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | `decoder-logit-letter:5492c97dfcdaf3fe9439` | 0.99 GB | 3.33 GB · 7.22/s | — | [summary](benchmarks/2026-09-26-surveyed-families/summary-decoder-letter.json) |
| [`decoder-logit-llm`](families/decoder-logit-llm.md) | Qwen2.5 decoder, GGUF q8_0 (label-logit readout), 0.5B | [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | `decoder-logit-llm:465963d705b6f35d6208` | 0.68 GB | 1.55 GB · 0.42/s | — | [summary](benchmarks/2026-09-26-surveyed-families/summary-decoder-llm.json) |
| [`winnow`](families/winnow.md) | Qwen2.5 decoder + in-house LoRA (script-aware router), 0.5B | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) + in-house LoRA (catalog-pinned) | `winnow:4dff8c5b03cfbf680db6` | 1.00 GB | 3.32 GB · 194.6/s † | — | [summary](benchmarks/2026-09-26-surveyed-families/summary-winnow.json) |
| [`decoder-logit-qwen35`](families/decoder-logit-qwen35.md) (Plumb-4B) | Qwen3.5 hybrid decoder (JevK5 fine-tune, single letter read), 4B | [crh225/plumb-4b](https://huggingface.co/crh225/plumb-4b) | `plumb-4b:c1f080794d38e94a0bc2` (`plumb:4b`) | 8.43 GB | 7.83 GB · 0.16/s (choice diagnostic) | — | [smoke + diagnostic](benchmarks/2026-09-30-jevbench-expansion/README.md) |
| [`decoder-logit-qwen3-4b`](families/decoder-logit-qwen3.md) | Qwen3 dense decoder (raw direct-logit control, temperature 1.0), 4B | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `decoder-logit-qwen3-4b:9dfaf11792a8d061b6b8` | 8.04 GB | measured run pending | — | measured run pending |
| [`decoder-logit-qwen3-17b`](families/decoder-logit-qwen3.md) | Qwen3 dense decoder (raw direct-logit control, temperature 1.0), 1.7B | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | `decoder-logit-qwen3-17b:8119b9271f8d011e7d03` | 4.06 GB | measured run pending | — | measured run pending |
| [`decoder-logit-qwen3-06b`](families/decoder-logit-qwen3.md) | Qwen3 dense decoder (raw direct-logit control, temperature 1.0), 0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `decoder-logit-qwen3-06b:d900f4af57509fe02e62` | 1.50 GB | measured run pending | — | measured run pending |
| [`decider-4b`](families/decider.md) | Qwen3.5 hybrid decoder (slot-logit readout, plain state-first layout, isolated score levels), 4B | [Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b) | `decider-4b:0529bf6f2bed84641701` | 8.43 GB | 7.91 GB · 0.06/s (smoke) | — | [smoke summary](benchmarks/2026-09-30-jevbench-expansion/summary-decider-4b-smoke.json) |
| [`von`](families/von.md) | ModernBERT-large encoder, option-marker scorer | [wfzyx/von](https://huggingface.co/wfzyx/von) (von-1.1) | `von:69219703407bd39cca0c` (`von:1.1`) | 1.58 GB | 1.83 GB · 2.25/s | — | [summary](benchmarks/2026-09-30-von/summary-von.json) |
| [`laya-english`](families/laya.md) | ModernBERT-large encoder, typed-decision marker head | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | `laya-english:c8ea29bf1e33a343c4b7` (`laya:en`) | 0.85 GB | 2.75 GB · 2.23/s | 2.12 GB · 24.34/s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| [`laya-multilingual`](families/laya.md) | mmBERT-base encoder, typed-decision marker head | [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) (encoder derives from [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)) | `laya-multilingual:f4064eb56fb7f7d325e1` (`laya:multilingual`) | 0.68 GB | 2.80 GB · 5.30/s | 2.94 GB · 56.37/s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| [`laya-typed-decisions`](families/laya.md) | ModernBERT-large encoder (fine-tuned), typed-decision marker head | [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | `laya-typed-decisions:9d28cfa9567902801ed1` (`laya:typed-decisions`) | 0.85 GB | 2.76 GB · 2.33/s | 2.13 GB · 23.77/s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| [`encoder-nli`](families/encoder-nli.md) | DistilBERT encoder (entailment-probability readout) | [typeform/distilbert-base-uncased-mnli](https://huggingface.co/typeform/distilbert-base-uncased-mnli) | `encoder-nli:1041a4c362338a61b820` | 0.27 GB | 0.56 GB · 35.14/s | — | [summary](benchmarks/2026-09-26-surveyed-families/summary-encoder-nli.json) |
| [`encoder-instruct-label`](families/encoder-instruct-label.md) | GLiClass uni-encoder over ModernBERT-base (label markers), 149M | [knowledgator/gliclass-modern-base-v3.0](https://huggingface.co/knowledgator/gliclass-modern-base-v3.0) | `encoder-instruct-label:9fd68313a5606eca42f2` | 0.61 GB | 1.27 GB · 4.73/s | 1.02 GB · 75.42/s | [mlx counterparts campaign](benchmarks/2026-09-29-mlx-counterparts/README.md) |
| [`schema-scorer`](families/schema-scorer.md) | MiniLM-L-6 cross-encoder (single-logit schema score) | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | `schema-scorer:5a7350af556f0ee66566` | 0.09 GB | 0.26 GB · 10.25/s | — | [summary](benchmarks/2026-09-26-surveyed-families/summary-schema-scorer.json) |
| [`encoder-embedding`](families/encoder-embedding.md) | BGE-small BERT sentence encoder, CLS pooling (proxy-cache student embedder), 33M | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | `encoder-embedding:8d9498269ef05d95d93c` | 0.13 GB | 0.27 GB · 40.52 emb/s | 0.27 GB · 220.36 emb/s | [component run](benchmarks/2026-09-30-encoder-embedding/README.md) |
| [`router-script`](families/router-script.md) | Unicode script detector over registered siblings | none — fixed rule table | — | — | 0.01 GB · 704,336/s † | — | [summary](benchmarks/2026-09-26-surveyed-families/summary-router-script.json) |

Reading notes:

- Throughput is decisions/s; `emb/s` marks the proxy-cache embedder, whose
  benchmark times single-text encodes rather than decision requests.
- Plumb uses the 96-row choice diagnostic here; Decider uses the 12-row smoke
  workload. These cells cannot be compared directly with shape777 or each other.
- † `winnow` and `router-script` measure the routing pass over mock siblings,
  not end-to-end decisions.
- The `qwen35-state-first` rows are `choose_strategy` (the scheduler picks a
  nested plan); fresh `repeated_full` scoring on MLX costs 0.32 decisions/s.
  The `decoder-logit-qwen35` rows are full-forward-per-pass compute and are
  not strategy-comparable with the state-first engine.
- The `qwen35-mlx-bf16` engine is an unqualified candidate: it fails the
  frozen probability tolerance (see [`MLX.md`](MLX.md)) and exists as a
  throughput probe only.
- All profiles are prototype status: they load pinned local artifacts offline
  and pass parity fixtures, but none carries reviewed task-quality evidence.
  A checkpoint's own model-card accuracies belong to its authors, not to
  `openkind` measurements.
- JevBench board names (benchmarkheaven.com/jev-models) map through the
  source-weights column: `decoder-logit-qwen35` is the board's JevK5 entry
  (#4 on v1.5.4), `kev` is the kev 0.6B checkpoint (the board's kev 4B and
  8B rows are different checkpoints), and the laya profiles sit below the
  top 50 (#86/#93/#106). The in-house `winnow` router is unrelated to the
  board's #2 Winnow-12B Q8 ([EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B)).

## Pull a catalog model

Sixteen profiles are catalog-installable: every Rust-loadable profile
except `router-script`, which needs no artifacts. `openkind pull NAME`
downloads pinned artifacts, verifies every digest, and installs them for the
daemon:

```bash
openkind catalog
openkind pull qwen35-state-first:a047d6802c3f06f085b8
openkind list
openkind serve --installed-models qwen35-state-first:a047d6802c3f06f085b8
```

`router-script` is not a pull: it has no artifacts and composes registered
sibling profiles through a fixed rule table configured with
`--router-script-aliases` / `--router-script-rules`.

## Naming schema and ollaya-compatible names

OpenKind pull names are `loader-id:profile-digest` (a 20-hex content digest),
which pins the exact artifact set a name identifies. Model names shared
between users are also pinned, so a name pasted from another operator always
installs the same bytes.

The [ollaya decision-model registry](https://github.com/ollaya-dev/ollaya)
(`registry/v2/library`) uses the Ollama-style `name:tag` schema instead —
short names such as `laya:en` or `jevk5:4b` with `latest` as the implicit
default tag. Where an ollaya name identifies the same underlying checkpoint
OpenKind pins, the catalog carries it as an alias and `openkind pull` accepts
both spellings. The alias policy:

- An alias is added only when the checkpoint identity matches: the laya
  weights and tokenizers are byte-identical between the two registries
  (verified by SHA-256), and `jevk5:4b` is the same JevK5 v0.3 checkpoint —
  OpenKind serves its pinned unquantized BF16-widened-to-FP32 weights while
  ollaya serves the author's Q8_0 GGUF, so numeric outputs differ even where
  the model identity matches.
- An alias resolves at pull time; the installation is keyed by the canonical
  pull name. `openkind list`, `openkind show`, `openkind rm`, and the
  daemon's `--installed-models` take canonical names only — the daemon never
  resolves names over the network at startup.
- There are no `latest` aliases: OpenKind has no moving default tag. There
  are no `-fp16`/`-fp32` precision aliases: those ollaya tags select an
  inference-graph precision, while OpenKind always executes the FP32
  reference graph.

## Pull-name aliases

| Alias | Canonical pull name | Weight identity |
|---|---|---|
| `laya:en` | `laya-english:c8ea29bf1e33a343c4b7` | byte-identical weights and tokenizer |
| `laya:multilingual` | `laya-multilingual:f4064eb56fb7f7d325e1` | byte-identical weights and tokenizer |
| `laya:typed-decisions` | `laya-typed-decisions:9d28cfa9567902801ed1` | byte-identical weights and tokenizer |
| `jevk5:4b` | `decoder-logit-qwen35:415bcf4a064e6dadcf85` | same JevK5 v0.3 checkpoint; FP32 here vs Q8_0 GGUF there |
| `plumb:4b` | `plumb-4b:c1f080794d38e94a0bc2` | JevBench board name with no ollaya counterpart; checkpoint pinned directly |
| `qwen3:4b` | `decoder-logit-qwen3-4b:9dfaf11792a8d061b6b8` | JevBench board name with no ollaya counterpart; checkpoint pinned directly |
| `von:1.1` | `von:69219703407bd39cca0c` | byte-identical `option_marker.pt` (digest-verified in both registries) |
| `decider:4b` | `decider-4b:0529bf6f2bed84641701` | byte-identical weights and tokenizer (digest-verified in both registries) |

```bash
openkind pull laya:en
# Resolved alias laya:en to the curated profile laya-english:c8ea29bf1e33a343c4b7.
```

Aliases share the canonical profile's engine, so the measured cells in
[All models](#all-models) and the per-backend table below are also the
benchmarks for the aliased names.

## Ollaya catalog coverage

Every family in the ollaya library, mapped to OpenKind. "Not loadable" means
no OpenKind backend executes those checkpoints yet, so there is no OpenKind
benchmark for them either; any published accuracy belongs to the checkpoint
authors.

| Ollaya name | Ollaya tags | OpenKind status |
|---|---|---|
| `laya` | `latest` (router) | not loadable — ollaya's laya router dispatches to the language tags; install the needed sibling directly here |
| `laya` language encoders | `en`, `multilingual`, `typed-decisions` (+`-fp16`/`-fp32`) | supported via aliases (see [Pull-name aliases](#pull-name-aliases)) |
| `kev` | `0.8b`, `4b`, `9b`, `latest` | same family, different checkpoints — OpenKind pins `kev:39d88c11faeb4ac165fa` (kev-0.6b over Qwen3-0.6B-Base); ollaya pins kev-0.8b/4b/9b LoRAs over Qwen3.5-0.8B/4B/9B-Base, which need a Qwen3.5 kev-head loader |
| `qwen3guard` | `0.6b`, `latest` | same family, different variant — OpenKind pins Qwen3Guard-**Stream**-0.6B; ollaya pins Qwen3Guard-**Gen**-0.6B |
| `nli` | `deberta-v3-large`, `modernbert-large`, `latest` | same family, different checkpoints — OpenKind pins `typeform/distilbert-base-uncased-mnli`; the MoritzLaurer zeroshot models need a ModernBERT NLI or DeBERTa-v3 readout port |
| `gliclass` | `large`, `latest` | same family, different checkpoint — OpenKind pins `gliclass-modern-base-v3.0`; ollaya pins `gliclass-instruct-large-v1.0` |
| `jevk5` | `4b`, `latest` | supported via alias `jevk5:4b` |
| `decider` | `0.8b`, `2b`, `2b-vision`, `4b`, `latest` | `decider:4b` supported via alias (byte-identical checkpoint); the 0.8b/2b/2b-vision sizes and the moving `latest` tag (2b) need their own profiles |
| `decision` | `eos`, `latest` | not loadable — Decision-1.0-Eos-0.8B (backbone + endpoint head) needs a Qwen3.5 head loader |
| `nimble` | `9b`, `latest` | not loadable — Qwen3.5-9B + Bespoke-Nimble-9B-v2 LoRA; FP32 weights also exceed the 16 GiB per-installation cap |
| `jeeves` | `9b`, `latest` | not loadable — PostHog/jeeves (Qwen3.5-9B + head); FP32 weights also exceed the 16 GiB per-installation cap |
| `clm` | `8b`, `latest` | not loadable — Qwen3-8B + CLM similarity head needs a Qwen3-8B loader |
| `von` | `1.1`, `latest` | supported via alias `von:1.1` — byte-identical checkpoint; the moving `latest` tag is not carried |
| `winnow` | `e4b`, `12b`, `latest` | name collision — ollaya's `winnow` is the EldanRing Gemma-4 fine-tune served as Q8_0 GGUF; OpenKind's `winnow` is the unrelated in-house script router. No alias will map the colliding name until a Gemma-4 GGUF backend exists, and never to a different checkpoint |
| `cygnet` | `12b`, `latest` | not loadable — gemma-4-12B-it Q8_0 GGUF needs a Gemma-4 GGUF backend |
| `jeb` | `4b`, `9b`, `27b`, `latest` | not loadable — the jebadiah GGUF checkpoints need llama.cpp-class GGUF serving |

## Execution backends, size, memory, and measured runs

Per-backend detail for the [All models](#all-models) table above: model-load
time and input token rate per profile and backend, plus the unqualified bf16
probe. CPU fp32 (candle) is the correctness oracle; the MLX backend is
parity-qualified for the profiles listed and does not exist for the others.

| Profile | Backend | Download | Peak RSS | Decisions/s | Input tok/s | Model load | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| `qwen35-state-first` | CPU fp32 | 9.34 GB | 18.62 GB | 0.22 | — | 16.5 s | [registry campaign](benchmarks/2026-09-28-registry-mlx-campaign/README.md) |
| `qwen35-state-first` | MLX fp32 (qualified) | same | 12.09 GB | 2.46 | — | ~21 s | [registry campaign](benchmarks/2026-09-28-registry-mlx-campaign/README.md) |
| `qwen35-state-first` | MLX bf16 (unqualified) | same | — | — | — | — | throughput-only probe; fails the frozen probability tolerance, never decision-safe |
| `decoder-logit-qwen35` | CPU fp32 | 8.43 GB | 8.04 GB | 0.16 | — | 15.2 s | [summary](benchmarks/2026-09-27-decoder-logit-qwen35/summary-decoder-logit-qwen35.json) |
| `decoder-logit-qwen35` | MLX fp32 (qualified) | same | 6.98 GB | 0.43 | — | 20.2 s | [mlx counterparts campaign](benchmarks/2026-09-29-mlx-counterparts/README.md) |
| `kev` | CPU fp32 | 1.25 GB | 4.22 GB | 4.84 | — | 5.6 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-kev.json) |
| `qwen3guard` | CPU fp32 | 1.21 GB | 4.20 GB | 4.97 | — | 2.4 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-qwen3guard.json) |
| `decoder-logit-letter` | CPU fp32 | 0.99 GB | 3.33 GB | 7.22 | — | 2.0 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-decoder-letter.json) |
| `decoder-logit-llm` | CPU fp32 (GGUF q8_0 weights) | 0.68 GB | 1.55 GB | 0.42 | — | 1.5 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-decoder-llm.json) |
| `winnow` | CPU fp32 (router pass) | 1.00 GB | 3.32 GB | 194.6 | — | 2.4 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-winnow.json) |
| `laya-english` | CPU fp32 | 0.85 GB | 2.75 GB | 2.23 | 455 | 1.8 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-english` | MLX fp32 (qualified) | same | 2.12 GB | 24.34 | 4,964 | 2.8 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-multilingual` | CPU fp32 | 0.68 GB | 2.80 GB | 5.30 | 1,075 | 2.0 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-multilingual` | MLX fp32 (qualified) | same | 2.94 GB | 56.37 | 11,431 | 3.5 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-typed-decisions` | CPU fp32 | 0.85 GB | 2.76 GB | 2.33 | 475 | 1.8 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-typed-decisions` | MLX fp32 (qualified) | same | 2.13 GB | 23.77 | 4,849 | 2.9 s | [laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `plumb-4b` | CPU fp32 | 8.43 GB | measured run pending | — | — | — | measured run pending |
| `von` | CPU fp32 | 1.58 GB | 1.83 GB | 2.25 | 493 | 3.3 s | [summary](benchmarks/2026-09-30-von/summary-von.json) |
| `encoder-nli` | CPU fp32 | 0.27 GB | 0.56 GB | 35.14 | — | 0.5 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-encoder-nli.json) |
| `encoder-instruct-label` | CPU fp32 | 0.61 GB | 1.27 GB | 4.73 | 1,013 | 1.2 s | [mlx counterparts campaign](benchmarks/2026-09-29-mlx-counterparts/README.md) (CPU re-run) |
| `encoder-instruct-label` | MLX fp32 (qualified) | same | 1.02 GB | 75.42 | 16,136 | 1.5 s | [mlx counterparts campaign](benchmarks/2026-09-29-mlx-counterparts/README.md) |
| `schema-scorer` | CPU fp32 | 0.09 GB | 0.26 GB | 10.25 | — | 0.2 s | [summary](benchmarks/2026-09-26-surveyed-families/summary-schema-scorer.json) |
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
- The `decoder-logit-qwen35` MLX row is full-forward-per-pass compute (no
  continuation reuse, JSON payloads of 1–2.5k tokens), so its 2.7× gain over
  the recorded CPU row is far below the continuation-aware state-first
  engine's; the two rows are not strategy-comparable.
- `winnow` and `router-script` measure the routing pass over mock siblings,
  not end-to-end decisions.
- The `qwen35-mlx-bf16` engine is an unqualified candidate: it fails the
  frozen probability tolerance (see [`MLX.md`](MLX.md)) and exists as a
  throughput probe only.
- This decision-path table does not include `encoder-embedding`, a BGE
  sentence encoder used by the proxy cache rather than a `DecisionEngine`.
  Its component benchmark and CPU/MLX parity record are listed separately
  below.
- Engines run through `openkind-bench score --engine NAME` (`qwen35`,
  `qwen35-mlx-fp32`, `decoder-letter`, `encoder-nli`, `encoder-instruct-label`,
  `decoder-llm`, `schema-scorer`, `qwen3-guard`, `kev`,
  `decoder-logit-qwen35`, `plumb-4b`, `von`, `laya-english`,
  `laya-multilingual`, `laya-typed-decisions`, `winnow`, `router-script`,
  and the `*-mlx-fp32` variants) or through the daemon (`--installed-models` for catalog profiles,
  `--qwen35-backend mlx-fp32`, `--laya-backend mlx-fp32`,
  `--encoder-instruct-label-backend mlx-fp32`, and
  `--decoder-logit-qwen35-backend mlx-fp32` for the MLX
  paths). Family pages document per-family artifact-path flags.

## Proxy-cache BGE embedding benchmark

`encoder-embedding` is a catalog profile for the proxy-cache state embedder,
not a standalone decision model. Its component benchmark times the same
single-text encode path used per request, including tokenization and excluding
model load. It is separate from the 777-decision `openkind-bench score` table
above.

| Backend | p50 | p95 | Embeddings/s | Peak RSS |
|---|---:|---:|---:|---:|
| Candle CPU FP32 | 26.06 ms | 29.49 ms | 40.52 | 261.2 MiB |
| MLX FP32 | 4.30 ms | 5.69 ms | 220.36 | 261.2 MiB |

These are one dirty-tree M4 Max component run, not semantic quality results.
The [recorded run](benchmarks/2026-09-30-encoder-embedding/README.md) has
checkpoint identity, input digest, CPU/MLX parity, and reproduction commands.

## Ownership and sync

[`registry/v1/catalog.json`](../registry/v1/catalog.json) is the machine
source for catalog names, aliases, profile IDs, and manifest digests; the
[All models](#all-models) table above is its human-readable mirror, and
[`crates/openkind-model-store/tests/models_doc.rs`](../crates/openkind-model-store/tests/models_doc.rs)
fails when pull names, profile IDs, or the [Pull-name aliases](#pull-name-aliases)
table drift in either direction. The same catalog is mirrored to the public
[OpenKind model registry](https://github.com/whit3rabbit/openkind-model-registry),
whose digest is pinned in `openkind-model-store`.
[`families/README.md`](families/README.md) owns the family-to-loader
catalogue and the surveyed-family list. [`MODEL_REGISTRY.md`](MODEL_REGISTRY.md)
owns pull, publication, and mirror verification commands.
[`BENCHMARKS.md`](BENCHMARKS.md) owns measurement methodology. When a change
edits `registry/v1` or lands a new loadable profile, update the tables here
in the same change.
