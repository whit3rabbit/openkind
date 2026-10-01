# Curated model registry

`openkind catalog` lists curated profiles available to pull. `openkind pull
NAME` downloads pinned artifacts and verifies every file before installation.
Terminal output shows per-artifact transfer progress, rate, resumed bytes, and
SHA-256 verification. Redirected output prints concise download and
verification milestones. Pull progress stays on stderr. `openkind list`,
`show NAME`, and `rm NAME` operate on local installations without a daemon or
network request. Read commands accept `--json` for scripts.

Sixteen profiles are catalog-installable: the native Qwen3.5 state-first
profile, the three laya decision encoders, and the surveyed-family
prototypes (`decoder-logit-letter`, `encoder-nli`, `encoder-instruct-label`,
`decoder-logit-llm`, `schema-scorer`, `qwen3guard`, `kev`,
`decoder-logit-qwen35`, `plumb-4b`, `von`, `winnow`), plus the proxy-cache `encoder-embedding`
sentence encoder. Every manifest pins each artifact's
source revision, byte size, and SHA-256; `rust-loadable` status describes
implementation and parity coverage, not reviewed task quality or release
approval. Two small derived assets that no upstream publishes — kev's
converted `head.safetensors` and the in-house winnow LoRA adapter — are
pinned as `github` assets in the public mirror at the commit recorded in
their manifests.

Pull names follow the `loader-id:profile-digest` schema. A catalog entry may
also declare `name:tag` aliases — the schema the
[ollaya decision-model registry](https://github.com/ollaya-dev/ollaya) uses —
so names transfer between users of both tools. An alias resolves to the
canonical pull name at pull time; installations stay keyed by the canonical
name, and `list`, `show`, `rm`, and `--installed-models` accept canonical
names only. Aliases are added only where the checkpoint identity matches,
never for `latest` or precision tags; [`MODELS.md`](MODELS.md) owns the alias
table and the full ollaya coverage matrix, both pinned by
[`models_doc.rs`](../crates/openkind-model-store/tests/models_doc.rs).

```bash
openkind catalog
openkind pull qwen35-state-first:a047d6802c3f06f085b8
openkind pull laya:en
openkind list
openkind show qwen35-state-first:a047d6802c3f06f085b8
openkind serve --installed-models qwen35-state-first:a047d6802c3f06f085b8
```

The daemon loads explicitly named installations at startup. A new pull needs
a restart or an explicit **Load model** action in the opt-in playground.
`GET /v1/models` lists only aliases served by that process; `openkind status` displays those registered aliases. Catalog
profiles are available to pull, installed profiles are local files, and
registered aliases are active in the current daemon. The existing `--models`
mock aliases and explicit `--qwen35-*` artifact-path configuration remain
available. Name collisions fail startup.

With `--playground on`, the local model panel lists installed profiles and
configured aliases. It can load verified supported installations and load or
unload mock aliases. Native engines configured with artifact-path flags and
composite engines still require a restart. Loading never downloads assets.

Unloading stops new requests; accepted work keeps its engine until completion.
Installation file locks remain until daemon exit because native tasks can
outlive a request timeout. Stop the daemon before using `openkind rm` on these
installations. Playground controls use the configured bearer key.

Use `--models-dir` or `OPENKIND_MODELS_DIR` to share a store location between
the CLI and daemon. The default is the user's platform data directory.

The versioned catalog and manifests live under [`../registry/v1/`](../registry/v1/).
The operator-facing model index that mirrors them is
[`MODELS.md`](MODELS.md); update both together when the catalog changes.
Manifests identify a compiled-in loader and immutable source revisions; they
cannot provide executable code. Source weights remain on their authors'
repositories. Downloads and tests are separate: builds and tests never fetch
model assets. A real Qwen pull and decision smoke test is an explicit,
multi-gigabyte operator gate.

## Execution backends and benchmark evidence per registry model

Each catalog model records which execution backends can serve it on Apple
silicon and where the measured evidence lives. An MLX conversion of a
backbone that openkind cannot load is not an executable equivalent; the
registry model is only as fast as the loaders in this repository.

| Registry model | Executable backends | Preferred on Apple silicon | Evidence |
|---|---|---|---|
| `qwen35-state-first:a047d6802c3f06f085b8` | Candle CPU fp32 (`--engine qwen35`), MLX FP32 (`--engine qwen35-mlx-fp32`, `--features mlx`), MLX BF16 candidate (unqualified) | `qwen35-mlx-fp32` — a parity-qualified MLX path | [BENCHMARKS.md](BENCHMARKS.md) records and the [2026-09-28 registry campaign](benchmarks/2026-09-28-registry-mlx-campaign/README.md) |
| `laya-english:c8ea29bf1e33a343c4b7` | Candle CPU fp32 (`--engine laya-english`), MLX FP32 (`--engine laya-english-mlx-fp32`, `--features mlx`; daemon `--laya-backend mlx-fp32`) | `laya-english-mlx-fp32` — golden-fixture parity gates on the same pinned shard (max probability drift 6.5e-6, zero selection flips) | [BENCHMARKS.md](BENCHMARKS.md) records, the [2026-09-28 registry campaign](benchmarks/2026-09-28-registry-mlx-campaign/README.md), and the [2026-09-28 laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-multilingual:f4064eb56fb7f7d325e1` | Candle CPU fp32 (`--engine laya-multilingual`), MLX FP32 (`--engine laya-multilingual-mlx-fp32`, `--features mlx`) | `laya-multilingual-mlx-fp32` — same encoder MLX path (max probability drift 7.2e-6, zero selection flips) | [2026-09-28 laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `laya-typed-decisions:9d28cfa9567902801ed1` | Candle CPU fp32 (`--engine laya-typed-decisions`), MLX FP32 (`--engine laya-typed-decisions-mlx-fp32`, `--features mlx`) | `laya-typed-decisions-mlx-fp32` — same encoder MLX path (max probability drift 2.5e-6, zero selection flips) | [2026-09-28 laya MLX campaign](benchmarks/2026-09-28-laya-mlx-campaign/README.md) |
| `decoder-logit-letter:5492c97dfcdaf3fe9439` | Candle CPU fp32 (`--engine decoder-letter`; daemon `--decoder-letter-aliases` / `--decoder-letter-model-root`) | CPU fp32 — no MLX path | [summary](benchmarks/2026-09-26-surveyed-families/summary-decoder-letter.json) |
| `encoder-nli:1041a4c362338a61b820` | Candle CPU fp32 (`--engine encoder-nli`; daemon `--encoder-nli-aliases` / `--encoder-nli-model-root`) | CPU fp32 — no MLX path | [summary](benchmarks/2026-09-26-surveyed-families/summary-encoder-nli.json) |
| `encoder-instruct-label:9fd68313a5606eca42f2` | Candle CPU fp32 (`--engine encoder-instruct-label`; daemon `--encoder-instruct-label-aliases` / `--encoder-instruct-label-model-root` / `--encoder-instruct-label-backend mlx-fp32`), MLX FP32 (`--engine encoder-instruct-label-mlx-fp32`, `--features mlx`) | `encoder-instruct-label-mlx-fp32` — golden-fixture parity gates on the same pinned FP32 shard (max probability drift 4.487e-6, zero selection flips) | [BENCHMARKS.md](BENCHMARKS.md) records and the [2026-09-29 mlx counterparts campaign](benchmarks/2026-09-29-mlx-counterparts/README.md) |
| `decoder-logit-llm:465963d705b6f35d6208` | Candle CPU fp32 over GGUF q8_0 weights (`--engine decoder-llm`; daemon `--decoder-llm-aliases` / `--decoder-llm-model-root`) | CPU fp32 — no MLX path | [summary](benchmarks/2026-09-26-surveyed-families/summary-decoder-llm.json) |
| `schema-scorer:5a7350af556f0ee66566` | Candle CPU fp32 (`--engine schema-scorer`; daemon `--schema-scorer-aliases` / `--schema-scorer-model-root`) | CPU fp32 — no MLX path | [summary](benchmarks/2026-09-26-surveyed-families/summary-schema-scorer.json) |
| `qwen3guard:0fcf416cab16d94f933d` | Candle CPU fp32 (`--engine qwen3-guard`; daemon `--qwen3guard-aliases` / `--qwen3guard-model-root`) | CPU fp32 — no MLX path | [summary](benchmarks/2026-09-26-surveyed-families/summary-qwen3guard.json) |
| `kev:39d88c11faeb4ac165fa` | Candle CPU fp32 (`--engine kev`; daemon `--kev-aliases` / `--kev-model-root` / `--kev-base-root`) | CPU fp32 — no MLX path | [summary](benchmarks/2026-09-26-surveyed-families/summary-kev.json) |
| `decoder-logit-qwen35:415bcf4a064e6dadcf85` | Candle CPU fp32 over BF16 checkpoint (`--engine decoder-logit-qwen35`; daemon `--decoder-logit-qwen35-aliases` / `--decoder-logit-qwen35-model-root` / `--decoder-logit-qwen35-backend mlx-fp32`), MLX FP32 over the same BF16 checkpoint widened on load (`--engine decoder-logit-qwen35-mlx-fp32`, `--features mlx`) | `decoder-logit-qwen35-mlx-fp32` — golden-fixture parity gates through the Qwen3.5 MLX backbone (max probability drift 1.003e-6, zero selection flips) | [BENCHMARKS.md](BENCHMARKS.md) records, the [2026-09-29 mlx counterparts campaign](benchmarks/2026-09-29-mlx-counterparts/README.md), and the CPU [summary](benchmarks/2026-09-27-decoder-logit-qwen35/summary-decoder-logit-qwen35.json) |
| `plumb-4b:c1f080794d38e94a0bc2` | Candle CPU fp32 over BF16 checkpoint (`--engine plumb-4b`; daemon `--plumb-4b-aliases` / `--plumb-4b-model-root` / `--decoder-logit-qwen35-backend mlx-fp32`), MLX FP32 over the same BF16 checkpoint widened on load (`--engine plumb-4b-mlx-fp32`, `--features mlx`) | `plumb-4b-mlx-fp32`: reported golden-fixture replay through the Qwen3.5 MLX backbone (max probability drift 1.274e-5, zero selection flips; replay logs unarchived); CPU smoke + choice diagnostic in the [2026-09-30 jevbench expansion](benchmarks/2026-09-30-jevbench-expansion/README.md) | [2026-09-30 jevbench expansion](benchmarks/2026-09-30-jevbench-expansion/README.md) |
| `von:69219703407bd39cca0c` | Candle CPU fp32 over the author's `option_marker.pt` pickle (`--engine von`; daemon `--von-aliases` / `--von-model-root`) | CPU fp32 — no MLX path | [summary](benchmarks/2026-09-30-von/summary-von.json) |
| `winnow:4dff8c5b03cfbf680db6` | Candle CPU fp32 router over registered siblings (daemon `--winnow-aliases` / `--winnow-model-root` / `--winnow-adapter`) | CPU fp32 — no MLX path; installed winnow binds label `A` to the installed `decoder-logit-letter` profile and label `B` to `encoder-nli` (falling back to the `--models` aliases) | [summary](benchmarks/2026-09-26-surveyed-families/summary-winnow.json) |
| `encoder-embedding:8d9498269ef05d95d93c` | Candle CPU fp32 and MLX FP32 (`--proxy-cache-encoder-backend mlx-fp32`, macOS arm64); embedding only, not a `DecisionEngine` | MLX FP32 measured 5.4x CPU throughput in one M4 Max component run; CPU remains the numerical oracle | [encoder parity and component benchmark](benchmarks/2026-09-30-encoder-embedding/README.md) |

Benchmark summaries carry the machine-readable comparison data
(`host_hardware`, `context`, per-strategy `cpu_time_seconds` /
`avg_cpu_percent`, peak resident bytes, decisions per second, input tokens
per second) and
[`scripts/build-recommendation-data.py`](../scripts/build-recommendation-data.py)
aggregates them for model-recommendation work. Surveyed-family MLX backends
are family modules over the same digest-locked checkpoint contract
(`families/laya/mlx/`, `families/encoder_instruct_label/mlx/`,
`families/decoder_logit_qwen35/mlx/`), with frozen golden-fixture parity
gates (each family's parity test, module `mlx_replay`). The ModernBERT
encoder families share one MLX body
(`families/mlx_modernbert.rs`); the JevK5 decoder reuses the parity-verified
Qwen3.5 MLX backbone through the `MlxSurveyCheckpoint` descriptor. A new MLX
backend for any other family still needs the same structure before a catalog
entry may claim it.

## Publishing the public mirror

OpenKind's checked-in [`registry/v1`](../registry/v1/) is the metadata source.
The separate public
[`whit3rabbit/openkind-model-registry`](https://github.com/whit3rabbit/openkind-model-registry)
repository serves the same catalog over raw HTTPS and holds only the small
profile bundle and tokenizer. It does not contain checkpoint shards. The CLI's
catalog URL points to that public repository, so OpenKind can stay private
while anonymous pulls work. The [sync script](../scripts/sync-model-registry.py)
lives in OpenKind; there is no automatic cross-repository push.

For a new profile, first commit its distributable assets to the public
registry. Pin that asset commit, file sizes, and SHA-256 values in the OpenKind
manifest, then update the catalog's manifest digest. The local parity fixtures
and loader must pass before catalog publication. From the OpenKind checkout,
use a local checkout of the public registry at the script's default path:

```bash
python3 scripts/sync-model-registry.py --write
# Review, commit, and push registry/v1 in the public registry checkout.
python3 scripts/sync-model-registry.py --remote
```

Without flags, the script checks local copies. `--write` copies only catalog
and manifest files, refuses a dirty mirror metadata tree, and never commits
or pushes. After the public push, `--remote` checks exact catalog and manifest
bytes and all pinned profile assets through public HTTPS. It does not fetch
checkpoint shards. The default mirror checkout is
`~/Documents/GitHub/openkind-model-registry`; pass `--mirror PATH` to each
command if the public checkout is elsewhere.
Keep prior asset commits available so existing manifests remain reproducible.

## Family expansion

The research evidence and qualification gates (M0–M4) documented in
[`RESEARCH.md`](RESEARCH.md) and [`whitepaper/WHITEPAPER.md`](whitepaper/WHITEPAPER.md)
govern task support.
After those gates, qualify one profile at a time: encoder NLI, GLiClass-style
label scoring, Laya-style decision encoders, then decoder-logit profiles where
the supported workload calls for them. Each needs its own checkpoint and
tokenizer revision, renderer, readout, calibration, offline parity fixtures,
and local loader. A language router requires two qualified target profiles.
Kev and Qwen3Guard have since cleared their blockers as Rust-loadable
prototype profiles; the Laya-style decision encoders landed 2026-09-27 as
three registry-installable rust-loadable profiles (`laya-english`,
`laya-multilingual`, `laya-typed-decisions`) with reference-parity fixtures;
Von remains surveyed and external-reference-only.
A family page or catalog description does
not make a model runnable.
