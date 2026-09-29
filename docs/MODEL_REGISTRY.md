# Curated model registry

`openkind catalog` lists curated profiles available to pull. `openkind pull
NAME` downloads pinned artifacts and verifies every file before installation.
Terminal output shows per-artifact transfer progress, rate, resumed bytes, and
SHA-256 verification. Redirected output prints concise download and
verification milestones. Pull progress stays on stderr. `openkind list`,
`show NAME`, and `rm NAME` operate on local installations without a daemon or
network request. Read commands accept `--json` for scripts.

The initial entry is
`qwen35-state-first:a047d6802c3f06f085b8`. It uses the pinned
`Qwen/Qwen3.5-4B-Base` checkpoint plus the exact exported tokenizer and
score-summary bundle required by the existing Rust loader. Its
`rust-loadable` status describes implementation and parity coverage, not
reviewed task quality or release approval.

```bash
openkind catalog
openkind pull qwen35-state-first:a047d6802c3f06f085b8
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

Benchmark summaries carry the machine-readable comparison data
(`host_hardware`, `context`, per-strategy `cpu_time_seconds` /
`avg_cpu_percent`, peak resident bytes, decisions per second, input tokens
per second) and
[`scripts/build-recommendation-data.py`](../scripts/build-recommendation-data.py)
aggregates them for model-recommendation work. The laya encoder MLX backend
is its own family module (`families/laya/mlx/`) over the same digest-locked
checkpoint contract, with frozen golden-fixture parity gates
(`tests/laya_parity.rs`, module `mlx_replay`); a new MLX backend for any
other encoder family still needs the same structure before a catalog entry
may claim it.

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

The [roadmap](ROADMAP.md) owns M0–M4 and the evidence required for task support.
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
