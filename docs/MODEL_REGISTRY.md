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
Manifests identify a compiled-in loader and immutable source revisions; they
cannot provide executable code. Source weights remain on their authors'
repositories. Downloads and tests are separate: builds and tests never fetch
model assets. A real Qwen pull and decision smoke test is an explicit,
multi-gigabyte operator gate.

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
