# Curated model registry

`openkind catalog` lists the curated profiles available to this OpenKind
release. `openkind pull NAME` downloads pinned artifacts and verifies every
file before installation. `openkind list`, `show NAME`, and `rm NAME` operate
on local installations without a daemon or network request. Pull progress
reports bytes present for each artifact and the final line reports bytes
actually downloaded in that invocation.

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
a restart to become available. `GET /v1/models` lists only aliases served by
that process. The existing `--models` mock aliases and explicit `--qwen35-*`
artifact-path configuration remain available. Name collisions fail startup.
Use `--models-dir` or `OPENKIND_MODELS_DIR` to share a store location between
the CLI and daemon. The default is the user's platform data directory.

The versioned catalog and manifests live under [`../registry/v1/`](../registry/v1/).
Manifests identify a compiled-in loader and immutable source revisions; they
cannot provide executable code. Source weights remain on their authors'
repositories. Downloads and tests are separate: builds and tests never fetch
model assets. A real Qwen pull and decision smoke test is an explicit,
multi-gigabyte operator gate.

## Family expansion

The [roadmap](ROADMAP.md) owns M0–M4 and the evidence required for task support.
After those gates, qualify one profile at a time: encoder NLI, GLiClass-style
label scoring, Laya-style decision encoders, then decoder-logit profiles where
the supported workload calls for them. Each needs its own checkpoint and
tokenizer revision, renderer, readout, calibration, offline parity fixtures,
and local loader. A language router requires two qualified target profiles.
Kev, Von, and Qwen3Guard remain surveyed until rights, contract, and
architecture blockers are resolved. A family page or catalog description does
not make a model runnable.
