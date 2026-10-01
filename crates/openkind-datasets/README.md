# openkind-datasets

Pinned, verified local evaluation-dataset installations for OpenKind
benchmarking. The crate downloads labeled public datasets from the Hugging
Face Hub at exact pinned revisions into a per-user cache outside the
repository, checks every shard digest, and reads installed parquet back as
JSON rows. `openkind-bench` is the only consumer; `dataset eval` turns those
installs into model-quality evidence.

## Downloads are explicit

Building this crate, running its tests, or reading an installed dataset
never touches the network. The only code paths that fetch anything are the
operator commands `openkind-bench dataset pull` and `dataset pin`
([src/store.rs](src/store.rs)). Download tests run against a local
in-process server.

Dataset bytes are never committed to this repository and never
redistributed. The checked-in [registry](registry/v1/datasets.json) pins
repository identities, revisions, sizes, and digests only.

## Pull, build, eval

Datasets are consumed through [openkind-bench](../openkind-bench/README.md):

```bash
cargo run -p openkind-bench -- dataset list
cargo run -p openkind-bench -- dataset pull sst2
cargo run -p openkind-bench -- dataset build sst2 --split dev
cargo run -p openkind-bench -- dataset eval sst2 --engine mock --output-dir bench-output
```

`build` materializes a labeled workload JSONL (`--split dev` for prompt and
threshold work, `--split eval` to report). `eval` scores it through the
normal request path and reports accuracy with provenance as an
`openkind-dataset-eval/v1` report.

`dataset verify <name>` re-checks every installed digest on demand, and
`dataset rm <name>` clears an install. Methodology and interpretation rules
live in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md#dataset-accuracy-evaluation).

## The curated set

Twelve datasets, all fully open. Gated or non-commercial datasets stay out
until a deliberate decision adds them. Several upstream cards leave their
license unstated, and those entries are marked internal-evaluation-only.

| Primitive | Datasets |
|---|---|
| `choice` | sst2, ag_news, banking77, clinc150, arc (Challenge and Easy configs), hellaswag, winogrande, commonsense_qa |
| `noul` | paws, boolq |
| `score` | stsb, sst5 |

Names, Hugging Face repositories, splits, configs, and licenses are defined
in [src/definitions.rs](src/definitions.rs).

## Pinning and verification

Each registry entry pins two 40-hex revisions — the repository `main` commit
and the `refs/convert/parquet` commit — plus the size and SHA-256 of every
parquet shard ([src/registry.rs](src/registry.rs)). Both are enforced at
validation and download time. A digest mismatch aborts the install rather
than storing different bytes.

Maintainers refresh entries with `dataset pin`, which resolves current
upstream revisions, installs, and prints the registry entry for review.
Committing it stays a human step. Request templates are ported from the
MIT-licensed jev-benchmarking harness (arXiv 2609.37647) at the reviewed
commit recorded in each entry.

## Where installs live

Installs go to the per-OS openkind data directory —
`~/Library/Application Support/openkind/datasets` on macOS,
`${XDG_DATA_HOME:-~/.local/share}/openkind/datasets` on Linux, or
`%APPDATA%\openkind\datasets` on Windows — overridable with
`OPENKIND_DATASETS_DIR`. Shards are stored once in content-addressed blobs
and hard-linked into installs.

Environment variables, including Hugging Face token resolution for gated
repositories, are documented in
[docs/ENV.md](../../docs/ENV.md#benchmark-dataset-variables).

## License

See the [MIT license](../../LICENSE). Cargo metadata declares `MIT OR Apache-2.0`.
