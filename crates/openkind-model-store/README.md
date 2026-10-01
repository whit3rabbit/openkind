# openkind-model-store

> Curated, verified local model installations for OpenKind.

`openkind-model-store` owns the curated model catalog, explicit verified downloads, the content-addressed local store, and the installation locks behind `openkind pull` and `openkind serve --installed-models`. It never runs inference and never changes Jev wire types. Only the explicit pull path touches the network: daemon startup, local reads, builds, and tests never download model assets.

## Using it

Operators reach this crate through the `openkind` binaries (see the [CLI guide](../openkind-cli/README.md) for installation):

```bash
openkind catalog
openkind pull laya-english:c8ea29bf1e33a343c4b7
openkind serve --installed-models laya-english:c8ea29bf1e33a343c4b7
```

`openkind-cli` drives the pull flow. `openkind-server` reads installations offline through the serving-lock path, which re-verifies every artifact before the daemon serves it and holds the lock for the daemon's lifetime, so a serving model cannot be removed underneath it.

## How a pull works

1. **Pin.** The catalog is fetched from the public [openkind-model-registry](https://github.com/whit3rabbit/openkind-model-registry) mirror and checked against a digest compiled into this crate, so a mutable mirror cannot authorize new content. Each catalog entry pins its manifest by SHA-256, and each manifest pins every artifact's source revision, byte size, and digest.
2. **Download.** Artifacts land as partial files inside the store and resume over HTTP byte ranges after an interruption. Manifests matching no loader compiled into this build are refused before any artifact downloads.
3. **Verify.** Size and SHA-256 are checked before an installation becomes visible. A mismatch deletes the partial file and fails the pull.
4. **Install.** Verified blobs are hard-linked into a staged snapshot on the same filesystem, never staged through `/tmp`, and renamed into place. Blobs shared by two models survive until both are removed.

`registry/v1` in this repository owns the metadata; the mirror carries that metadata plus small profile assets, while checkpoint weights remain at their authors' repositories. The [registry guide](../../docs/MODEL_REGISTRY.md) covers operator commands and evidence, and [MODELS.md](../../docs/MODELS.md) indexes every loadable profile.

## Names and aliases

A pull name is `family:version`, such as `laya-english:c8ea29bf1e33a343c4b7`. A catalog entry may also declare `name:tag` aliases like `laya:en` or `jevk5:4b`, which resolve to the canonical name at pull time. Installations stay keyed by the canonical name, and `list`, `show`, `rm`, and `--installed-models` accept canonical names only.

Windows forbids `:` in filenames, so install directories, stage directories, and lock files spell names with `@` in place of `:`. Unix keeps the raw spelling for stores that already exist.

## Storage layout

The store root holds shared verified blobs, one directory per installation, and lock files:

```text
blobs/sha256/<digest>   shared, digest-verified artifacts
models/<name>/          hard-linked snapshot plus manifest.json
locks/                  global and per-model file locks
```

The root defaults to the platform data directory. Set `OPENKIND_MODELS_DIR` or the `--models-dir` flag to share one store between the CLI and the daemon.

## Scope and limits

`rust-loadable` support status describes implementation and parity coverage, not reviewed task quality or release approval. One installation is capped at 16 GiB of artifacts, remote catalog and manifest responses at 4 MiB. Downloads follow only HTTPS redirects, partial-file opens refuse symlink substitution, and concurrent pulls of the same model fail fast on its lock rather than racing.

## Testing

```bash
cargo test -p openkind-model-store
```

Tests run against a local mock server with in-memory fixtures. Nothing downloads model assets.

## License

See the [MIT license](../../LICENSE). Cargo metadata declares `MIT OR Apache-2.0`.
