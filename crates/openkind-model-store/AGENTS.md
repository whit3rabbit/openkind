# openkind-model-store

This crate owns curated model metadata, explicit downloads, the content-addressed
local store, and installation locks. It does not implement model inference or
change Jev wire types. `openkind-cli` is its online caller; `openkind-server`
uses its offline read and serving-lock path.

- Only immutable, checked-in catalog entries with compiled-in loaders are
  installable. The catalog and manifest are separate from `EngineRegistry`.
- OpenKind's `registry/v1` owns metadata. The public
  `whit3rabbit/openkind-model-registry` repository mirrors that metadata and
  hosts small profile assets. Use `scripts/sync-model-registry.py` from the
  OpenKind root to copy and verify the mirror; it does not push changes.
- Never download during construction, serving startup, builds, or tests.
  Downloads happen only under the explicit `openkind pull` command.
- Verify artifact size and SHA-256 before making an installation visible.
  Keep partial downloads in the store and resume them by byte range.
- Never copy multi-gigabyte checkpoint shards into `/tmp`. The store links
  verified blobs into a staged snapshot on the same filesystem.
- Keep serving locks alive for the daemon lifetime so `rm` cannot remove an
  active model. Never log credentials, request content, or token digests.
- Model names are `family:version`, but `:` is illegal in Windows filenames:
  on-disk spellings (install directories, stage directories, lock files)
  replace it with `@` on Windows and keep the raw name on Unix, where
  existing stores already use it. `list()` applies the inverse mapping.
- `rust-loadable`, task qualification, and release promotion are distinct.

Run `cargo test -p openkind-model-store` and the workspace verification battery
after changes to manifests or store behavior.
