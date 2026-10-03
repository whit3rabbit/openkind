# Changelog

## Unreleased

- No release has been published yet. Complete the [release checklist](release.md)
  against the final `v0.1.0` commit before assigning a release date.

## 0.1.0 (planned)

### Added

- Typed Jev HTTP and gRPC interfaces for Noul, Choice, and Score decisions,
  with request-bound answer validation.
- `openkind` CLI and `openkindd` daemon, curated model installation, local
  playground controls, backend diagnostics, and a Rust client SDK.
- Native CPU execution, optional MLX and CUDA backends, optional ONNX execution,
  and Linux ROCm provider selection. Backend compilation and discovery have
  separate runtime and model qualification gates.
- Pinned evaluation-dataset installation and a native benchmark harness.
- TypeScript, Python, and Swift HTTP clients and daemon wrappers available from
  source. npm and PyPI packages remain unpublished.
- Native API-key generation in the CLI and binding wrappers.
- Standard and ONNX release archive automation, checksums, relocated archive
  checks, and Homebrew packaging.
- A release checklist and serial crates.io publisher with version validation,
  per-crate checksums, upload timing, and resumable reports.

### Changed

- Rust crates use one workspace version and the repository's MIT license.
  Every crate archive includes the license text.
- API-key configuration rejects empty, invalid, or conflicting values before
  model loading. Remove unused migration aliases from the environment rather
  than setting them to empty. See [ENV.md](docs/ENV.md).
- Release publication waits for CI and archive verification on the same commit.
  GitHub assets and Homebrew updates follow successful crate publication.

### Fixed

- macOS release command handling under Bash 3.2, Windows MSVC linker selection,
  and Linux CI/CUDA installation disk pressure.
- Windows dataset and model staging renames while metadata files remained open.
- Daemon startup stack overflow in Windows debug builds.
- Proxy-cache integration assertions that treated random upstream audits as failures.
- ROCm backend test compilation and missing optional-feature CI coverage.
- Repeated ONNX initialization after a missing-library failure. Repair the
  runtime path and restart the process before retrying.
- Binding daemon lifecycle, request snapshots, packaged license metadata,
  probability validation, and authentication forwarding.
