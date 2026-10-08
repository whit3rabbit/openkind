# Changelog

## Unreleased

### Added

- Pinned survey of the `nativ-community` JEV-27B-VL and GEV-26B-Decide 8-bit MLX conversions
  in `registry/v1/jev-gev-mlx-models.json` (immutable revisions, shard sizes and SHA-256
  digests, `decision_config`, and the mlx-vlm reference commits). Neither is installable.
- `openkind_backends::families::jev_protocol`: hardware-neutral JEV/GEV readout ported from
  the mlx-vlm `feat/jev` and `feat/gev` branches (config validation, prompt rendering,
  choice-label token derivation, bias and per-kind temperature readout, GEV softcap, and the
  more-than-16-option tournament) with reference-value tests. It is not a backbone or engine.

## 0.2.1 (2026-10-07)

### Added

- Model support for Strands Decider 2B (Hobson v21, `StrandsAgents/strands-decider-2B-hobson-v21`
  at revision `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5`, profile ID `f7156bf28400a79ea1b8`)
  with fitted calibration temperatures (`strands_decider_config.json`) and multi-profile
  daemon loading.
- Leaderboard survey metadata in `registry/v1/jev-decision-index.json` indexing the top 10
  open reproductions on the Jev Decision Index (0.2.1).
- Local decision fine-tuning, post-training (RL/PPO/DPO), and release packaging notebooks
  and test validation tooling under `research/`.

### Changed

- Updated curated catalog to 26 models, including Hobson v19 and Hobson v21 profiles.
- Synced `whit3rabbit/openkind-model-registry` with the Hobson v21 profile and updated manifests.

## 0.2.0 (2026-10-04)

### Added

- `openkind batch`: Sequential, resumable JSONL evaluation jobs (`run`, `status`,
  `stop`, `resume`, `export`) with bounded streaming input, line range controls
  (`--start-row`, `--end-row`), a crash-resilient SQLite recovery journal, runner
  file locking, and graceful signal handling.
- `RequestLimits`: Independent failed-authentication rate limiting budget in
  `openkind-api` so unauthorized attempts do not exhaust valid callers' quotas.
- Output file reservation (`reserve_paths`, `reserve_outputs`) in `openkind-bench`
  with exclusive creation prior to model loading and evaluation, preventing
  accidental clobbering of prior benchmark evidence.
- Full prediction ID coverage and canonical score rubric key validation in
  `openkind-bench dataset eval`.
- Observability counters `openkind_request_outcomes_total` (fixed outcome labels)
  and `openkind_auth_failures_total` (fixed transport label), with fractional
  millisecond durations in `openkind_request_duration_ms`.
- Host hardware reports `available_parallelism` separately from OS `logical_cores`.
- `ClientBuilder::allow_unauthenticated()` in `openkind-client` for keyless daemons.
- Unattended Colab T4 NF4 FP16 pilot training notebook and generator with
  single-dimension parameter sweeps (`learning_rate`, `rank`).
- Verification workflow for published archive attestations, fresh Cargo
  installations, and Homebrew installations on macOS and Linux.

### Changed

- Minimum supported Rust version raised from 1.88 to 1.90.
- `openkind serve` and `openkindd` default HTTP and gRPC bind addresses to
  loopback (`127.0.0.1:8080` / `127.0.0.1:9090`) rather than `0.0.0.0`.
- Graceful shutdown handles a second `SIGINT` or `SIGTERM` to force an immediate
  nonzero exit while the first signal drains active in-flight requests.
- Proxy-cache verified key registry sweeps expired credentials on admission and
  caps in-memory entries at 4,096 keys.
- Proxy-cache sample store pruning bounds `co_deferred` diagnostic rows
  independently using the training budget, preserving IID calibration capacity.
- Model and dataset stores verify existing cached blobs before reuse, unlinking
  corrupt blobs for clean refetching while clearing abandoned `.stage-*`
  directories under lock.
- Explicit store directories must be nonempty; empty platform fallback variables
  are safely ignored.
- Macro-F1 in dataset evaluation averages only classes with gold support in the split.
- `ApiError` display formatting bounds error message strings to 200 characters
  with ellipsis while retaining the full raw message in programmatic fields.

### Fixed

- Score rubric index validation in `openkind-core` rejects non-canonical integer
  string keys (`+1`, `01`, `007`).
- Von decision Python repr rendering preserves the full 64-bit unsigned integer range.
- Client retry-after header parsing preserves 64-bit integer millisecond bounds
  without float precision loss and safely ignores overflowing values.
- gRPC overload responses include `retry-after` and `retry-after-ms` metadata keys.
- Backend response validation failures surface as `EngineError::BackendValidation`
  mapped to HTTP 500 / gRPC Internal.
- Documented OpenAPI error response schemas (`413`, `502`, `504`, `529`) are
  executable and tested on both `/v1/systemone` and `/v1/system_one`.

## 0.1.0 (2026-10-03)

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
- CUDA family loader error conversion and an unused backbone import.
- Repeated ONNX initialization after a missing-library failure. Repair the
  runtime path and restart the process before retrying.
- Binding daemon lifecycle, request snapshots, packaged license metadata,
  probability validation, and authentication forwarding.
