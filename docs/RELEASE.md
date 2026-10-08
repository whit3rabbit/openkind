# Release Guide & Verification Checklist

This document is the standard, repeatable checklist and operating guide for
preparing, verifying, and publishing an `openkind` release (`v<VERSION>`).

Rust crates in this workspace share one lockstep version declared in the root
`Cargo.toml`. npm and PyPI packages remain source-distributed, and the Swift
package tracks repository Git tags. Binary distribution archives, the Homebrew
tap formula, and crates.io packages are released together.

---

## 1. Pre-Release Hygiene & Commit Preparation

- [ ] **Clean Git Environment**:
  - Ensure the active branch is `main` and up to date with `origin/main`.
  - Prune stale worktrees and deleted remote-tracking branches:
    ```bash
    git worktree prune
    git fetch --prune --all
    ```
  - Verify `git worktree list` only shows the active workspace.
- [ ] **Commit Functional Work Logically First**:
  - Commit all features, fixes, and docs in focused, logical commits before
    touching release versions or packaging metadata.
  - Never mix substantive application logic changes with version bumps.
- [ ] **Update Root `CHANGELOG.md`**:
  - Keep the changelog strictly named [`CHANGELOG.md`](../CHANGELOG.md) in the
    repository root (avoid spaces or non-standard variations).
  - Collect unreleased changes under `## <VERSION> (planned)` or
    `## <VERSION> (YYYY-MM-DD)` with standard categories:
    - `### Added`
    - `### Changed`
    - `### Fixed`
- [ ] **Lockstep Version Bump**:
  - Set `version = "<VERSION>"` in root [`Cargo.toml`](../Cargo.toml) under
    `[workspace.package]`.
  - Update all internal crate dependency versions under `[workspace.dependencies]`
    to match `"<VERSION>"`:
    - `openkind-core`
    - `openkind-client`
    - `openkind-api`
    - `openkind-engine`
    - `openkind-runtime`
    - `openkind-backends`
    - `openkind-model-store`
    - `openkind-datasets`
    - `openkind-proto`
  - Update API spec version: [`crates/openkind-api/openapi.yaml`](../crates/openkind-api/openapi.yaml) (`info.version`).
  - Update client binding packages to align versions:
    - [`bindings/typescript/package.json`](../bindings/typescript/package.json) (`version`)
    - [`bindings/python/pyproject.toml`](../bindings/python/pyproject.toml) (`version`)
  - Run `cargo check` to update and synchronize [`Cargo.lock`](../Cargo.lock).
  - Update the published version shown in [`README.md`](../README.md), including
    the pinned `cargo install` command and the `openkind-client` dependency
    example. Keep these at the version being prepared for the tag.
- [ ] **Verify License Inheritance**:
  - Confirm every published crate manifest inherits the workspace MIT license
    and `license-file = "LICENSE"`.
- [ ] **Verify External Asset Mirrors**:
  - Confirm all pinned model profiles and dataset digests referenced in
    [`registry/v1`](../registry/v1) exist in the public mirror
    (`whit3rabbit/openkind-model-registry`). Builds and tests must never download
    weights at runtime.

---

## 2. Local Verification Battery

Run the complete project-specific verification battery from a clean checkout:

```bash
# Code formatting
cargo fmt --check

# Strict workspace lints (0 warnings allowed)
cargo clippy --workspace --all-targets --locked -- -D warnings

# Full workspace unit and integration test suite
env -u RUST_LOG cargo test --workspace --locked

# Benchmark compilation and test battery
env -u RUST_LOG cargo test --workspace --benches --locked

# Minimum Supported Rust Version check (Rust 1.90)
cargo +1.90.0 check --workspace --all-targets --locked

# JSON Schema generation and drift check
cargo run -p openkind-gen-schemas --locked -- --write
git diff --exit-code -- crates/openkind-core/schemas

# Git whitespace and diff validation
git diff --check

# Script and release-tooling unit tests
python3 -m unittest discover -s scripts/tests

# Language binding test suites
(cd bindings/typescript && npm install && npm test)
(cd bindings/python && python3 -m unittest discover -s tests)
(cd bindings/swift && swift test)

# Crate packaging dry-run & digest verification (requires network for sparse index)
python3 scripts/publish-crates.py plan --tag v<VERSION> --verify --output dist/crate-versions.json
```

- [ ] All verification commands pass without warnings or errors.
- [ ] Schemas under `crates/openkind-core/schemas` remain unchanged.
- [ ] Package verification passes for all 12 publishable crates.

> [!NOTE]
> When executing integration tests with mock servers (e.g. `openkind-cli` batch tests)
> under heavy background load, limit test threads (`-- --test-threads=4`) to
> prevent mock socket port contention.

---

## 3. Crate Dependency Order & Publishing Rules

The workspace publisher ([`scripts/publish-crates.py`](../scripts/publish-crates.py))
publishes crates serially in strict topological dependency order:

| Order | Package | Version | Upload Interval |
|---:|---|---|---|
| 1 | `openkind-core` | `<VERSION>` | 600s if new, 60s for update |
| 2 | `openkind-datasets` | `<VERSION>` | 600s if new, 60s for update |
| 3 | `openkind-model-store` | `<VERSION>` | 600s if new, 60s for update |
| 4 | `openkind-proto` | `<VERSION>` | 600s if new, 60s for update |
| 5 | `openkind-engine` | `<VERSION>` | 600s if new, 60s for update |
| 6 | `openkind-runtime` | `<VERSION>` | 600s if new, 60s for update |
| 7 | `openkind-api` | `<VERSION>` | 600s if new, 60s for update |
| 8 | `openkind-backends` | `<VERSION>` | 600s if new, 60s for update |
| 9 | `openkind-cli` | `<VERSION>` | 600s if new, 60s for update |
| 10 | `openkind-bench` | `<VERSION>` | 600s if new, 60s for update |
| 11 | `openkind-client` | `<VERSION>` | 600s if new, 60s for update |
| 12 | `openkind-server` | `<VERSION>` | 600s if new, 60s for update |

- `openkind-gen-schemas` tracks `<VERSION>` with `publish = false`.
- The CLI and server packages install binary targets `openkind` and `openkindd`.
- **Publisher Invariant**: The publisher requires a clean git working tree
  matching the tagged commit. It stages archives to verify checksums before
  uploading and waits for crates.io index propagation between crates.

---

## 4. Tagging & CI Pipeline

- [ ] **Commit Release Version**:
  - Stage the release files (`Cargo.toml`, `Cargo.lock`, `CHANGELOG.md`, `openapi.yaml`,
    `release.md` / `docs/RELEASE.md`, bindings manifests).
  - Commit with message: `release: bump workspace crates to v<VERSION>`
  - Push commit: `git push origin main`
- [ ] **Tag and Push**:
  - Create annotated tag: `git tag -a v<VERSION> -m "Release v<VERSION>"`
  - Push tag to origin: `git push origin v<VERSION>`
- [ ] **CI & Packaging Workflows**:
  - The [CI workflow](https://github.com/whit3rabbit/openkind/actions/workflows/ci.yml)
    passes across Linux, macOS, and Windows.
  - The [Release workflow](https://github.com/whit3rabbit/openkind/actions/workflows/release.yml)
    runs on the tag, executes backend archive packaging, publishes crates in order,
    attaches GitHub release archives, and updates Homebrew tap.
- [ ] **Binary Packaging Smoke Checks**:
  - macOS arm64 archive bundles `mlx.metallib` and passes the MLX GPU probe.
  - ONNX archives include bundled ONNX Runtime libraries and run the Gemm probe.
  - Linux and Windows binaries start and execute CPU diagnostics without requiring
    CUDA driver installations.
  - Homebrew formula passes local audit and installation tests.

---

## 5. Post-Publication Verification & Manual Recovery

- [ ] **Run Verification Workflow**:
  - Dispatch the published-release verification workflow:
    ```bash
    gh workflow run verify-published-release.yml --ref main -f tag=v<VERSION>
    ```
  - Verifies published checksums, attestations, Homebrew installation on macOS arm64/Intel/Linux,
    and crates.io installation on Linux.
- [ ] **Host Verification**:
  - On a clean environment, test installing published crates:
    ```bash
    cargo install --locked --version <VERSION> openkind-cli openkind-server
    openkind version
    openkind doctor
    ```
  - Test Homebrew tap install:
    ```bash
    brew update
    brew install whit3rabbit/tap/openkind
    openkind version
    ```
- [ ] **Manual Publication Recovery** (if workflow is interrupted):
  - In case of network timeout or partial upload on the exact tagged commit:
    ```bash
    python3 scripts/publish-crates.py publish --tag v<VERSION> --report dist/crate-publish-report.json
    ```
  - The publisher resumes from the report ledger, reconciling registry state and
    skipping already verified crates.
- [ ] **Finalize Changelog Date**:
  - If `<VERSION> (planned)` was used, replace with the verified publication date
    (`YYYY-MM-DD`) and commit to `main`.

---

## Release History

| Tag | Date | Commit | Crates | Notes |
|---|---|---|---|---|
| `v0.1.0` | 2026-10-03 | `325701be0582c8986ce84c54f88caa21630020c0` | 12 | Initial public release. [Action run](https://github.com/whit3rabbit/openkind/actions/runs/37157085832). Verified [release assets](https://github.com/whit3rabbit/openkind/releases/tag/v0.1.0). |
| `v0.2.0` | 2026-10-04 | `0153911a4c43d700c2a419c47b4a265848514dff` | 12 | Batch runner, loopback defaults, MSRV 1.90. [Action run](https://github.com/whit3rabbit/openkind/actions/runs/37189124401). |
