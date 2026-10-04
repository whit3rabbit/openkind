# Release checklist

`v0.1.0` was published on October 3, 2026 from commit
`325701be0582c8986ce84c54f88caa21630020c0`. Rust packages
share the version in the root `Cargo.toml`. npm and PyPI publication are outside
this release; the Swift package follows the repository tag.

The [publication run](https://github.com/whit3rabbit/openkind/actions/runs/37157085832)
passed all 35 jobs: CI, archive builds, fresh-host smoke checks, Homebrew checks,
and publication. The [release](https://github.com/whit3rabbit/openkind/releases/tag/v0.1.0)
contains nine archives, nine checksum files, and both crate publication reports.
All 12 published crate archives match the reviewed checksums, include the MIT
license, and identify the tagged source commit.

## Before tagging

- [x] Review the release diff, including existing uncommitted work. Use a clean
  release commit and preserve changes that belong to other work.
- [x] Update [CHANGELOG.md](CHANGELOG.md) for the code included in that commit.
- [x] Confirm every Rust package and internal dependency matches `0.2.0`.
- [x] Confirm each crate archive contains the repository's [MIT license](LICENSE).
- [x] Check that `CARGO_REGISTRY_TOKEN` can publish the intended crates and that
  `HOMEBREW_TAP_TOKEN` can update `whit3rabbit/homebrew-tap`. Store them as GitHub
  Actions secrets; keep tokens out of commands, logs, and release reports.
- [x] Confirm pinned model-registry assets exist in the public mirror, following
  [MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md). Builds and tests never fetch weights.

## Verify the release commit

Install Rust 1.88 for compatibility checks and Rust 1.98.1 for release packaging.
Also install `protoc` and the native C/C++ build tools. The publisher uses Python
3.11 or newer. Package verification reads registry metadata.

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --locked -- -D warnings
env -u RUST_LOG cargo test --workspace --locked
env -u RUST_LOG cargo test --workspace --benches --locked
cargo +1.88.0 check --workspace --all-targets --locked
cargo run -p openkind-gen-schemas --locked -- --write
git diff --exit-code -- crates/openkind-core/schemas
git diff --check
python3 -m unittest discover -s scripts/tests
python3 scripts/publish-crates.py plan --tag v0.1.0 --verify --output dist/crate-versions.json
```

- [x] All commands pass on the final clean commit. A dirty-snapshot check is
  useful during preparation but does not identify the final release source.
- [x] The [CI workflow](.github/workflows/ci.yml) passes on Linux, macOS, and
  Windows, including Rust 1.88, schema synchronization, and backend feature tests.
- [x] TypeScript, Python, and Swift tests pass against a freshly built mock
  `openkindd`, using both `OPENKIND_TEST_BINARY` and `OPENKIND_TEST_URL`.
- [x] Windows dataset installation and CLI child-process forwarding tests pass.
- [x] ONNX, ONNX CUDA, and ONNX ROCm backend tests compile and pass their offline
  checks. A provider that is unavailable on the runner remains an unrun runtime gate.
- [x] MLX checks pass on macOS arm64 with `SDKROOT` set. Follow
  [MLX.md](docs/MLX.md) for GPU and checkpoint qualification.

Cargo 1.98.1 verifies unpublished workspace packages through a temporary
registry. Offline verification can fail with `no hash listed`; allow registry
metadata access. This does not authorize model downloads.

## Crate versions and publish order

This table records the published `0.1.0` release. The publisher derives and
validates the order from Cargo metadata, including retained development
dependencies. With `--verify`, `dist/crate-versions.json` records package
checksums. Its registry status remains `unchecked` until publication reconciles
the registry. `dist/crate-publish-report.json` records attempts and completion.

| Order | Package | Version | Minimum upload interval |
|---:|---|---|---|
| 1 | `openkind-core` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 2 | `openkind-datasets` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 3 | `openkind-model-store` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 4 | `openkind-proto` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 5 | `openkind-engine` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 6 | `openkind-runtime` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 7 | `openkind-api` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 8 | `openkind-backends` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 9 | `openkind-cli` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 10 | `openkind-bench` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 11 | `openkind-client` | `0.2.0` | 600 seconds if new, 60 seconds for an update |
| 12 | `openkind-server` | `0.2.0` | 600 seconds if new, 60 seconds for an update |

`openkind-gen-schemas` tracks `0.2.0` but has `publish = false`.
The Cargo packages `openkind-cli` and `openkind-server` install the binaries
`openkind` and `openkindd`.

The publisher uploads one crate at a time and waits for its version and checksum
in the sparse index before proceeding. Before each upload, registry-resolved
packaging must reproduce the prepared archive checksum. With 12 new crates,
upload spacing alone takes at least 110 minutes. A longer `Retry-After` extends the wait.
These defaults follow the [crates.io rate limiter](https://github.com/rust-lang/crates.io/blob/main/src/rate_limiter.rs);
account limits can differ.

## Verify binary distribution

- [x] Every build in [backend packaging CI](.github/workflows/backend-releases.yml)
  passes for the release source.
- [x] Both standard and ONNX archives pass relocation and checksum checks on
  fresh macOS arm64, macOS Intel, Linux glibc, and Windows runners.
- [x] The portable Linux musl archive passes its separate smoke check.
- [x] macOS arm64 archives include `mlx.metallib` and execute the MLX GPU probe.
- [x] ONNX archives execute the bundled runtime's tiny Gemm probe and include
  runtime license notices.
- [x] Linux and Windows binaries start without CUDA drivers or CUDA user
  libraries installed. Record actual CUDA compiler/package versions.
- [x] Homebrew installation tests pass for macOS arm64, macOS Intel, and Linux.

CPU parity, package compilation, mock API tests, and backend discovery each
establish different evidence. Use [BENCHMARKS.md](docs/BENCHMARKS.md) and the
backend guides for separate model quality, accelerated parity, and throughput
claims. Retain skipped checkpoint checks as skipped.

## Publish and check installation

The [release workflow](.github/workflows/release.yml) resolves the tag to one
commit, reruns CI and archive verification, publishes crates, uploads GitHub
assets, and updates Homebrew in that order. It serializes release runs.

- [x] Tag the reviewed clean commit `v0.1.0` and push that tag, or dispatch the
  release workflow against the same existing tag.
- [x] All 12 crate versions are visible with matching checksums. Retain the
  publisher report from the workflow artifacts.
- [x] GitHub contains all expected archives, checksums, and attestations.
- [x] The Homebrew formula points to those archives and checksums.
- [x] On fresh supported hosts, run `cargo install --locked --version 0.1.0
  openkind-cli openkind-server`, then check `openkind version` and mock serving.
- [x] Install the Homebrew formula and test both binaries and backend diagnostics.
- [x] Replace the planned changelog label with the actual release date after
  publication succeeds.

The [published verification run](https://github.com/whit3rabbit/openkind/actions/runs/37165264892)
passed all 14 jobs. Both published Cargo binaries also passed installation, CPU
diagnostics, and mock serving on macOS arm64.

The [published-release verification workflow](.github/workflows/verify-published-release.yml)
downloads the public archives, verifies their checksums and attestations against
the tag, tests Homebrew on macOS arm64, macOS Intel, and Linux, and installs both
Cargo binaries from crates.io on Linux. Run it after publication and the tap
update finish:

```bash
gh workflow run verify-published-release.yml --ref main -f tag=v0.1.0
```

For manual recovery on the same clean tagged commit, use:

```bash
python3 scripts/publish-crates.py publish --tag v0.1.0 --report dist/crate-publish-report.json
```

Reruns reconcile registry state before skipping completed crates. A checksum
conflict stops publication. Authentication, metadata, build, and ownership errors
also stop. Transient metadata failures retry with 30, 60, 120, 240, and 480-second
delays; publishing cooldowns still apply.

An upload timeout can occur after the upload succeeded. The publisher checks
registry visibility for up to 15 minutes, then records `unknown` if visibility
cannot be confirmed. Inspect that report and rerun against the same commit;
never bump a version or change source to mask a partial release. See the
[Cargo publishing documentation](https://doc.rust-lang.org/cargo/commands/cargo-publish.html).
