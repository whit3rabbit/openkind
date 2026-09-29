# AGENTS.md

> Repository map and architectural rules for agents working on `openkind`.

## Project

`openkind` is a Rust decision engine. It speaks the Jev protocol and returns
typed `Noul`, `Choice`, and `Score` answers without an autoregressive
text-generation loop.

The public HTTP and gRPC surfaces remain wire-compatible with TypeSafe's
System One interfaces, and the independent neural implementation uses a
curated open-weight Qwen 3.5 profile documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Sources of Truth

- [`README.md`](README.md) covers workspace setup. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) maps crate boundaries and data flow.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) owns milestone status. [`docs/RESEARCH.md`](docs/RESEARCH.md) and [`docs/whitepaper/WHITEPAPER.md`](docs/whitepaper/WHITEPAPER.md) own research evidence and scientific interpretation.
- [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md), [`docs/families/README.md`](docs/families/README.md), and [`docs/families/NEW_FAMILY.md`](docs/families/NEW_FAMILY.md) cover profiles, the catalog, and family qualification.
- [`docs/MODELS.md`](docs/MODELS.md) indexes every loadable profile: model type, backbone, catalog pull names, backends, and measured runs. When a change edits `registry/v1` or lands a new loadable profile, update that page's model names and tables in the same change.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`docs/MLX.md`](docs/MLX.md), and [`docs/JEV_COMPATIBILITY.md`](docs/JEV_COMPATIBILITY.md) cover measurement, MLX operations, and provider routes. [`docs/ARROW.md`](docs/ARROW.md) owns the unofficial, opt-in Arrow bulk endpoint; it is outside the TypeSafe wire contract.
- [`bindings/README.md`](bindings/README.md) covers the TypeScript, Python, and Swift HTTP clients and server wrappers. Crate-specific instructions are linked in the workspace map below.

Test the contract. If documentation disagrees with code, establish current
behavior through an executable test and update the stale source.

## Public Model Registry

`registry/v1` is the metadata source for curated profiles. The external
[OpenKind model registry](https://github.com/whit3rabbit/openkind-model-registry)
is the public mirror used by `openkind catalog` and `openkind pull`. Checkpoint
weights remain at their authors' repositories. Follow
[`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md) for asset publishing and
mirror verification. Keep distributable assets in the mirror before pinning
their commit, sizes, and digests in this repository.

## Release Names

`openkind-cli` and `openkind-server` are Cargo packages. Their binaries are
`openkind` and `openkindd`. Release archives and the Homebrew formula use
`openkind`. Keep these names aligned in
[the release workflow](.github/workflows/release.yml) and
[formula template](packaging/homebrew/openkind.rb).

## Native Parity Boundary

CPU parity is scoped. It establishes agreement with the CPU reference, while
accelerated parity, task quality, practical throughput, and release readiness
each require separate evidence. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), and [`docs/MLX.md`](docs/MLX.md)
for architecture, evidence, and backend limits.

## Workspace Map

All eleven crates under `crates/` have their own `AGENTS.md`. Read the relevant
guide before changing a crate. The `docs/` and `proto/` directories have scoped
guides too.

| Area | Briefing |
|---|---|
| Jev types and validation | [`crates/openkind-core/AGENTS.md`](crates/openkind-core/AGENTS.md) |
| Engine traits and profiles | [`crates/openkind-engine/AGENTS.md`](crates/openkind-engine/AGENTS.md) |
| HTTP and gRPC API | [`crates/openkind-api/AGENTS.md`](crates/openkind-api/AGENTS.md) |
| Daemon lifecycle | [`crates/openkind-server/AGENTS.md`](crates/openkind-server/AGENTS.md) |
| Operator CLI | [`crates/openkind-cli/AGENTS.md`](crates/openkind-cli/AGENTS.md) |
| Rust client SDK | [`crates/openkind-client/AGENTS.md`](crates/openkind-client/AGENTS.md) |
| TypeScript, Python, and Swift HTTP clients and server wrappers | [`bindings/README.md`](bindings/README.md) |
| Hardware and state lifecycle | [`crates/openkind-runtime/AGENTS.md`](crates/openkind-runtime/AGENTS.md) |
| Model artifacts and readouts | [`crates/openkind-backends/AGENTS.md`](crates/openkind-backends/AGENTS.md) |
| Curated model catalog and local installations | [`crates/openkind-model-store/AGENTS.md`](crates/openkind-model-store/AGENTS.md) |
| Native decision-workload benchmark harness | [`crates/openkind-bench/AGENTS.md`](crates/openkind-bench/AGENTS.md) |
| JSON Schema generation | [`crates/openkind-gen-schemas/AGENTS.md`](crates/openkind-gen-schemas/AGENTS.md) |
| Protobuf contract | [`proto/AGENTS.md`](proto/AGENTS.md) |
| Project documentation | [`docs/AGENTS.md`](docs/AGENTS.md) |

## HTTP Language and Server Bindings

The TypeScript, Python, and Swift packages in `bindings/` call `openkindd` over
HTTP. Their wrappers start a local daemon without invoking the CLI or embedding
Rust. Keep them aligned with
[`crates/openkind-api/openapi.yaml`](crates/openkind-api/openapi.yaml). The
[bindings guide](bindings/README.md) owns route, error, lifecycle, and test
details.

## Benchmark Evidence

Criterion measures mock and component overhead, while `openkind-bench score`
measures pinned native request-path performance and memory. Labeled datasets
establish model quality. Never infer quality from throughput or predictions
alone. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the methodology.

## Cross-Workspace Invariants

1. Use relative repository paths in committed documentation, comments, and links.
2. Keep Jev wire floats as JSON `f64` or Protobuf `double`. `NoulAnswer` has no confidence field. TypeScript and Swift cannot represent state integers above `2^53 - 1` exactly.
3. Validate each answer against its request, including question IDs, answer types, selected options, and probability keys. Consumers separately authorize and verify any resulting action.
4. Qwen branching must isolate attention KV, DeltaNet recurrent state, and convolution state.
5. `openkind` returns structured decisions. Do not add token-by-token text generation.
6. Tests and builds must not download model assets. Parity fixtures and test tokenizers stay vendored and digest-checked.
7. Native Choice questions must include a non-empty `__none__` option and report its semantic-none probability mass.
8. Keep role-typed token digests in explicitly requested offline evidence. Never emit request-derived digests in daemon logs because low-entropy inputs can be guessed.
9. Admission and scheduler estimates must include each strategy's full retained state. For `repeated_full`, sum candidate states across questions with saturating arithmetic.
10. Verify large checkpoint shards in the read-only source directory and mmap them in place. Never stage them in `/tmp` or another temporary directory.
11. Preserve dependency direction. `core` is foundational, `engine` depends on `core`, and `runtime` and `backends` sit below `api`, `server`, and `cli`.
12. When core wire types change, regenerate schemas with the command in Verification. Never hand-edit generated JSON Schema files.
13. Preserve concurrent work. Do not reset, clean, broad-stage, or overwrite unrelated changes.

## Verification

Run the project-specific battery before submitting changes:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
env -u RUST_LOG cargo test --workspace
env -u RUST_LOG cargo test --workspace --benches --locked
cargo run -p openkind-gen-schemas -- --write
git diff --check
```

After schema generation, confirm that unrelated schema files did not change.

For changes under `bindings/`, also run the language-specific tests:

```bash
(cd bindings/typescript && npm install && npm test)
(cd bindings/python && python3 -m unittest discover -s tests)
(cd bindings/swift && swift test)
```

The optional MLX backend is for macOS arm64. Follow [`docs/MLX.md`](docs/MLX.md)
for feature-build and toolchain qualification details. Use
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for warm throughput commands and
recorded runs. BF16 comparisons must use `--strategies repeated_full` until
nested continuation is qualified.
