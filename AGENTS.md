# AGENTS.md

> Repository map and architectural rules for agents working on `openkind`.
>
> Documentation baseline: v0.14.0, 26 September 2026.

## Project

`openkind` is an independent Rust decision-inference engine that speaks the
Jev protocol. It returns typed `Noul`, `Choice`, and `Score` answers without an
autoregressive text-generation loop.

The public HTTP and gRPC surfaces remain wire-compatible with TypeSafe's
System One interfaces. The neural implementation is independent and uses the
selected open-weight Qwen 3.5 profile described below.

## Sources of Truth

- [`README.md`](README.md): workspace overview and quickstart.
- [`docs/ROADMAP.md`](docs/ROADMAP.md): current milestone status and remaining work.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): crate boundaries and data flow.
- [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md): curated catalog, public mirror, and sync procedure.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): benchmark methodology, harness usage, and recorded runs.
- [`docs/MLX.md`](docs/MLX.md): MLX runtime contract, implementation guide, limitations, and enhancement path.
- [`docs/RESEARCH.md`](docs/RESEARCH.md): empirical research and prior-art evidence.
- [`docs/JEV_COMPATIBILITY.md`](docs/JEV_COMPATIBILITY.md): cross-provider Jev compatibility matrix and provider routes.
- [`docs/families/README.md`](docs/families/README.md): static model-profile registry and family survey.
- [`docs/families/NEW_FAMILY.md`](docs/families/NEW_FAMILY.md): family evaluation gates and Rust integration workflow.
- [`docs/whitepaper/WHITEPAPER.md`](docs/whitepaper/WHITEPAPER.md): scientific rationale and measured results.
- [`bindings/README.md`](bindings/README.md): TypeScript, Python, and Swift HTTP clients and local server wrappers.
- Each crate's `AGENTS.md`: module-specific invariants and verification commands.

If documentation and code disagree, do not silently choose one. Use executable
contract tests to establish the current behavior, then update the stale source.

## Public Model Registry

This repository's `registry/v1` is the metadata source for curated model
profiles. The external [OpenKind model registry](https://github.com/whit3rabbit/openkind-model-registry)
is the public mirror used by `openkind catalog` and `openkind pull`. Its local
checkout is a sibling of this repository at `../openkind-model-registry`.
The mirror holds catalog metadata, pinned profile assets, and exported
tokenizers; checkpoint weights remain at their authors' repositories.

When a profile changes, put distributable assets in the external repository
first, then pin their commit, sizes, and digests in this repository's manifest.
Verify the sibling checkout with
`python3 scripts/sync-model-registry.py --mirror ../openkind-model-registry`.
Use `--write` only to copy catalog and manifest metadata into a clean mirror.
After reviewing, committing, and pushing that mirror, use `--remote` to verify
public HTTPS bytes. The script never commits, pushes, or copies checkpoint
shards. See [the registry guide](docs/MODEL_REGISTRY.md) for the full sequence.

## Release Names

`openkind-cli` and `openkind-server` are the Cargo package names; their
binaries are `openkind` and `openkindd`. Release archives and the Homebrew
formula use `openkind`. Keep these names aligned in
[the release workflow](.github/workflows/release.yml) and
[formula template](packaging/homebrew/openkind.rb).

## Native Parity Boundary & Architecture

Profile `a047d6802c3f06f085b8` is the native integration target:

- **Backbone**: `Qwen/Qwen3.5-4B-Base` at revision `1001bb4d826a52d1f399e183466143f4da7b741b`.
  Executes 24 DeltaNet linear-attention layers and 8 full grouped-query attention layers
  plus final RMSNorm in FP32 on CPU.
- **Renderer**: State-first segmented tokenization. The state document is prefilled once
  to form an immutable root prefix; question and candidate suffixes branch off this root.
- **Readout**: Score-summary rejection head with normalization, projection, rejection,
  temperature calibration (`1.8186799910442777`), policy threshold (`0.98`), and stable softmax.
- **Continuation State**: Backend-neutral `BranchableState` / `BranchBatch` abstractions
  (defined in `openkind-runtime`) capturing attention KV, DeltaNet recurrent state,
  and convolution state together with exact tensor-payload accounting and typed scheduling/content fingerprints.
- **Execution Strategies**:
  - `repeated_full`: Evaluates every complete candidate sequence independently.
  - `nested_sequential`: Prefills immutable root state once, then sequentially forks question and candidate lanes.
  - `nested_batched`: Prefills immutable root state once, then fans out question lanes (`fork_batch`) and candidate lanes. This is state topology; it is compute-batched only when the backend advertises vectorized forward.
  - `choose_strategy`: Capability-aware scheduler using the lowest measured ratio (`2.52`), tensor/process memory ceilings, and lane limits. The current CPU default is `nested_sequential`.
  - Every decision records the selected plan and its physical `BatchForwardMode` (`per_lane` vs `vectorized`); the same plan name can execute with different physical graphs on different backends.
  - `--qwen35-execution` on `openkindd` forces one plan for diagnostics and reproducibility. It bypasses the profitability policy only — admission ceilings and real backend capabilities still apply.
- **Daemon Registration**: `Qwen35DecisionEngine` integrates directly behind `DecisionEngine`
  and is registered by `openkindd` via `--qwen35-*` CLI flags and environment variables.
  Surveyed-family engines (`openkind_backends::families`) and composite router scripts register via
  `--<family>-*` CLI flags and environment variables.
- **Execution Identity**: Finalized token sequences are the execution contract. Role-typed digests
  (`StateTokenDigest`, `QuestionTokenDigest`, `CandidateTokenDigest`, an order-sensitive
  `ExecutionInputDigest`, and an order-independent `SemanticSetDigest`) live in `openkind-runtime`
  and are reserved for explicitly requested offline evidence artifacts (raw digests of low-entropy inputs must stay out of daemon logs).
- **Native-Run Evidence**: Harness evidence uses the backend-neutral `openkind-native-run/v1`
  schema (`openkind-runtime::evidence`): sanitized invocation (never raw argv), environment,
  profile/backend/execution identity, parity/performance/memory reports, and per-file checksums.
- **Native Semantic None**: Choice questions handle semantic-none mass explicitly via
  the reserved criteria key `__none__` (`SEMANTIC_NONE_OPTION`). The profile declares this explicitly
  as `ProbabilitySpace::OfferedOptionsPlusSemanticNone`; the adapter fails closed on any other
  declared space instead of renormalizing.

Parity tolerances against golden fixtures:
- Bundle SHA-256: `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`.
- Probability tolerance: `0.005`.
- Ordering tolerance: `1e-5`.
- Hidden-vector absolute/RMS/cosine differences are localization diagnostics, not acceptance tolerances.

Current native CPU parity does not imply Metal or accelerated parity. Accelerated kernels
and production release promotion are governed separately by the roadmap. Bounded
model-backed high-K and fresh-process replay gates passed on the named Mac. Practical
high-K latency, Metal or accelerated parity, queue-inclusive load/soak, and release
promotion remain open.

## Workspace Map

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

The source packages in `bindings/typescript`, `bindings/python`, and
`bindings/swift` call a running `openkindd` over HTTP. They do not invoke the
`openkind` CLI process. Each also has a process wrapper for starting a local
`openkindd`; this does not embed Rust or load model weights in the host language.
Keep them aligned with
[`openkind-api/openapi.yaml`](crates/openkind-api/openapi.yaml):

- Send an explicit `model` on the wire to `POST /v1/systemone`. Also support
  `GET /v1/models` and the unauthenticated `GET /health` probe.
- Preserve wire `f64` values and the tagged `noul`, `choice`, and `score`
  shapes. A Noul answer has no confidence field. TypeScript and Swift state
  numbers cannot represent integers above `2^53 - 1` exactly.
- Check successful answers against the submitted question IDs and types.
  Choice selections and probability keys must match the submitted criteria.
  The caller still authorizes and verifies any action based on a decision.
- Surface the HTTP error envelope and `x-typesafe-request-id` header. These
  packages do not implement the Rust client's retries, gRPC, OpenRouter, or
  Cloudflare transport.
- Server wrappers own only the `openkindd` child they start, bind HTTP to an
  explicit loopback port, disable gRPC, and pass API keys through the child
  environment. The Swift server wrapper is macOS only; the TypeScript server
  entrypoint requires Node.

The [bindings guide](bindings/README.md) owns setup and language-specific test
commands. Mock-daemon transport checks establish wire behavior, not native
model quality, full API coverage, or release readiness.

## Benchmark Taxonomy

Always qualify which kind of benchmark was run. These produce different
evidence and are not interchangeable:

| Benchmark type | Primary path | What it measures |
|---|---|---|
| Criterion component microbenchmark | `cargo bench` targets in `openkind-cli`, `openkind-server`, and `openkind-client` | Warm MockEngine parsing, validation, middleware, serialization, dispatch, and localhost SDK overhead. These runs do not measure native-model throughput or classification quality. |
| Native request-path benchmark | `openkind-bench score` | Full engine request-path predictions, timing, strategy parity, and attributed peak RSS for a pinned backend and workload. Predictions alone are not classification-accuracy evidence. |
| Classification or model-quality evaluation | Labeled evaluation datasets with explicit metrics and recorded provenance | Semantic task quality. Report the dataset revision, metric definition, seed, checkpoint/profile identity, and exclusions. Do not infer this evidence from Criterion or throughput-only runs. |

Criterion reports live under ignored `target/criterion/`. Use
`scripts/bench-rss.sh` for whole-process peak RSS, not bytes per operation.
Methodology, complete commands, and evidence boundaries live in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Critical Workspace Gotchas

1. **Wire Precision (`f64`)**:
   Wire floating-point values must remain `f64` (JSON) or Protobuf `double`. Never cast to `f32`
   on wire boundaries; `0.92f32` serializes to `0.9200000166893005`, breaking client wire compatibility.
2. **Hybrid Branch State Isolation**:
   In Qwen 3.5, attention KV alone does NOT isolate branches. The model contains both attention KV,
   DeltaNet recurrent state, and convolution state. Branching or cloning state must isolate all three tensor families.
3. **Non-Autoregressive Invariant**:
   `openkind` is a decision engine, not a generative chatbot. Host code computes candidate logits
   and serializes structured JSON responses directly. Never introduce token-by-token text generation loops.
4. **Offline Reproducibility**:
   Tests and builds must never download model assets from Hugging Face or the internet. All parity
   fixtures and test tokenizers are vendored locally and digest-checked.
5. **Semantic None in Choice**:
   Native Choice questions must explicitly include a non-empty `__none__` option in criteria to reserve
   and report semantic-none probability mass on the wire.
6. **No Token or Request Digests in Daemon Logs**:
   Role-typed token digests (`ExecutionInputDigest`, `StateTokenDigest`, `SemanticSetDigest`) are strictly reserved
   for explicitly requested offline evidence artifacts. They must NEVER be emitted in daemon telemetry or debug logs,
   as low-entropy inputs can be guessed offline from raw digests.
7. **Complete Strategy Memory Retention Accounting**:
   Admission ceilings and scheduler memory estimations must account for the full retained state footprint of the strategy.
   `repeated_full` retains one state per candidate; its tensor memory estimate must sum all candidate states across all
   questions using saturating arithmetic to prevent integer wraparound.
8. **In-Place Read-Only Checkpoint Verification**:
   Model checkpoint shards are multi-gigabyte files (up to 4 GB+ per shard). They must be verified in-place on the
   read-only checkpoint directory and mmapped directly. Never copy or stage checkpoint shards to `/tmp` or ephemeral
   directories during model load.
9. **Choice and Consumer Authority**:
   The `Choice` wire type does not close the option set by itself. Validate answers against the originating request,
   including question IDs, answer types, selected option, and probability keys. Trusted consumer code determines
   eligible actions and separately authorizes and verifies any action selected from the decision.

## Non-Negotiable Invariants

1. Use relative repository paths in committed documentation, comments, and links.
2. Preserve Jev wire compatibility. Wire floating-point values remain `f64` or
   Protobuf `double`. `NoulAnswer` has no confidence field.
3. Regenerate schemas after changing core wire types (`cargo run -p openkind-gen-schemas -- --write`).
   Do not hand-edit generated JSON Schema files.
4. Qwen branch state includes attention KV, DeltaNet recurrent state, and
   convolution state. Attention masks or KV-only cloning do not isolate branches.
5. Preserve dependency direction. `core` is foundational. `engine` depends on
   `core`. `runtime` and `backends` sit below `api`, `server`, and `cli`.
6. Keep implementation equivalence separate from release promotion and model
   quality claims.
7. Tests and builds must not download model artifacts. Parity fixtures are
   vendored and digest-checked.
8. Preserve concurrent work. Do not reset, clean, broad-stage, or overwrite
   unrelated changes.

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

Optional MLX parity backend (macOS arm64 only): build with
`SDKROOT=$(xcrun --show-sdk-path)` and `--features mlx` for
`openkind-backends`. Clippy and tests should also be run in that
configuration on the Mac. The mlx-sys build compiles the vendored, pinned
mlx-c (MLX 0.32.2) and needs CMake plus the Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain` if missing). Rebuilding
mlx-c under a different Xcode/Metal toolchain changes the runtime identity:
re-run the 3M.0 qualification gate before trusting any MLX parity result.
For warm throughput comparisons, use `openkind-bench` with
`--features mlx` and `--engine qwen35-mlx-fp32` or `qwen35-mlx-bf16`; the
command, pinned `Qwen/Qwen3.5-4B-Base` model revision, and current results are
recorded in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). BF16 benchmark runs
must use `--strategies repeated_full` until nested continuation is qualified.
