# AGENTS.md

> Repository map and architectural rules for agents working on `opendecision`.
>
> Documentation baseline: v0.8.0, 20 September 2026.

## Project

`opendecision` is an independent Rust decision-inference engine that speaks the
Jev protocol. It returns typed `Noul`, `Choice`, and `Score` answers without an
autoregressive text-generation loop.

The public HTTP and gRPC surfaces remain wire-compatible with TypeSafe's
System One interfaces. The neural implementation is independent and uses the
selected open-weight Qwen 3.5 profile described below.

## Sources of Truth

- [`README.md`](README.md): workspace overview and quickstart.
- [`docs/ROADMAP.md`](docs/ROADMAP.md): current milestone status and remaining work.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): crate boundaries and data flow.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): benchmark methodology, harness usage, and recorded runs.
- [`docs/RESEARCH.md`](docs/RESEARCH.md): empirical research and prior-art evidence.
- [`docs/whitepaper/OpenDecision_Whitepaper_v0.8.0.md`](docs/whitepaper/OpenDecision_Whitepaper_v0.8.0.md): scientific rationale and measured results.
- Each crate's `AGENTS.md`: module-specific invariants and verification commands.

If documentation and code disagree, do not silently choose one. Use executable
contract tests to establish the current behavior, then update the stale source.

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
  (defined in `opendecision-runtime`) capturing attention KV, DeltaNet recurrent state,
  and convolution state together with exact tensor-payload accounting and typed scheduling/content fingerprints.
- **Execution Strategies**:
  - `repeated_full`: Evaluates every complete candidate sequence independently.
  - `nested_sequential`: Prefills immutable root state once, then sequentially forks question and candidate lanes.
  - `nested_batched`: Prefills immutable root state once, then fans out question lanes (`fork_batch`) and candidate lanes. This is state topology; it is compute-batched only when the backend advertises vectorized forward.
  - `choose_strategy`: Capability-aware scheduler using the lowest measured ratio (`2.52`), tensor/process memory ceilings, and lane limits. The current CPU default is `nested_sequential`.
  - Every decision records the selected plan and its physical `BatchForwardMode` (`per_lane` vs `vectorized`); the same plan name can execute with different physical graphs on different backends.
  - `--qwen35-execution` on `opendecisiond` forces one plan for diagnostics and reproducibility. It bypasses the profitability policy only — admission ceilings and real backend capabilities still apply.
- **Daemon Registration**: `Qwen35DecisionEngine` integrates directly behind `DecisionEngine`
  and is registered by `opendecisiond` via `--qwen35-*` CLI flags and environment variables.
- **Execution Identity**: Finalized token sequences are the execution contract. Role-typed digests
  (`StateTokenDigest`, `QuestionTokenDigest`, `CandidateTokenDigest`, an order-sensitive
  `ExecutionInputDigest`, and an order-independent `SemanticSetDigest`) live in `opendecision-runtime`
  and are reserved for explicitly requested offline evidence artifacts (raw digests of low-entropy inputs must stay out of daemon logs).
- **Native-Run Evidence**: Harness evidence uses the backend-neutral `opendecision-native-run/v1`
  schema (`opendecision-runtime::evidence`): sanitized invocation (never raw argv), environment,
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
and production release promotion are governed separately by the roadmap. Model-backed
high-K, fresh-process replay, and queue-inclusive load/soak remain open.

## Workspace Map

| Area | Briefing |
|---|---|
| Jev types and validation | [`crates/opendecision-core/AGENTS.md`](crates/opendecision-core/AGENTS.md) |
| Engine traits and profiles | [`crates/opendecision-engine/AGENTS.md`](crates/opendecision-engine/AGENTS.md) |
| HTTP and gRPC API | [`crates/opendecision-api/AGENTS.md`](crates/opendecision-api/AGENTS.md) |
| Daemon lifecycle | [`crates/opendecision-server/AGENTS.md`](crates/opendecision-server/AGENTS.md) |
| Operator CLI | [`crates/opendecision-cli/AGENTS.md`](crates/opendecision-cli/AGENTS.md) |
| Rust client SDK | [`crates/opendecision-client/AGENTS.md`](crates/opendecision-client/AGENTS.md) |
| Hardware and state lifecycle | [`crates/opendecision-runtime/AGENTS.md`](crates/opendecision-runtime/AGENTS.md) |
| Model artifacts and readouts | [`crates/opendecision-backends/AGENTS.md`](crates/opendecision-backends/AGENTS.md) |
| Scoring/timing benchmark harness | [`crates/opendecision-bench/AGENTS.md`](crates/opendecision-bench/AGENTS.md) |
| JSON Schema generation | [`crates/opendecision-gen-schemas/AGENTS.md`](crates/opendecision-gen-schemas/AGENTS.md) |
| Protobuf contract | [`proto/AGENTS.md`](proto/AGENTS.md) |
| Project documentation | [`docs/AGENTS.md`](docs/AGENTS.md) |

## Critical Workspace Gotchas

1. **Wire Precision (`f64`)**:
   Wire floating-point values must remain `f64` (JSON) or Protobuf `double`. Never cast to `f32`
   on wire boundaries; `0.92f32` serializes to `0.9200000166893005`, breaking client wire compatibility.
2. **Hybrid Branch State Isolation**:
   In Qwen 3.5, attention KV alone does NOT isolate branches. The model contains both attention KV,
   DeltaNet recurrent state, and convolution state. Branching or cloning state must isolate all three tensor families.
3. **Non-Autoregressive Invariant**:
   `opendecision` is a decision engine, not a generative chatbot. Host code computes candidate logits
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

## Non-Negotiable Invariants

1. Use relative repository paths in committed documentation, comments, and links.
2. Preserve Jev wire compatibility. Wire floating-point values remain `f64` or
   Protobuf `double`. `NoulAnswer` has no confidence field.
3. Regenerate schemas after changing core wire types (`cargo run -p opendecision-gen-schemas -- --write`).
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
cargo run -p opendecision-gen-schemas -- --write
git diff --check
```

After schema generation, confirm that unrelated schema files did not change.

Optional MLX parity backend (macOS arm64 only): build with
`SDKROOT=$(xcrun --show-sdk-path)` and `--features mlx` for
`opendecision-backends`. Clippy and tests should also be run in that
configuration on the Mac. The mlx-sys build compiles the vendored, pinned
mlx-c (MLX 0.32.2) and needs CMake plus the Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain` if missing). Rebuilding
mlx-c under a different Xcode/Metal toolchain changes the runtime identity:
re-run the 3M.0 qualification gate before trusting any MLX parity result.
