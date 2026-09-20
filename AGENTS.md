# AGENTS.md

> Repository map and architectural rules for agents working on `opendecision`.
>
> Documentation baseline: v0.7.2, 20 September 2026.

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
- [`docs/RESEARCH.md`](docs/RESEARCH.md): empirical research and prior-art evidence.
- [`docs/whitepaper/OpenDecision_Whitepaper_v0.7.2.md`](docs/whitepaper/OpenDecision_Whitepaper_v0.7.2.md): scientific rationale and measured results.
- Each crate's `AGENTS.md`: module-specific invariants and verification commands.

If documentation and code disagree, do not silently choose one. Use executable
contract tests to establish the current behavior, then update the stale source.

## Native Parity Boundary

Profile `a047d6802c3f06f085b8` is the native integration target:

- Backbone: `Qwen/Qwen3.5-4B-Base` at revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Renderer: state first.
- Readout: score-summary rejection head.
- Bundle SHA-256:
  `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`.
- Calibration temperature: `1.8186799910442777`.
- Policy threshold: `0.98`.
- Probability tolerance: `0.005`.
- Ordering tolerance: `1e-5`.

Phase 3.1 through Phase 3.8 implement the deterministic readout, tokenization,
full-sequence backbone, Qwen-specific continuation, backend-neutral
branch-state, sequential nested execution, breadth-first batched Q/K, and
measured adaptive-scheduling slices:

- immutable model execution metadata.
- offline manifest and safetensors validation.
- f64 normalization, projection, rejection, calibration, and stable softmax.
- native semantic-none mass for Choice only.
- four exported golden-feature fixtures.
- digest-locked offline Qwen tokenizer loading.
- exact state-first segment encoding for all four token fixtures.
- fail-closed loading of the Phase 3B architecture, 47 golden vectors, and
  34-stage diagnostic trace.
- digest-locked loading of both pinned checkpoint shards and exact FP32
  equality with `diagnostic.embedding`.
- Candle CPU execution of all 32 decoder blocks and final RMSNorm in FP32.
- replay of all 34 diagnostic stages and 10 full-sequence candidate features.
- probability parity across all four Phase 3B questions with zero argmax and
  policy changes.
- Qwen-specific continuation state containing attention KV, DeltaNet
  recurrent state, convolution state, and absolute position.
- backend-neutral `BranchableState`/`BranchBatch` contract with
  profile/model/tokenizer/renderer/arithmetic state identity, structural and
  strict fingerprints, exact hybrid byte accounting, immutable-root fork,
  batched fork, and gather/select.
- sequential nested `state → question → candidate` execution: one immutable
  state prefill, question forks, per-candidate forks, fail-closed position and
  immutability verification, and exact replay determinism.
- breadth-first batched Q/K execution: `fork_batch` question lanes from one
  immutable root, per-question candidate fan-outs, fail-closed root/sibling/
  position verification, exact sequential-baseline parity, and fan-out byte
  accounting.
- measured adaptive scheduling: all three strategies behind `run_strategy`
  with forward-call and staged-token accounting, and a crossover-plus-ceiling
  policy whose threshold is measured on the named M4 Max host.

This establishes Phase 3.3 CPU backbone parity, the Phase 3.4 branch-state
contract, Phase 3.5 sequential nested execution parity, Phase 3.6/3.7
batched Q/K parity, and Phase 3.8 measured scheduling for the frozen
fixtures. It does not establish Metal,
vectorized suffix kernels, high-cardinality or
repeatability gates, service registration, release promotion, or full Rust
parity. CPU native parity does not imply Metal or accelerated parity.
Do not register the native backend or map native semantic none onto the Jev wire
format before those contracts are implemented explicitly.

The named M4 Max checkpoint records maximum final-norm absolute error
`5.8174e-05`, maximum candidate-feature error `1.0300e-04`, maximum probability
delta `4.5869e-06`, zero argmax or policy changes, and exact native
cached-versus-full candidate equality. The Phase 3.4 branch gate replays the
root/question/candidate continuation through `BranchableState` and holds the
exact `59,899,904`-byte root accounting, root immutability under fork, batch
fan-out, gather, and exact cached-versus-full state equality (maximum
absolute delta `0.0`). The Phase 3.5 nested gate replays all four Phase 3B questions and 10 candidates through one prefill per fixture case with
maximum probability delta `4.5869e-06`, zero argmax or policy changes, root
content identical to an independent prefill after all fork work, exact replay
and sibling-order determinism, and exact cached-versus-full feature and state
equality (`0.0`). The Phase 3.6/3.7 batched gate fans the same fixtures
through `fork_batch` question lanes (case 0: exactly `3 × 59,899,904 =
179,699,712` root bytes) and per-question candidate lanes; every batched
feature and strict state fingerprint equals the sequential baseline exactly
(`0.0`), the head reaches the same `4.5869e-06` maximum probability delta
with zero argmax, zero policy, and zero cross-strategy decision changes, and
fan-out byte accounting is exact. The Phase 3.8 scheduler gate measured five
workloads on the named Mac with per-repetition feature parity across all
three strategies: sharing beat `repeated_full` everywhere measured
(1.25×–1.98×), `nested_sequential` and `nested_batched` are equal within
noise, `T(3)/T(1)` is 2.87 repeated versus 2.11–2.18 shared, and the measured
crossover threshold is recorded as `MEASURED_MIN_SHARED_SAVINGS_RATIO = 2.0`.
Treat
hidden-vector differences as localization diagnostics, not new acceptance
tolerances.

Follow this integration order:

1. ~~Lift the Qwen-specific continuation state into backend-neutral
   `BranchableState`, including profile identity, stable fingerprinting, fork,
   batched fork, and gather/select.~~ Complete for the CPU path (Phase 3.4).
2. ~~Sequential state, question, and candidate execution.~~ Complete for the
   CPU path (Phase 3.5).
3. ~~Batched question and candidate execution.~~ Complete for the CPU path
   (Phases 3.6/3.7).
4. ~~Amortization (Q-amortization crossover measurement).~~ Measured on the
   named Mac (Phase 3.8). High-cardinality and service-lifecycle validation
   remain.

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
| JSON Schema generation | [`crates/opendecision-gen-schemas/AGENTS.md`](crates/opendecision-gen-schemas/AGENTS.md) |
| Protobuf contract | [`proto/AGENTS.md`](proto/AGENTS.md) |
| Project documentation | [`docs/AGENTS.md`](docs/AGENTS.md) |

## Non-Negotiable Invariants

1. Use relative repository paths in committed documentation, comments, and links.
2. Preserve Jev wire compatibility. Wire floating-point values remain `f64` or
   Protobuf `double`. `NoulAnswer` has no confidence field.
3. Regenerate schemas after changing core wire types. Do not hand-edit generated
   JSON Schema files.
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
