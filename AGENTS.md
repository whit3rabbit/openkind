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

Phase 3.1 through Phase 3.3 implement the deterministic readout, tokenization,
full-sequence backbone, and Qwen-specific continuation slices:

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

This establishes Phase 3.3 CPU backbone parity for the frozen fixtures. It does
not establish Metal, backend-neutral `BranchableState`, batched Q/K execution,
service registration, release promotion, or full Rust parity.
Do not register the native backend or map native semantic none onto the Jev wire
format before those contracts are implemented explicitly.

The named M4 Max checkpoint records maximum final-norm absolute error
`5.8174e-05`, maximum candidate-feature error `1.0300e-04`, maximum probability
delta `4.5869e-06`, zero argmax or policy changes, and exact native
cached-versus-full candidate equality. Treat hidden-vector differences as
localization diagnostics, not new acceptance tolerances.

Follow this integration order:

1. Lift the Qwen-specific continuation state into backend-neutral
   `BranchableState`, including profile identity, stable fingerprinting, fork,
   batched fork, and gather/select.
2. Sequential state, question, and candidate execution.
3. Batched question and candidate execution.
4. Amortization, high-cardinality, and service-lifecycle validation.

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
