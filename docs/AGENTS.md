# AGENTS.md

> Documentation rules for files under `docs/`.

## Scope

The documentation set separates current project status, architecture, empirical
evidence, and wire contracts. Do not copy the same detailed facts into several
documents. Link to the canonical owner instead.

## Canonical Owners

| Subject | File |
|---|---|
| Current phases and open work | [`ROADMAP.md`](ROADMAP.md) |
| Crate topology and data flow | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Research dossier and prior art | [`RESEARCH.md`](RESEARCH.md) |
| Scientific claims and measured results | [`whitepaper/OpenDecision_Whitepaper_v0.7.2.md`](whitepaper/OpenDecision_Whitepaper_v0.7.2.md) |
| HTTP contract | [`../crates/opendecision-api/openapi.yaml`](../crates/opendecision-api/openapi.yaml) |
| JSON Schema contract | [`../crates/opendecision-core/schemas/`](../crates/opendecision-core/schemas/) |
| Protobuf contract | [`../proto/proto/opendecision.proto`](../proto/proto/opendecision.proto) |

The roadmap owns milestone status. The whitepaper owns research interpretation.
Architecture documentation describes landed structure, not planned structure,
unless the text labels a proposal explicitly.

When implementation evidence advances a parity gate, update the roadmap,
architecture, whitepaper evidence/status summary, root agent briefing, and
affected crate briefing together. Search the full documentation set for the
superseded "next" or "not implemented" claim before finishing.

## Evidence Rules

- Distinguish measured Python or hardware evidence from landed Rust behavior.
- Distinguish implementation equivalence from release promotion and model quality.
- State the device, dtype, fixture set, and tolerance for numerical claims.
- Do not describe compilation, artifact loading, or head parity as backbone or
  accelerator parity.
- Phase 3.1 proves the selected head and probability algebra against saved
  candidate features.
- Phase 3.2 proves exact offline tokenizer and state-first segment IDs. It does
  not prove the Qwen hidden-state path.
- Loading Phase 3B architecture and diagnostic vectors is a reference-contract
  gate, not backbone execution parity.
- Phase 3.3 CPU parity covers the frozen 34-stage trace, 10 full-sequence
  candidates, probability/decision replay, and Qwen-specific cached
  continuation. It does not establish Metal, batching, service integration,
  or release promotion.
- Phase 3.4 lifts the complete Qwen continuation state into the backend-neutral
  `BranchableState`/`BranchBatch` contract with profile-bound identity,
  structural and strict fingerprints, exact byte accounting, immutable-root
  fork, batched fork, and gather/select on the CPU path. It does not establish
  Metal, nested or batched Q/K execution, service integration, or release
  promotion. CPU native parity does not imply Metal or accelerated parity.
- Phase 3.5 reproduces sequential nested `state → question → candidate`
  execution on the CPU path: one immutable state prefill per fixture case,
  question and candidate forks for all four Phase 3B questions and 10
  candidates, probability/argmax/policy parity, root immutability, exact
  replay determinism, and exact `repeated_full` agreement. It does not
  establish Metal, batched Q/K execution, service integration, or release
  promotion. CPU native parity does not imply Metal or accelerated parity.
- Phase 3.6/3.7 reproduce breadth-first batched Q/K execution on the CPU
  path: `fork_batch` question lanes from one immutable root and per-question
  candidate fan-outs, with exact sequential-baseline parity, root
  immutability, exact fan-out byte accounting, and unchanged head
  probability/argmax/policy behavior. Per-lane executor calls remain the
  primitive; vectorized suffix kernels, the adaptive scheduler, Metal,
  service integration, and release promotion stay open. CPU native parity
  does not imply Metal or accelerated parity.
- Phase 3.8 measures the adaptive scheduler on the named M4 Max host:
  `run_strategy`/`run_repeated_full` account forward calls and staged tokens
  across all three parity-proven strategies, and `choose_strategy` applies a
  measured crossover threshold (2.0) plus a state-byte ceiling. Sharing beat
  `repeated_full` in every measured cell (1.25x-1.98x) and the two shared
  strategies are equal within noise. Cold-start/cache-warmth cells,
  memory-pressure fallback, vectorized suffix kernels, Metal, service
  integration, and release promotion stay open. CPU native parity does not
  imply Metal or accelerated parity.
- Hidden-vector max-absolute/RMS/cosine values remain localization diagnostics,
  not newly invented acceptance tolerances.
- Native semantic none remains internal until the wire mapping is specified.

## Wire Documentation

The Rust types in `opendecision-core` are the code authority for request and
response shapes. Generated JSON Schema, Protobuf, and OpenAPI must remain
synchronized with those types and their conformance tests.

When changing a wire type:

1. Update the Rust type and validation behavior.
2. Add or update conformance and SDK compatibility tests.
3. Regenerate JSON Schema with
   `cargo run -p opendecision-gen-schemas -- --write`.
4. Update Protobuf and OpenAPI when the same contract is exposed there.
5. Document intentional compatibility changes explicitly.

## Documentation Style

- Use relative repository links. Never commit machine-specific absolute paths.
- Prefer concise present-tense statements over changelog narration.
- Put open work in `ROADMAP.md`, not in module briefings.
- Keep benchmark tables attributable to checked-in artifacts or named sources.
- Do not update test totals by hand. Point to the verification command instead.
- Preserve user-owned research artifacts and notebooks unless the task names them.

## Verification

For documentation-only changes, check formatting and paths without implying that
code or hardware tests ran:

```bash
st --files docs --type md
git diff --check -- docs/
```

For wire or code changes, run the repository battery in [`../AGENTS.md`](../AGENTS.md).
