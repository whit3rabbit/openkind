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

## Evidence Rules

- Distinguish measured Python or hardware evidence from landed Rust behavior.
- Distinguish implementation equivalence from release promotion and model quality.
- State the device, dtype, fixture set, and tolerance for numerical claims.
- Do not describe compilation, artifact loading, or head parity as backbone or
  accelerator parity.
- Phase 3.1 proves only the selected head and probability algebra against saved
  candidate features.
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
