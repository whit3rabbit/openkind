# AGENTS.md — Documentation

> Documentation architecture, ownership rules, and verification standards for `docs/`.

## Scope & Purpose

The documentation suite maintains a strict division of responsibility across project status, landed architecture, empirical evidence, and wire protocols. Do not duplicate facts across multiple documents. Link to the canonical owner instead.

## Canonical Owners

| Subject | Canonical Owner |
|---|---|
| Current milestone phases, progress, and remaining work | [`ROADMAP.md`](ROADMAP.md) |
| Landed crate boundaries, module topology, and data flow | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Benchmark methodology, harness usage, and recorded runs | [`BENCHMARKS.md`](BENCHMARKS.md) |
| Research dossier, background, and prior art | [`RESEARCH.md`](RESEARCH.md) |
| Scientific rationale, theoretical grounding, and measured results | [`whitepaper/OpenDecision_Whitepaper_v0.8.0.md`](whitepaper/OpenDecision_Whitepaper_v0.8.0.md) |
| HTTP wire specification | [`../crates/opendecision-api/openapi.yaml`](../crates/opendecision-api/openapi.yaml) |
| JSON Schema definitions | [`../crates/opendecision-core/schemas/`](../crates/opendecision-core/schemas/) |
| Protobuf contract | [`../proto/proto/opendecision.proto`](../proto/proto/opendecision.proto) |
| Module-specific rules and invariants | Each crate's `AGENTS.md` |

- `ROADMAP.md` owns phase tracking and upcoming tasks.
- `ARCHITECTURE.md` documents what is currently landed in code, never speculative designs unless explicitly marked as proposals.
- Each crate's `AGENTS.md` serves as the developer guide and invariant boundary for that specific crate.

## Evidence & Parity Standards

1. **Distinguish Measured Evidence from Landed Code**:
   - Explicitly distinguish offline Python research evidence or prototype scripts from landed Rust behavior.
   - State the target device, host architecture, data type, fixture set, and tolerances for all numerical claims.
2. **Implementation Equivalence vs Release Promotion**:
   - Algorithmic equivalence or parity against reference vectors does not automatically constitute release promotion or model quality claims.
   - CPU reference parity does not imply Metal or GPU acceleration parity.
3. **Diagnostics vs Acceptance Tolerances**:
   - Hidden-vector max-absolute, RMS, and cosine metrics are localization diagnostics to pinpoint divergence, not newly invented pass/fail criteria.
   - Acceptance criteria are governed by final decision argmax parity, calibrated probabilities within tolerance (`0.005`), and policy threshold consistency.

## Wire Documentation Synchronization

The Rust types in `opendecision-core` are the code authority for request and response wire shapes. Generated JSON Schema, Protobuf, and OpenAPI contracts must remain synchronized.

When changing a wire type:
1. Update the Rust types and validation logic in `opendecision-core`.
2. Add or update conformance tests in `crates/opendecision-core/tests/conformance/`.
3. Regenerate JSON Schema files:
   ```bash
   cargo run -p opendecision-gen-schemas -- --write
   ```
4. Update Protobuf (`proto/proto/opendecision.proto`) and OpenAPI (`crates/opendecision-api/openapi.yaml` / `docs/openapi.yaml`) if the same surface is exposed. Validate OpenAPI using `npx --yes @redocly/cli lint docs/openapi.yaml`.
5. Update SDK compatibility tests in `crates/opendecision-api/tests/sdk_compat/`.

## Documentation Style Rules

- **No Changelog Narration**: Describe landed architecture and current contracts in concise present tense. Avoid historical phase-by-phase chronological logs in architectural docs and briefings.
- **Relative Links Only**: Use relative repository paths. Never commit machine-specific absolute file paths (`/Users/...` or `C:\...`).
- **Upcoming Work**: Keep planned features and milestone roadmaps in `ROADMAP.md`, not scattered across module briefings.
- **Benchmark Attribution**: Attribute performance numbers to specific hardware configurations, commit hashes, or checked-in benchmark logs.

## Verification Commands

For documentation formatting and link checks:

```bash
# Check indexed documentation files
st --files -t md docs

# Check git diff for trailing whitespace or formatting issues
git diff --check -- docs/
```

For wire contract or code changes, run the root verification battery in [`../AGENTS.md`](../AGENTS.md).
