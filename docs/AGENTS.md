# AGENTS.md — Documentation

> Documentation architecture, ownership rules, and verification standards for `docs/`.

## Scope & Purpose

The documentation suite maintains a strict division of responsibility across project status, landed architecture, empirical evidence, and wire protocols. Do not duplicate facts across multiple documents. Link to the canonical owner instead.

## Canonical Owners

| Subject | Canonical Owner |
|---|---|
| Current milestone phases, progress, and remaining work | [`ROADMAP.md`](ROADMAP.md) |
| Static model-profile registry and family survey | [`families/README.md`](families/README.md) |
| Procedure for adding a model or family | [`families/NEW_FAMILY.md`](families/NEW_FAMILY.md) |
| Landed crate boundaries, module topology, and data flow | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Curated catalog, public mirror, and publication commands | [`MODEL_REGISTRY.md`](MODEL_REGISTRY.md) |
| Loadable-model index: types, backends, pull names, sizes, and measured runs | [`MODELS.md`](MODELS.md) |
| Benchmark methodology, harness usage, and recorded runs | [`BENCHMARKS.md`](BENCHMARKS.md) |
| MLX runtime contract, implementation guide, limitations, and enhancement path | [`MLX.md`](MLX.md) |
| Unofficial Arrow bulk endpoint: mapping, limits, and usage | [`ARROW.md`](ARROW.md) |
| Research dossier, background, and prior art | [`RESEARCH.md`](RESEARCH.md) |
| Supported environment variables across binaries, SDKs, and bindings | [`ENV.md`](ENV.md) |
| Proxy-cache daemon mode: flags, lifecycle, and guarantees | [`PROXY_CACHE.md`](PROXY_CACHE.md) |
| Scientific rationale, theoretical grounding, and measured results | [`whitepaper/WHITEPAPER.md`](whitepaper/WHITEPAPER.md) |
| Cross-provider Jev compatibility matrix and provider routes | [`JEV_COMPATIBILITY.md`](JEV_COMPATIBILITY.md) |
| TypeScript, Python, and Swift HTTP clients and local server wrappers | [`../bindings/README.md`](../bindings/README.md) |
| HTTP wire specification | [`../crates/openkind-api/openapi.yaml`](../crates/openkind-api/openapi.yaml) |
| JSON Schema definitions | [`../crates/openkind-core/schemas/`](../crates/openkind-core/schemas/) |
| Protobuf contract | [`../proto/proto/openkind.proto`](../proto/proto/openkind.proto) |
| Module-specific rules and invariants | Each crate's `AGENTS.md` |

- `ROADMAP.md` owns phase tracking and upcoming tasks.
- `families/README.md` owns the static family-to-loader catalogue. Use
  `families/NEW_FAMILY.md` for family evaluation gates and Rust integration.
  Keep surveyed, Rust-loadable, task-qualified, and release-promoted as
  separate states. A family is not Rust-loadable until its pinned profile
  loads from local artifacts, has offline parity fixtures and a
  `DecisionEngine` adapter, and is registered with the daemon.
- `MODELS.md` mirrors the catalog and the runnable-profile catalogue for
  operators. Update its model names, backends, and benchmark links in the
  same change that edits `registry/v1` or lands a loadable profile. It
  summarizes measured runs and links to their `benchmarks/` records; it does
  not replace `BENCHMARKS.md` methodology or campaign records. Model quality
  and release promotion require their own roadmap gates.
- `EngineRegistry` routes aliases to already-loaded engines. Do not describe
  it as a generic artifact loader or use a registry entry alone as evidence
  that a model family is implemented.
- `ARCHITECTURE.md` documents what is currently landed in code, never speculative designs unless explicitly marked as proposals.
- `MLX.md` is the operational guide for the landed MLX backend. It links to canonical benchmark and roadmap owners instead of replacing them.
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

The Rust types in `openkind-core` are the code authority for request and response wire shapes. Generated JSON Schema, Protobuf, and OpenAPI contracts must remain synchronized.

When changing a wire type:
1. Update the Rust types and validation logic in `openkind-core`.
2. Add or update conformance tests in `crates/openkind-core/tests/conformance/`.
3. Regenerate JSON Schema files with the [root verification command](../AGENTS.md#verification), then inspect the generated diff.
4. Update Protobuf (`proto/proto/openkind.proto`) and OpenAPI (`crates/openkind-api/openapi.yaml` / `docs/openapi.yaml`) if the same surface is exposed. Validate OpenAPI using `npx --yes @redocly/cli@1.34.5 lint docs/openapi.yaml`.
5. Update SDK compatibility tests in `crates/openkind-api/tests/sdk_compat/`.

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
