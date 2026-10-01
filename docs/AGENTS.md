# AGENTS.md — Documentation

> Documentation architecture, ownership rules, and verification standards for `docs/`.

## Scope & Purpose

The documentation suite maintains a strict division of responsibility across project status, landed architecture, empirical evidence, and wire protocols. Do not duplicate facts across multiple documents. Link to the canonical owner instead.

## Canonical Owners

| Subject | Canonical Owner |
|---|---|
| Static model-profile registry and family survey | [`families/README.md`](families/README.md) |
| Procedure for adding a model or family | [`families/NEW_FAMILY.md`](families/NEW_FAMILY.md) |
| Landed crate boundaries, module topology, and data flow | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Curated catalog, public mirror, and publication commands | [`MODEL_REGISTRY.md`](MODEL_REGISTRY.md) |
| Loadable-model index: types, backends, pull names, sizes, and measured runs | [`MODELS.md`](MODELS.md) |
| Benchmark methodology, harness usage, and recorded runs | [`BENCHMARKS.md`](BENCHMARKS.md) |
| Pinned evaluation datasets: acquisition policy and registry | [`../crates/openkind-datasets/AGENTS.md`](../crates/openkind-datasets/AGENTS.md) |
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

- `families/README.md` owns the catalogue. `families/NEW_FAMILY.md` defines
  evaluation and integration gates. Keep surveyed, Rust-loadable,
  task-qualified, and release-promoted separate. Rust-loadable requires pinned
  artifacts, offline parity fixtures, an adapter, and daemon registration.
- `MODELS.md` mirrors the catalog and runnable profiles. Update names,
  backends, and benchmark links when `registry/v1` or loadable profiles change.
  It links runs; `BENCHMARKS.md` owns methodology. Quality and release
  promotion need separate evidence.
- `EngineRegistry` routes aliases to already-loaded engines. Do not describe
  it as a generic artifact loader or use a registry entry alone as evidence
  that a model family is implemented.
- `ARCHITECTURE.md` documents landed code. Mark speculative designs as proposals.
- `MLX.md` documents the landed MLX backend and links to canonical implementation and benchmark guidance.
- Each crate's `AGENTS.md` defines its local developer rules and invariants.

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
4. When the surface is exposed, update Protobuf (`proto/proto/openkind.proto`) and OpenAPI (`crates/openkind-api/openapi.yaml` / `docs/openapi.yaml`). Validate OpenAPI with `npx --yes @redocly/cli@1.34.5 lint docs/openapi.yaml`.
5. Update SDK compatibility tests in `crates/openkind-api/tests/sdk_compat/`.

## Documentation Style Rules

- **No Changelog Narration**: Describe landed architecture and current contracts in concise present tense. Avoid historical phase-by-phase chronological logs in architectural docs and briefings.
- **Relative Links Only**: Use relative repository paths. Never commit machine-specific absolute file paths (`/Users/...` or `C:\...`).
- **Benchmark Attribution**: Tie performance numbers to hardware, commit hashes, or checked-in benchmark logs.

## Verification Commands

For documentation formatting and link checks:

```bash
# Check indexed documentation files
st --files -t md docs

# Check git diff for trailing whitespace or formatting issues
git diff --check -- docs/
```

For wire contract or code changes, run the root verification battery in [`../AGENTS.md`](../AGENTS.md).
