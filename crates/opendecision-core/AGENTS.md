# AGENTS.md — opendecision-core

> LLM developer guide for `opendecision-core`. Read this before modifying wire types or validation rules.

## Crate Purpose & Boundaries

`opendecision-core` is the **single source of truth** for the Jev wire protocol. It contains:
- Request/Response data structures (`SystemRequest`, `SystemResponse`, `Question`, `Answer`, `State`, `Usage`).
- Input validation (`validate_request`) and response verification (`validate_response`).
- Model listing structures (`ModelInfo`, `ModelsResponse`).
- JSON Schema derivations (`schemars::JsonSchema`).

### Critical Invariants

1. **Zero Internal Dependencies**: `opendecision-core` MUST NOT depend on any other crate in the workspace (`engine`, `api`, `server`, etc.). Dependency flows outward only.
2. **Wire Format Precision (`f64`)**:
   - `probabilities`, `score`, `noul`, and `confidence` MUST remain `f64`.
   - **Never change to `f32`**. `0.92f32` round-trips through serde as `0.9200000166893005`, which is a wire regression.
3. **Strict Breaking Change Protocol**:
   Any change to fields in `request.rs` or `response.rs` breaks the contract for all SDK consumers.
   If you change wire types:
   - Bump `API_VERSION` in `crates/opendecision-core/src/request.rs`.
   - Run `cargo test -p opendecision-core`.
   - Regenerate schemas: `cargo run -p opendecision-gen-schemas -- --write`.
   - Update `crates/opendecision-core/schemas/jev-v1-{request,response}.json`.
   - Add a test case in `tests/conformance.rs`.

## Key Files & Types

- [`src/request.rs`](./src/request.rs):
  - `SystemRequest { state, model, questions }`
  - `API_VERSION`: Current wire specification version constant.
- [`src/response.rs`](./src/response.rs):
  - `SystemResponse { model, answers, usage }`
  - `Usage { input_tokens, output_tokens }`: Required on all responses.
- [`src/question.rs`](./src/question.rs):
  - Tagged union enum `Question`:
    - `Noul(NoulQuestion)`: Yes/no probability; optional `NoulCriteria` (`r#true`, `r#false`).
    - `Choice(ChoiceQuestion)`: Categorical selection over `criteria: HashMap<String, Option<String>>`. Values can be `null`.
    - `Score(ScoreQuestion)`: Ordered rubric `criteria: Vec<String>`, requiring $\ge 2$ levels.
  - `Instructions`: Permissive `serde_json::Value` (`string | object | array`).
- [`src/answer.rs`](./src/answer.rs):
  - Tagged union enum `Answer`:
    - `Noul(NoulAnswer)`: `noul: f64`. **No `confidence` field**.
    - `Choice(ChoiceAnswer)`: `choice: String`, `probabilities: HashMap<String, f64>`, `confidence: f64`.
    - `Score(ScoreAnswer)`: `score: f64`, `legend: HashMap<String, String>`, `probabilities: HashMap<String, f64>`, `confidence: f64`.
- [`src/error/`](./src/error/):
  - Modularized validation and error types:
    - [`src/error/types.rs`](./src/error/types.rs): `ValidationError` enum and `ValidationResult` alias.
    - [`src/error/validate.rs`](./src/error/validate.rs): `validate_request` (catches empty questions, missing instructions, empty choice criteria, $< 2$ score levels, empty score level strings, empty noul criteria strings) and `validate_response` (checks probabilities sum to $1.0 \pm 0.001$, confidence $\in [0.0, 1.0]$, noul $\in [0.0, 1.0]$, score indices are numeric strings, and criteria key consistency).
    - [`src/error/tests/`](./src/error/tests/): Unit tests partitioned into `request_tests.rs` and `response_tests.rs`.
- [`src/state.rs`](./src/state.rs):
  - `State`: Untagged serde enum supporting `Text(String)`, `Object(Map)`, `Array(Vec)`.
- [`tests/conformance.rs`](./tests/conformance.rs):
  - Modular suite of 23 conformance tests directly pinned to TypeSafe Jev spec examples:
    - [`tests/conformance/examples.rs`](./tests/conformance/examples.rs): Spec example requests and responses (`noul`, `choice`, `score`).
    - [`tests/conformance/edge_cases.rs`](./tests/conformance/edge_cases.rs): Flexible instructions (`string | object | array`), null criteria values, and structured state shapes.
    - [`tests/conformance/validation.rs`](./tests/conformance/validation.rs): Validation invariants, required fields, sum-to-one constraints, and confidence bounds.
    - [`tests/conformance/schema.rs`](./tests/conformance/schema.rs): JSON Schema derivations and literal model string preservation.

## Verification Commands

```bash
# Run unit and conformance tests
cargo test -p opendecision-core

# Verify and update JSON schema generation
cargo run -p opendecision-gen-schemas -- --write
```
