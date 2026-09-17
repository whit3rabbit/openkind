# AGENTS.md — openpick-core

> LLM developer guide for `openpick-core`. Read this before modifying wire types or validation rules.

## Crate Purpose & Boundaries

`openpick-core` is the **single source of truth** for the Jev wire protocol. It contains:
- Request/Response data structures (`SystemRequest`, `SystemResponse`, `Question`, `Answer`, `State`, `Usage`).
- Input validation (`validate_request`) and response verification (`validate_response`).
- Model listing structures (`ModelInfo`, `ModelsResponse`).
- JSON Schema derivations (`schemars::JsonSchema`).

### Critical Invariants

1. **Zero Internal Dependencies**: `openpick-core` MUST NOT depend on any other crate in the workspace (`engine`, `api`, `server`, etc.). Dependency flows outward only.
2. **Wire Format Precision (`f64`)**:
   - `probabilities`, `score`, `noul`, and `confidence` MUST remain `f64`.
   - **Never change to `f32`**. `0.92f32` round-trips through serde as `0.9200000166893005`, which is a wire regression.
3. **Strict Breaking Change Protocol**:
   Any change to fields in `request.rs` or `response.rs` breaks the contract for all SDK consumers.
   If you change wire types:
   - Bump `API_VERSION` in `crates/openpick-core/src/request.rs`.
   - Run `cargo test -p openpick-core`.
   - Regenerate schemas: `cargo run -p openpick-gen-schemas -- --write`.
   - Update `crates/openpick-core/schemas/jev-v1-{request,response}.json`.
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
- [`src/error.rs`](./src/error.rs):
  - `validate_request`: Catches empty questions, missing instructions, empty choice criteria, $< 2$ score levels, empty score level strings, empty noul criteria strings.
  - `validate_response`: Checks probabilities sum to $1.0 \pm 0.001$, confidence $\in [0.0, 1.0]$, noul $\in [0.0, 1.0]$, score indices are numeric strings, and criteria key consistency.
- [`src/state.rs`](./src/state.rs):
  - `State`: Untagged serde enum supporting `Text(String)`, `Object(Map)`, `Array(Vec)`.
- [`tests/conformance.rs`](./tests/conformance.rs):
  - 23 conformance tests directly pinned to TypeSafe Jev spec examples.

## Verification Commands

```bash
# Run unit and conformance tests
cargo test -p openpick-core

# Verify and update JSON schema generation
cargo run -p openpick-gen-schemas -- --write
```
