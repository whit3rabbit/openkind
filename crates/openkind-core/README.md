# openkind-core

> The canonical Jev wire contract, data structures, and validation rules for `openkind`.

`openkind-core` is the zero-dependency foundational crate in the `openkind` workspace. It defines the exact wire types, serialization formats, and validation invariants mandated by the [TypeSafe Jev wire specification](https://docs.typesafe.ai/api).

All upper layers (`openkind-engine`, `openkind-api`, `openkind-server`, `openkind-cli`) depend on this crate. `openkind-core` never depends on upper layers.

## Features

- **Strict Wire Types**:
  - `SystemRequest`: Represents incoming `/v1/systemone` requests containing `state`, `model`, and a map of `Question`s.
  - `SystemResponse`: Represents the model's structured decision answers with token `Usage`.
  - `Question`: Tagged enum of `NoulQuestion`, `ChoiceQuestion`, and `ScoreQuestion`.
  - `Answer`: Tagged enum of `NoulAnswer`, `ChoiceAnswer`, and `ScoreAnswer`.
  - `State`: Polymorphic input supporting plain text, structured JSON objects, or JSON arrays.
  - `ModelInfo` / `ModelsResponse`: Wire structures for the `/v1/models` endpoint.
- **Precision Guarantees**:
  - Floating point values (`probabilities`, `score`, `noul`, `confidence`) use `f64` to prevent floating-point rounding degradation during serialization (e.g. `0.92f32` round-tripping to `0.9200000166893005`).
- **Validation Engine**:
  - `validate_request(&req)`: Validates that requests contain at least one question, questions have non-empty instructions, Choice questions define criteria, and Score questions define at least 2 non-empty levels.
  - `validate_response(&resp, criteria)`: Validates probability distributions (summing to 1.0 within tolerance), confidence bounds ($[0.0, 1.0]$), noul probability ranges, and matching criteria keys.
- **JSON Schema Codegen**:
  - Derives `schemars::JsonSchema` for all wire types to generate standard JSON schemas.

## Usage

```rust
use openkind_core::{validate_request, Question, NoulQuestion, State, SystemRequest};
use std::collections::HashMap;

let mut questions = HashMap::new();
questions.insert(
    "is_urgent".into(),
    Question::Noul(NoulQuestion {
        instructions: serde_json::json!("Does this message convey urgency?"),
        criteria: None,
    }),
);

let request = SystemRequest {
    state: State::Text("Server down in production!".into()),
    model: "mock".into(),
    questions,
};

assert!(validate_request(&request).is_ok());
```

## Testing

```bash
cargo test -p openkind-core
```
