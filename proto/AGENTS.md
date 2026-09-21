# AGENTS.md — opendecision-proto

> LLM developer guide for `opendecision-proto`. Read this before modifying the Protobuf definitions or build script.

## Crate Purpose & Boundaries

`opendecision-proto` contains the Protocol Buffers definition ([`proto/proto/opendecision.proto`](proto/opendecision.proto)) and build-time code generator (`build.rs`) for the `opendecision.system_one.SystemOne` gRPC service.

It generates the Rust structs and client/server service traits consumed by `opendecision-api::grpc`.

## Protobuf Schema Breakdown (`proto/proto/opendecision.proto`)

- **Service**: `opendecision.SystemOne`
  - `rpc Evaluate(SystemOneRequest) returns (SystemOneResponse)`
- **Request Messages**:
  - `SystemOneRequest`:
    - `State state = 1`: Context to evaluate (`oneof value { string text = 1; Structured structured = 2; }`).
    - `string model = 2`: Backend model identifier or alias.
    - `map<string, Question> questions = 3`: Question map.
  - `Question`: Tagged union `oneof kind { NoulQuestion noul = 1; ChoiceQuestion choice = 2; ScoreQuestion score = 3; }`.
  - `NoulQuestion`: `bytes instructions_json = 1`, optional `NoulCriteria criteria = 2` (`string is_true = 1; string is_false = 2`).
  - `ChoiceQuestion`: `bytes instructions_json = 1`, `map<string, string> criteria = 2`.
  - `ScoreQuestion`: `bytes instructions_json = 1`, `repeated string criteria = 2` ($\ge 2$ ordered levels).
- **Response Messages**:
  - `SystemOneResponse`: `string model = 1`, `map<string, Answer> answers = 2`, `Usage usage = 3`.
  - `Answer`: Tagged union `oneof kind { NoulAnswer noul = 1; ChoiceAnswer choice = 2; ScoreAnswer score = 3; }`.
  - `NoulAnswer`: `double noul = 1` (strictly no `confidence`).
  - `ChoiceAnswer`: `string choice = 1`, `map<string, double> probabilities = 2`, `double confidence = 3`.
  - `ScoreAnswer`: `double score = 1`, `map<string, string> legend = 2`, `map<string, double> probabilities = 3`, `double confidence = 4`.
  - `Usage`: `uint32 input_tokens = 1`, `uint32 output_tokens = 2`.

### Critical Invariants

1. **Protobuf $\leftrightarrow$ Core Semantic Parity**:
   - `opendecision.proto` must maintain exact semantic parity with the JSON wire format in `opendecision-core`.
   - `state` uses a `oneof` supporting `string text = 1` or `Structured structured = 2` (raw JSON bytes).
   - `instructions_json` is serialized JSON bytes to preserve the `string | object | array` polymorphism of the Jev spec.
   - Criteria strings for Noul use `string is_true = 1` and `string is_false = 2` to avoid keyword collisions with Protobuf/Rust `true`/`false`.
   - Float fields (`noul`, `score`, `confidence`, `probabilities` values) MUST be `double` (64-bit IEEE 754), matching `opendecision-core`'s `f64`. **Never use `float`**.
2. **Build-Time Compilation (`build.rs`)**:
   - Compiles during `cargo build` using `tonic-prost-build` and `prost-build`.
   - Generated code is written to `OUT_DIR` and re-exported via `src/lib.rs` (`tonic::include_proto!("opendecision")`).

## Critical Gotchas & Rules

1. **Synchronized gRPC Conversions**:
   Whenever a message in `proto/proto/opendecision.proto` changes, the bidirectional conversion functions in `crates/opendecision-api/src/grpc.rs` (`pb_state_to_core`, `pb_questions_to_core`, `core_to_pb_response`) must be updated to keep HTTP and gRPC behavior identical.
2. **Double Precision**:
   Never use single-precision `float` in Protobuf definitions. IEEE 754 32-bit floats cannot represent wire probabilities with sufficient precision.
3. **`NoulAnswer` Has No Confidence**:
   Protobuf `NoulAnswer` has only `double noul = 1`. Do not add a confidence field to `NoulAnswer`.

## Step-by-Step Change Protocol

1. Edit `proto/proto/opendecision.proto`. Ensure doc comments cross-reference the canonical Jev API (`https://docs.typesafe.ai/api`) and Python SDK (`https://docs.typesafe.ai/sdk/python/api`).
2. Verify Protobuf code generation: `cargo build -p opendecision-proto`.
3. Update conversions in `crates/opendecision-api/src/grpc.rs`.
4. Keep wire parity with OpenAPI (`crates/opendecision-api/openapi.yaml` / `docs/openapi.yaml`).
5. Run gRPC integration tests:
   ```bash
   cargo test -p opendecision-api --test grpc_roundtrip
   ```

## Verification Commands

```bash
# Build proto crate and regenerate bindings
cargo build -p opendecision-proto

# Run full gRPC integration test suite
cargo test -p opendecision-api --test grpc_roundtrip
```
