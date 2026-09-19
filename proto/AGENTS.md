# AGENTS.md — opendecision-proto

> LLM developer guide for `opendecision-proto`. Read this before modifying the Protobuf definitions or build script.

## Crate Purpose & Boundaries

`opendecision-proto` contains the Protocol Buffers definition ([`proto/proto/opendecision.proto`](proto/opendecision.proto)) and build-time code generator (`build.rs`) for the `opendecision.system_one.SystemOne` gRPC service.

It produces the Rust types consumed by `opendecision-api::grpc` for client and server stubs.

## Protobuf Schema Breakdown (`proto/proto/opendecision.proto`)

- **Service**: `opendecision.SystemOne`
  - `rpc Evaluate(SystemOneRequest) returns (SystemOneResponse)`
- **Request Messages**:
  - `SystemOneRequest`:
    - `State state = 1`: Context to evaluate (`oneof value { string text = 1; Structured structured = 2; }`).
    - `string model = 2`: Backend model identifier.
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
   - `opendecision.proto` must maintain exact field and type parity with the JSON wire format in `opendecision-core`.
   - `state` uses a `oneof` supporting `string text = 1` or `Structured structured = 2` (raw JSON bytes).
   - `instructions_json` is serialized JSON bytes to preserve the `string | object | array` polymorphism of the Jev spec.
   - Criteria strings for Noul use `string is_true = 1` and `string is_false = 2` to avoid keyword collision with Protobuf/Rust `true`/`false`.
   - Float fields (`noul`, `score`, `confidence`, `probabilities` values) MUST be `double` (64-bit IEEE 754), matching `opendecision-core`'s `f64`. Never use `float`.
2. **Build-Time Compilation (`build.rs`)**:
   - Compiles during `cargo build` using `tonic-prost-build` and `prost-build`.
   - Generated code is written to `OUT_DIR` and re-exported via `src/lib.rs` (`tonic::include_proto!("opendecision")`).

## When You Touch `proto/proto/opendecision.proto`

1. Edit `proto/proto/opendecision.proto`.
2. Verify compilation: `cargo check -p opendecision-proto`.
3. Update conversions in `crates/opendecision-api/src/grpc.rs`:
   - `pb_state_to_core`
   - `pb_questions_to_core`
   - `core_to_pb_response`
4. Run gRPC integration tests:
   ```bash
   cargo test -p opendecision-api --test grpc_roundtrip
   ```

## Verification Commands

```bash
cargo build -p opendecision-proto
cargo test -p opendecision-api --test grpc_roundtrip
```
