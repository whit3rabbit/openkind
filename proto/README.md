# opendecision-proto

> Protocol Buffers schema and generated Rust client/server types for `opendecision`.

`opendecision-proto` contains the canonical gRPC service definitions and protobuf messages for the `opendecision.system_one.SystemOne` service.

## Schema: `proto/opendecision.proto`

Defines the binary service equivalent of the HTTP `POST /v1/systemone` endpoint:

```protobuf
syntax = "proto3";
package opendecision;

service SystemOne {
  rpc Evaluate (SystemOneRequest) returns (SystemOneResponse);
}
```

Key protobuf messages:
- `State`: Polymorphic message supporting plain text strings or raw JSON bytes (`Structured`).
- `Question`: Polymorphic container for `NoulQuestion`, `ChoiceQuestion`, and `ScoreQuestion`.
- `Answer`: Polymorphic container for `NoulAnswer`, `ChoiceAnswer`, and `ScoreAnswer`.
- `Usage`: Token accounting (`input_tokens`, `output_tokens`).

## Compilation & Codegen

`build.rs` compiles `proto/opendecision.proto` via `tonic-prost-build` during `cargo build`, exposing generated types inside the `opendecision` module:

```rust
use opendecision_proto::opendecision::system_one_client::SystemOneClient;
use opendecision_proto::opendecision::system_one_server::{SystemOne, SystemOneServer};
```
