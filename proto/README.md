# openkind-proto

> Protocol Buffers schema and generated Rust client/server types for `openkind`.

`openkind-proto` contains the canonical gRPC service definitions and protobuf messages for the `openkind.system_one.SystemOne` service.

## Schema: `proto/openkind.proto`

Defines the binary service equivalent of the HTTP `POST /v1/systemone` endpoint:

```protobuf
syntax = "proto3";
package openkind;

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

`build.rs` compiles `proto/openkind.proto` via `tonic-prost-build` during `cargo build`, exposing generated types inside the `openkind` module:

```rust
use openkind_proto::openkind::system_one_client::SystemOneClient;
use openkind_proto::openkind::system_one_server::{SystemOne, SystemOneServer};
```
