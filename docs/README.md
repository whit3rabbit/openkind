# docs/

Project documentation and architectural references for `openpick`.

## Reading Order

1. **[`AGENTS.md`](./AGENTS.md)** — **Start here.** Core briefing for developers and AI agents:
   - System orientation & project vision.
   - Authoritative sources of truth.
   - **Schema Information & Wire Contracts**: Jev JSON Schema (Draft 2020-12) & Protobuf definitions.
   - Phasing and milestone status.
   - Critical invariant checklists and conventions.
2. **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** — System design and topology:
   - Layered crate topology and dependency graph.
   - Dual transport design: HTTP/REST (`/v1/systemone`, `/v1/system_one`) & gRPC (`openpick.SystemOne/Evaluate`).
   - Token accounting and execution dispatch.
   - Error mapping and status code conventions (including HTTP 529 for server overload).
3. **[`ROADMAP.md`](./ROADMAP.md)** — Implementation roadmap and phase tracking:
   - Phase 0 (Wire contract) ✅
   - Phase 1 (Daemon, CLI, & Python SDK compatibility suite) ✅
   - Phase 2 (Real model backends: Candle, llama.cpp/GGUF, ONNX) ⏳
   - Test totals and verification matrices across the workspace.
4. **[`RESEARCH.md`](./RESEARCH.md)** — Research dossier & technical background:
   - System 1 inference paradigm and judgment-envelope protocol origins.
   - Hardware requirements, quantization, and model sizing.
   - Backend runtime evaluation (Candle vs llama.cpp vs ONNX).
   - Zero-copy tensor evaluation strategies.

---

## Wire Schemas & Specifications

| Schema | Format | Specification File | Description |
|---|---|---|---|
| **Jev Request** | JSON Schema (Draft 2020-12) | [`crates/openpick-core/schemas/jev-v1-request.json`](../crates/openpick-core/schemas/jev-v1-request.json) | Request schema: `state`, `model`, `questions` (`noul`, `choice`, `score`). |
| **Jev Response** | JSON Schema (Draft 2020-12) | [`crates/openpick-core/schemas/jev-v1-response.json`](../crates/openpick-core/schemas/jev-v1-response.json) | Response schema: `model`, `answers`, `usage` (`input_tokens`, `output_tokens`). |
| **OpenAPI 3.1** | OpenAPI 3.1.0 (YAML) | [`crates/openpick-api/openapi.yaml`](../crates/openpick-api/openapi.yaml) (also [`docs/openapi.yaml`](./openapi.yaml)) | Complete HTTP surface, status codes, headers, and Jev data schemas. |
| **gRPC Service** | Protocol Buffers (proto3) | [`proto/proto/openpick.proto`](../proto/proto/openpick.proto) | `openpick.SystemOne` service definition with binary wire parity. |
| **Schema Tool** | Rust CLI | [`crates/openpick-gen-schemas`](../crates/openpick-gen-schemas) | Schema generator: `cargo run -p openpick-gen-schemas -- --write`. |
| **Spec Fixtures** | JSON | [`examples/`](../examples/) | Spec example payloads (`01_noul.json` ... `08_response_score.json`). |

