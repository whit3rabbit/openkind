# docs/

Project documentation and architectural references for `opendecision`.

## Reading Order

1. **[`AGENTS.md`](./AGENTS.md)**: Start here for documentation ownership,
   evidence boundaries, wire-documentation rules, and verification guidance.
2. **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** — System design and topology:
   - Layered crate topology and dependency graph.
   - Dual transport design: HTTP/REST (`/v1/systemone`, `/v1/system_one`) & gRPC (`opendecision.SystemOne/Evaluate`).
   - Token accounting and execution dispatch.
   - Error mapping and status code conventions (including HTTP 529 for server overload).
3. **[`ROADMAP.md`](./ROADMAP.md)**: Implementation roadmap and phase tracking:
   - Phases 0 and 1 wire, service, CLI, and SDK foundations are complete.
   - Phase 2H required scope and the exploratory 2I/2J model-selection screen
     are complete, with reviewed release confirmation still open.
   - Phase 3A and 3B Python reference scopes are complete.
   - Rust Phase 3.1 through 3.4 CPU parity and backend-neutral
     `BranchableState` contract gates pass. Sequential nested parity is the
     next gate. CPU native parity does not imply Metal or accelerated parity.
   - Current verification comes from the commands in the root
     [`AGENTS.md`](../AGENTS.md), not a manually maintained test total.
4. **[`RESEARCH.md`](./RESEARCH.md)** — Research dossier & technical background:
   - System 1 inference paradigm and judgment-envelope protocol origins.
   - Hardware requirements, quantization, and model sizing.
   - Backend runtime evaluation (Candle vs llama.cpp vs ONNX).
   - Zero-copy tensor evaluation strategies.
5. **[`whitepaper/OpenDecision_Whitepaper_v0.7.2.md`](./whitepaper/OpenDecision_Whitepaper_v0.7.2.md)**:
   Canonical scientific interpretation, evidence register, and measured Phase
   3A, Phase 3B, and Rust parity results.

---

## Wire Schemas & Specifications

| Schema | Format | Specification File | Description |
|---|---|---|---|
| **Jev Request** | JSON Schema (Draft 2020-12) | [`crates/opendecision-core/schemas/jev-v1-request.json`](../crates/opendecision-core/schemas/jev-v1-request.json) | Request schema: `state`, `model`, `questions` (`noul`, `choice`, `score`). |
| **Jev Response** | JSON Schema (Draft 2020-12) | [`crates/opendecision-core/schemas/jev-v1-response.json`](../crates/opendecision-core/schemas/jev-v1-response.json) | Response schema: `model`, `answers`, `usage` (`input_tokens`, `output_tokens`). |
| **OpenAPI 3.1** | OpenAPI 3.1.0 (YAML) | [`crates/opendecision-api/openapi.yaml`](../crates/opendecision-api/openapi.yaml) (also [`docs/openapi.yaml`](./openapi.yaml)) | Complete HTTP surface, status codes, headers, and Jev data schemas. |
| **gRPC Service** | Protocol Buffers (proto3) | [`proto/proto/opendecision.proto`](../proto/proto/opendecision.proto) | `opendecision.SystemOne` service definition with binary wire parity. |
| **Schema Tool** | Rust CLI | [`crates/opendecision-gen-schemas`](../crates/opendecision-gen-schemas) | Schema generator: `cargo run -p opendecision-gen-schemas -- --write`. |
| **Spec Fixtures** | JSON | [`examples/`](../examples/) | Spec example payloads (`01_noul.json` ... `08_response_score.json`). |
