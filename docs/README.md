# docs/

Project documentation and architectural references for `openkind`.

## Reading Order

1. **[`AGENTS.md`](./AGENTS.md)**: Start here for documentation ownership,
   evidence boundaries, wire-documentation rules, and verification guidance.
2. **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** — System design and topology:
   - Layered crate topology and dependency graph.
   - Dual transport design: HTTP/REST (`/v1/systemone`, `/v1/system_one`) & gRPC (`openkind.SystemOne/Evaluate`).
   - Token accounting and execution dispatch.
   - Error mapping and status code conventions (including HTTP 529 for server overload).
3. **[`ROADMAP.md`](./ROADMAP.md)**: Implementation roadmap and phase tracking:
   - Phases 0 and 1 wire, service, CLI, and SDK foundations are complete.
   - Phase 2H required scope and the exploratory 2I/2J model-selection screen
     are complete, with reviewed release confirmation still open.
   - Phase 3A and 3B Python reference scopes are complete.
   - Rust Phase 3.1 through 3.9b CPU parity, backend-neutral `BranchableState`,
     sequential nested execution, batched Q/K execution, and the measured
     adaptive scheduler pass. Bounded model-backed high-K stress, structural
     fresh-process replay, and native service lifecycle smoke are recorded in
     the follow-up verification. Full restored head/decision replay, practical
     high-K latency and production load/soak remain open. The separate MLX
     FP32 parity backend passes its frozen gates, while BF16, native
     vectorized batching, and production promotion remain open. CPU native
     parity does not imply Metal or accelerated parity.
   - The clean checkpoint is the [`v0.8.0 commit verification`](./verification/2026-09-20-v0.8.0-35c481a.md).
     The latest dirty-tree native follow-up is the
     [`native follow-up verification`](./verification/2026-09-20-v0.8.0-native-follow-up-working-tree.md);
     it records the bounded high-K, structural persistence, and service smoke
     evidence without claiming release promotion.
   - The latest MLX runtime and kernel evidence is the
     [`Phase 3M.5 working-tree review`](./verification/phase3m5-2026-09-21-working-tree.md),
     with raw timing artifacts under
     [`benchmarks/2026-09-21-qwen35-mlx-gdn-review/`](./benchmarks/2026-09-21-qwen35-mlx-gdn-review/).
4. **[`RESEARCH.md`](./RESEARCH.md)** — Research dossier & technical background:
   - System 1 inference paradigm and judgment-envelope protocol origins.
   - Hardware requirements, quantization, and model sizing.
   - Backend runtime evaluation (Candle vs llama.cpp vs ONNX).
   - Zero-copy tensor evaluation strategies.
5. **[`BENCHMARKS.md`](./BENCHMARKS.md)** — Benchmark methodology & records:
   - Timing scope, warm policy, and execution-strategy sweep definition.
   - `openkind-bench` harness usage and workload fixture inventory.
   - Prior-art (SemIf) comparability mapping and recorded run evidence.
6. **[`MLX.md`](./MLX.md)**: MLX/Metal backend operations and development:
   - Serialized explicit-stream execution and memory policy.
   - Checkpoint loading, continuation-state, and kernel architecture.
   - Current limitations, benchmark decision, correctness gates, and
     prioritized enhancements.
7. **[`whitepaper/OpenKind_Whitepaper_v0.8.1.md`](./whitepaper/OpenKind_Whitepaper_v0.8.1.md)**:
   Canonical scientific interpretation, evidence register, and measured Phase
   3 native CPU reference engine results through adaptive scheduling.

---

## Wire Schemas & Specifications

| Schema | Format | Specification File | Description |
|---|---|---|---|
| **Jev Request** | JSON Schema (Draft 2020-12) | [`crates/openkind-core/schemas/jev-v1-request.json`](../crates/openkind-core/schemas/jev-v1-request.json) | Request schema: `state`, `model`, `questions` (`noul`, `choice`, `score`). |
| **Jev Response** | JSON Schema (Draft 2020-12) | [`crates/openkind-core/schemas/jev-v1-response.json`](../crates/openkind-core/schemas/jev-v1-response.json) | Response schema: `model`, `answers`, `usage` (`input_tokens`, `output_tokens`). |
| **OpenAPI 3.1** | OpenAPI 3.1.0 (YAML) | [`crates/openkind-api/openapi.yaml`](../crates/openkind-api/openapi.yaml) (also [`docs/openapi.yaml`](./openapi.yaml)) | Complete HTTP surface, status codes, headers, and Jev data schemas. |
| **gRPC Service** | Protocol Buffers (proto3) | [`proto/proto/openkind.proto`](../proto/proto/openkind.proto) | `openkind.SystemOne` service definition with binary wire parity. |
| **Schema Tool** | Rust CLI | [`crates/openkind-gen-schemas`](../crates/openkind-gen-schemas) | Schema generator: `cargo run -p openkind-gen-schemas -- --write`. |
| **Spec Fixtures** | JSON | [`examples/`](../examples/) | Spec example payloads (`01_noul.json` ... `08_response_score.json`). |
