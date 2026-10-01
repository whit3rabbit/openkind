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
3. **[`RESEARCH.md`](./RESEARCH.md)** — Research dossier & technical background:
   - System 1 inference paradigm and judgment-envelope protocol origins.
   - Hardware requirements, quantization, and model sizing.
   - Backend runtime evaluation (Candle vs llama.cpp vs ONNX).
   - Zero-copy tensor evaluation strategies.
4. **[`BENCHMARKS.md`](./BENCHMARKS.md)** — Benchmark methodology & records:
   - Timing scope, warm policy, and execution-strategy sweep definition.
   - `openkind-bench` harness usage and workload fixture inventory.
   - Prior-art (SemIf) comparability mapping and recorded run evidence.
5. **[`MLX.md`](./MLX.md)**: MLX/Metal backend operations and development:
   - Serialized explicit-stream execution and memory policy.
   - Checkpoint loading, continuation-state, and kernel architecture.
   - Current limitations, benchmark decision, correctness gates, and
     prioritized enhancements.
6. **[`whitepaper/WHITEPAPER.md`](./whitepaper/WHITEPAPER.md)**:
   Canonical scientific interpretation, evidence register, native CPU reference
   results, Phase 4B–4D closeout, and Phase 4E audit gate.
7. **[`families/`](./families/README.md)** — Decision-model family catalogue:
   - Architectural pattern per family, status (implemented vs surveyed), and
     what blocks implementation.
   - One page per family. Each page links to the canonical owner for every
     quantitative or status claim and never duplicates facts from other docs.
   - Use this index to scope new family selection work; do not invent a new
     architecture pattern without first adding it here.
   - [`families/NEW_FAMILY.md`](./families/NEW_FAMILY.md): family-specific
     evaluation gates and the Rust integration workflow.
8. **[`PROXY_CACHE.md`](./PROXY_CACHE.md)**: Proxy-cache daemon mode:
   - Distilling cache in front of a remote Jev API: routing, confidence
     calibration, and the disagreement budget.
   - CLI flag surface, encoder profiles (hash vs pinned bge), and on-disk
     layout.
   - Guarantees, limits, and the test inventory.
9. **[`MODEL_REGISTRY.md`](./MODEL_REGISTRY.md)**: Curated model distribution:
   - OpenKind owns catalog metadata and compiled-in loaders. The separate
     public registry serves manifests and small pinned profile assets.
   - CLI pull and explicit daemon serving commands, plus the
     [`sync-model-registry.py`](../scripts/sync-model-registry.py) publication check.
10. **[`ARROW.md`](./ARROW.md)**: Unofficial Arrow bulk endpoint:
    - Opt-in `POST /v1/arrow` for many states per request with an Arrow IPC
      stream response, outside the TypeSafe wire contract.
    - Column mapping, metadata keys, limits, and client usage.

---

## Wire Schemas & Specifications

The [Jev compatibility matrix](JEV_COMPATIBILITY.md) compares OpenKind, TypeSafe,
OpenRouter, and Cloudflare field by field and records the Rust SDK's provider routes.

| Schema | Format | Specification File | Description |
|---|---|---|---|
| **Jev Request** | JSON Schema (Draft 2020-12) | [`crates/openkind-core/schemas/jev-v1-request.json`](../crates/openkind-core/schemas/jev-v1-request.json) | Request schema: `state`, `model`, `questions` (`noul`, `choice`, `score`). |
| **Jev Response** | JSON Schema (Draft 2020-12) | [`crates/openkind-core/schemas/jev-v1-response.json`](../crates/openkind-core/schemas/jev-v1-response.json) | Response schema: `model`, `answers`, `usage` (`input_tokens`, `output_tokens`). |
| **OpenAPI 3.1** | OpenAPI 3.1.0 (YAML) | [`crates/openkind-api/openapi.yaml`](../crates/openkind-api/openapi.yaml) (also [`docs/openapi.yaml`](./openapi.yaml)) | Complete HTTP surface, status codes, headers, and Jev data schemas. |
| **gRPC Service** | Protocol Buffers (proto3) | [`proto/proto/openkind.proto`](../proto/proto/openkind.proto) | `openkind.SystemOne` service definition with binary wire parity. |
| **Schema Tool** | Rust CLI | [`crates/openkind-gen-schemas`](../crates/openkind-gen-schemas) | Schema generator: `cargo run -p openkind-gen-schemas -- --write`. |
| **Spec Fixtures** | JSON | [`examples/`](../examples/) | Spec example payloads (`01_noul.json` ... `08_response_score.json`). |
