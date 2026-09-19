# opendecision-backends

> Model loaders and neural execution backends for `opendecision` (Phase 2).

`opendecision-backends` will house real decision-model loaders and forward-pass inference engines behind the `DecisionEngine` trait defined in `opendecision-engine`.

## Scope & Roadmap (Phase 2)

As detailed in `docs/ROADMAP.md` and `docs/RESEARCH.md`, Jev-style decision models do not run autoregressive token generation loops. Instead, they execute a single forward pass over shared state and project representations into typed decision logits.

Planned backends:
- **Candle / GGUF**: In-process lightweight inference using Hugging Face's `candle` framework with quantized weights.
- **ONNX Runtime**: Cross-platform acceleration utilizing ONNX Runtime for CPU/GPU.
- **Remote Provider Passthrough**: Proxy backend that forwards requests to hosted TypeSafe or compatible inference APIs while matching the local `DecisionEngine` interface.
- **Fine-tuned Qwen 3.5 Backbone**: Native implementation following the architecture identified in Phase 2A/2B (shared trunk representation + parallel decision heads).

## Current Status

Phase 2 placeholder. Gated on Phase 2B benchmarking results before the concrete model runtime is chosen.
