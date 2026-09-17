# AGENTS.md — openpick-runtime

> LLM developer guide for `openpick-runtime`. Read this before implementing Phase 2 hardware and device execution abstractions.

## Crate Purpose & Boundaries

`openpick-runtime` is the hardware abstraction layer scheduled for implementation in **Phase 2** of the roadmap.

Its primary role will be managing compute devices, memory limits, thread pools, and shared-state KV caches.

### Invariants & Design Principles (From docs/RESEARCH.md & docs/ROADMAP.md)

1. **Shared State Cache**:
   - In Jev workloads, a single shared state document (often 10k–50k+ tokens) is evaluated across dozens of independent questions.
   - The runtime must compute the state representation **once**, store the intermediate representation/KV-cache, and allow parallel decision heads to evaluate independent question branches against that cache.
2. **Block Attention Isolation**:
   - Questions in a batch must NOT attend to one another: $P(y_A \mid x, q_A)$ must not condition on $q_B, q_C$.
   - The runtime must enforce block-diagonal / isolated attention masks.
3. **Hardware Backing**:
   - Must support device detection and execution dispatch across:
     - CPU (fallback)
     - Apple Silicon (Metal Performance Shaders / MPS via Candle)
     - NVIDIA GPUs (CUDA)
4. **Memory Hygiene**:
   - VRAM accounting must be tracked before batch admission to prevent Out-Of-Memory aborts during concurrent serving.

## Roadmap Prerequisites

Phase 3 (Rust engine implementation) is gated on the results of the Phase 2 Qwen model research experiments in Python documented in `docs/ROADMAP.md` and `docs/RESEARCH.md`. Do not start writing backend tensor operations until the Python benchmarks validate dynamic candidate schemas, LoRA adapters, and cache branching.

## Verification Commands

```bash
cargo check -p openpick-runtime
cargo test -p openpick-runtime
```
