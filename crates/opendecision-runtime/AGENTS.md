# AGENTS.md — opendecision-runtime

> LLM developer guide for `opendecision-runtime`. Read this before modifying hardware or state-lifecycle abstractions.

## Crate Purpose & Boundaries

`opendecision-runtime` is the hardware and execution-limits abstraction layer.

It provides device identities, host discovery, worker limits, and memory-budget configuration. Model state lifecycle and branching remain outside the current implementation.

### Invariants and Design Principles

1. **Complete Shared State**:
   - In Jev workloads, a single shared state document (often 10k–50k+ tokens) is evaluated across dozens of independent questions.
   - Qwen 3.5 state includes attention KV, DeltaNet recurrent state, and convolution state. All three must be captured and isolated at a branch point.
   - KV-only cloning is invalid for the hybrid architecture.
2. **Branch Isolation**:
   - Questions in a batch must NOT attend to one another: $P(y_A \mid x, q_A)$ must not condition on $q_B, q_C$.
   - Attention masks alone do not isolate DeltaNet recurrent or convolution state.
3. **Hardware Backing**:
   - Must support device detection and execution dispatch across:
     - CPU (fallback)
     - Apple Silicon (Metal Performance Shaders / MPS via Candle)
     - NVIDIA GPUs (CUDA)
4. **Memory Hygiene**:
   - VRAM accounting must be tracked before batch admission to prevent Out-Of-Memory aborts during concurrent serving.

## Branchable-State Gate

Head/probability parity and exact tokenizer/rendering parity precede backbone integration. Add `BranchableState` only with the complete hybrid state contract and golden backbone evidence.

## Verification Commands

```bash
cargo check -p opendecision-runtime
cargo test -p opendecision-runtime
```
