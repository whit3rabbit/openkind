# opendecision-runtime

> Hardware discovery, device management, and execution runtime for `opendecision` (Phase 2).

`opendecision-runtime` is the hardware and device management layer planned for Phase 2 of the `opendecision` roadmap.

## Scope & Roadmap (Phase 2)

When implemented, `opendecision-runtime` will provide:
- **Device Discovery**: Enumeration of available compute devices (CPU, Apple Metal, NVIDIA CUDA / ROCm).
- **VRAM Accounting & Budgeting**: Tracking available memory to schedule weights and activations safely without OOM crashes.
- **Worker Pools**: Thread and execution pools for concurrent inference batches across hardware devices.
- **KV-Cache / State Cache Management**: Shared-state computation cache management across parallel question evaluations, as outlined in `docs/RESEARCH.md`.

## Current Status

Phase 2 placeholder. The crate currently links into the workspace to verify dependency resolution while Phase 2B benchmarking is completed.
