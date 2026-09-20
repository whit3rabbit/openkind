# opendecision-runtime

> Hardware discovery, device management, and execution limits for `opendecision`.

`opendecision-runtime` is the hardware and device-management layer used by native backends.

## Current Scope

The crate currently provides:
- **Device identities and discovery**: CPU, Metal, and CUDA configuration with host discovery for available targets.
- **Execution limits**: Worker-count normalization and optional memory budgets.

## Phase 3 State Requirement

Future branchable execution must capture attention KV, DeltaNet recurrent state, and convolution state. Attention masks and KV-only caches are insufficient to isolate question and candidate branches in Qwen 3.5.
