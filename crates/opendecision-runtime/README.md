# opendecision-runtime

> Hardware discovery, device management, and execution limits for `opendecision`.

`opendecision-runtime` is the hardware and device-management layer used by native backends.

## Current Scope

The crate currently provides:
- **Device identities and discovery**: CPU, Metal, and CUDA configuration with host discovery for available targets.
- **Execution limits**: Worker-count normalization and optional memory budgets.
- **Execution topology**: backend capability declarations and portable repeated, nested-sequential, and nested-batched plans.
- **Branch-state contracts**: typed scheduling and content fingerprints, immutable fork/batch/gather semantics, and tensor-payload accounting.
- **State caching**: tenant-scoped, TTL-bound, tensor-byte-limited LRU storage keyed only by content fingerprints.
- **Process evidence**: an OS peak-resident-memory reading for conservative admission and benchmark records.

## Memory Boundary

`tensor_storage_bytes()` is deliberately narrow: it counts continuation tensor
payload, not allocator overhead, mapped weights, framework objects, or forward
scratch. Admission combines that count with an explicit process-memory envelope.
The reported resident figure is a process high-water mark, not current RSS.

Qwen 3.5 branch implementations must capture attention KV, DeltaNet recurrent
state, and convolution state. Attention masks and KV-only caches are
insufficient to isolate question and candidate branches.
