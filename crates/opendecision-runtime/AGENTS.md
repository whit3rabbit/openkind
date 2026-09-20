# AGENTS.md — opendecision-runtime

> LLM developer guide for `opendecision-runtime`. Read this before modifying hardware discovery, memory accounting, or branch-state abstractions.

## Crate Purpose & Boundaries

`opendecision-runtime` is the foundation for hardware abstraction, process execution limits, memory observation, and backend-neutral continuation-state management.

It defines:
- Compute device discovery and target configuration (`DeviceType`, `RuntimeConfig`, `detect_available_devices`).
- Process-level memory observation (`peak_resident_bytes`) for admission control.
- Generic execution plan vocabulary (`ExecutionPlan`) and backend vectorization capabilities (`BackendCapabilities`).
- The backend-neutral `BranchableState` / `BranchBatch` continuation-state contracts.
- Tenant-isolated, byte-bounded branch-state caching with TTL and LRU eviction (`BranchStateCache`).

### Critical Invariants

1. **Complete Shared State Isolation**:
   - In Jev workloads, a large state document (often 10k–50k+ tokens) is evaluated across multiple questions and candidates.
   - For hybrid architectures like Qwen 3.5, branch state includes **attention KV, DeltaNet recurrent state, and convolution state**.
   - Branch isolation requires deep-copying all three state tensor families. KV-only cloning fails to isolate recurrent and convolution state.
2. **Strict vs Structural Fingerprints**:
   - `SchedulingFingerprint`: Fast structural fingerprint hashing profile identity, lineage root, position, and tensor shapes. Used by schedulers without copying or hashing megabytes of tensor payloads.
   - `ContentFingerprint`: Cryptographic/strict fingerprint hashing exact little-endian tensor bytes. Required for cross-process replay verification and persistent cache keys.
3. **Tenant Cache Isolation**:
   - `StateCacheKey` requires an explicit, non-empty tenant namespace. States can never leak or be retrieved across tenant boundaries.
4. **Hardware Detection Fallbacks**:
   - `detect_available_devices()` always includes `DeviceType::Cpu`. Metal MPS is detected on macOS Apple Silicon (`aarch64`).
   - `RuntimeConfig` guarantees at least 1 worker thread to prevent thread pool panics.

## Key Files & Types

- [`src/lib.rs`](./src/lib.rs):
  - `DeviceType`: `Cpu`, `Metal { device_id }`, `Cuda { device_id }`.
  - `RuntimeConfig`: Configuration for target device, memory limits, and worker thread pool.
  - `detect_available_devices()`: Enumerates host acceleration targets.
- [`src/memory.rs`](./src/memory.rs):
  - `peak_resident_bytes()`: Queries the OS `getrusage` high-water mark. macOS reports bytes; other Unix targets are converted from KiB. It is peak RSS, not current RSS.
- [`src/execution.rs`](./src/execution.rs):
  - `ExecutionPlan`: `RepeatedFull`, `NestedSequential`, `NestedBatched`.
  - `BackendCapabilities`: Advertises whether a backend supports vectorized question forwards, vectorized candidate forwards, and lane capacity limits.
- [`src/branch/`](./src/branch/):
  - [`src/branch/state.rs`](./src/branch/state.rs):
    - `pub trait BranchableState: Send + Sync`: `fork_one()`, `fork_batch(lanes)`, `position()`, `tensor_storage_bytes()`, `scheduling_fingerprint()`.
    - `pub trait BranchBatch`: `lanes()`, `select(index)`, `gather(indices)`, `tensor_storage_bytes()`; its state type is `BranchableState`.
  - [`src/branch/identity.rs`](./src/branch/identity.rs):
    - `StateIdentity`: Pinned profile, model, tokenizer, renderer, and arithmetic IDs.
    - `TensorStorageBreakdown`: Exact tensor-payload accounting for KV, recurrence, and convolution only.
  - [`src/branch/fingerprint.rs`](./src/branch/fingerprint.rs):
    - `SchedulingFingerprint`: Fast metadata-only hash for admission and scheduling.
    - `ContentFingerprint`: Exact byte hash for replay verification.
  - [`src/branch/cache.rs`](./src/branch/cache.rs):
    - `BranchStateCache<S>`: Thread-safe in-memory cache bounded by tensor payload bytes.
    - `StateCacheKey`: Tenant-scoped cache key bound to `ContentFingerprint`.
    - `CacheError`: Handles `EmptyTenant`, `StateTooLarge`, and `Poisoned` errors.
  - [`src/branch/error.rs`](./src/branch/error.rs):
    - `StateError`: Lane index out-of-bounds, batch size mismatches, and profile divergence.

## Critical Gotchas & Architectural Rules

1. **State Cache Accounting**:
   The `BranchStateCache` byte limit covers *continuation tensor payloads only*. Process-level admission must separately account for base model weights, scratch buffers, and allocator overhead using `peak_resident_bytes()`.
2. **Zero Worker Threads Guard**:
   Constructing `RuntimeConfig` with 0 threads clamps to 1 automatically to avoid deadlocks or panics in downstream task executors.
3. **Branch Batch Isolation**:
   When implementing `BranchBatch::gather` or `BranchBatch::select`, each extracted lane must be independently mutable without side-effects on sibling lanes.

## Verification Commands

```bash
cargo check -p opendecision-runtime
cargo test -p opendecision-runtime
```
