# openkind-runtime

> Hardware discovery, execution limits, and backend-neutral branch-state runtime for `openkind`.

`openkind-runtime` is the hardware and continuation-state layer of the `openkind` workspace. Model backends query it to discover compute targets and size thread pools; shared-input execution relies on its branch-state contracts to split one evaluation into isolated question and candidate branches.

It defines no model execution itself and depends only on [`openkind-core`](../openkind-core/README.md). The [architecture guide](../../docs/ARCHITECTURE.md) maps where it sits.

## Quickstart

```rust
use openkind_runtime::{detect_available_devices, RuntimeConfig};

// The CPU is always present; CUDA and Metal appear when the host exposes them.
let devices = detect_available_devices();
let config = RuntimeConfig::new(devices[0].clone(), 0, None);

// Zero worker threads clamp to one so downstream pools cannot panic.
assert_eq!(config.worker_threads, 1);
```

## Current scope

- **Device discovery**: `detect_available_devices()` always includes the CPU, enumerates CUDA through NVML when the driver is loadable, and adds Metal on Apple Silicon macOS. The probe reports hardware only and fails soft per device; a detected CUDA device still needs a CUDA-enabled backend build before it can run work.
- **Host and process observation**: `host_hardware()` reads `sysctl` on macOS and `/proc` on Linux, keeping keys the host does not report as `None`. `cpu_time_seconds()` returns cumulative process CPU time (`getrusage` on Unix, `GetProcessTimes` on Windows). `peak_resident_bytes()` returns the OS peak RSS (`getrusage` on Unix, peak working set on Windows) — a high-water mark, not current RSS.
- **Execution topology**: `ExecutionPlan` (`repeated_full`, `nested_sequential`, `nested_batched`) describes how branches share an immutable root. `BackendCapabilities` advertises vectorized question and candidate forwards with lane ceilings, and `BatchForwardMode` records how a plan physically executed: per-lane or vectorized. A plan name is state topology, not a compute guarantee.
- **Branch-state contracts**: `BranchableState` and `BranchBatch` fork, select, and gather lanes without mutating the source. `SchedulingFingerprint` is a cheap process-local fingerprint over lineage, layout, and position, never a content key. `ContentFingerprint` is the strict cross-process fingerprint over execution identity, position, and exact tensor contents, used for persistent cache keys and replay verification.
- **State caching**: `BranchStateCache` is a thread-safe LRU with TTL eviction, bounded by tensor-payload bytes. `StateCacheKey` requires an explicit, non-empty tenant namespace plus a content fingerprint, so states never cross tenant boundaries.
- **Token digests**: role-domain-separated SHA-256 digests over finalized token-ID sequences. `ExecutionInputDigest` is order-sensitive and identifies exactly what ran; `SemanticSetDigest` is order-independent for invariance and isolation testing.
- **Native-run evidence**: `NativeRunWriter` records what a native harness ran and produced under the `openkind-native-run/v1` schema — a sanitized invocation (never raw `argv`), the host environment, row-level outputs, and checksums over every sibling file.

## Memory boundary

`tensor_storage_bytes()` is deliberately narrow: it counts continuation tensor payload and excludes allocator overhead, mapped weights, framework objects, and forward scratch. The cache byte budget covers the same payload only. Admission must account for everything else separately through a measured process envelope such as `peak_resident_bytes()`.

Qwen 3.5 branch implementations must capture attention KV, DeltaNet recurrent state, and convolution state together. Attention masks and KV-only caches are insufficient to isolate question and candidate branches.

## Testing

```bash
cargo test -p openkind-runtime
```

Unit tests sit beside the modules in `src/`. They run offline and do not download model assets.

## License

See the [MIT license](../../LICENSE). Cargo metadata declares `MIT OR Apache-2.0`.
