# CUDA backend

> Implementation guide for native CUDA execution across the native Qwen3.5
> backbone and every candle-backed surveyed family. Benchmark methodology and
> recorded numbers remain canonical in [`BENCHMARKS.md`](BENCHMARKS.md).

## Status

CUDA execution is a build-time option behind `--features cuda`. It reuses the
existing candle model implementations: every family loader now threads a
resolved `candle_core::Device` through the same architecture code the CPU
oracle runs. CUDA kernels can differ numerically from CPU kernels. No CUDA
host has run the parity gates yet: **CPU parity does not imply CUDA parity**. Treat every CUDA answer
as an unpromoted candidate until the gates below run on a CUDA host.

- Every candle-backed family and the native backbone accept a CUDA device;
  selection is per engine at load time and fails closed when device creation
  fails (no driver, bad ordinal, or a binary built without `--features cuda`).
- Continuation-state arithmetic identity follows the device: the native
  backbone emits `candle-cuda-fp32` instead of the pinned
  `candle-cpu-fp32`, so CPU and CUDA states can never be mixed. Surveyed
  families surface the same distinction through their `backend_id` strings
  (for example `decider-4b/cuda-fp32`, `encoder-nli/cuda-fp32:0`).
- The Qwen3.5 backbone re-maps checkpoint shards per forward by design
  (admission semantics); on CUDA this copies weights to the device every
  forward. Correctness precedes throughput; a persistent-device-weight path
  is future work and needs its own evidence.
- `BackendCapabilities` for CUDA loads remain per-lane. No vectorized
  forward is claimed.

## Building

```bash
# Requires the CUDA toolchain (nvcc) at build time; candle compiles its
# kernels during the build. Linux/Windows with a supported CUDA release.
cargo build -p openkind-server --release --features cuda
```

The workspace defaults never enable `cuda`; CI and default builds stay
offline CPU builds.

## Selecting CUDA at runtime

| Surface | Selection |
|---|---|
| Native Qwen3.5 engine | `--qwen35-backend cuda` |
| Surveyed families | `--<family>-backend cuda` (encoder-nli, decoder-letter, kev, decoder-llm, schema-scorer, qwen3guard, von, decoder-logit-qwen3, decider-4b, winnow, laya, encoder-instruct-label, decoder-logit-qwen35) |
| Proxy-cache encoder | `--proxy-cache-encoder-backend cuda` |
| Device ordinal | `--cuda-device <N>` (default `0`, env `OPENKIND_CUDA_DEVICE`) |

Bench probes accept `--backend cuda` with the same `--cuda-device` ordinal.

## Detection

`openkind_runtime::detect_accelerators()` reports the CPU plus every CUDA
device the NVIDIA Management Library (NVML) enumerates, with name and total
memory. These are physical NVML indices; `CUDA_VISIBLE_DEVICES` and CUDA
device ordering can change the ordinals accepted by `--cuda-device`.
`openkind_backends::device::execution_support()` reports what the
running binary actually compiled in (`candle_cuda`, `onnx`, `onnx_cuda`,
`mlx`). The daemon logs both at startup, so an operator can distinguish
"GPU present, binary lacks CUDA" from "no GPU".

## Evidence gates (open)

Before any CUDA answer is trusted for a profile, record the following on a
real CUDA host using the [`BENCHMARKS.md`](BENCHMARKS.md) methodology:

1. Decision argmax parity and calibrated-probability tolerance against the
   pinned CPU oracle fixtures. The existing family replays use CPU loaders;
   CUDA qualification requires explicitly loading the CUDA execution path
   and comparing it against those fixtures.
2. Native request-path throughput and memory, labeled as `candle-cuda-fp32`.
3. A clean-commit promotion record.

## Module map

- Device selection vocabulary: [`crates/openkind-backends/src/device.rs`](../crates/openkind-backends/src/device.rs)
- Accelerator detection: [`crates/openkind-runtime/src/accelerators.rs`](../crates/openkind-runtime/src/accelerators.rs)
- Native backbone device threading: [`crates/openkind-backends/src/qwen35/backbone/model.rs`](../crates/openkind-backends/src/qwen35/backbone/model.rs)
- Family execution seam: each family's `load_with_execution` entry point.
