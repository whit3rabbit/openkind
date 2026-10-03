# ROCm backend

> Implementation guide for AMD ROCm execution through the ONNX Runtime ROCm
> execution provider. Benchmark methodology and recorded numbers remain
> canonical in [`BENCHMARKS.md`](BENCHMARKS.md); the ONNX artifact and runtime
> contract is canonical in [`ONNX.md`](ONNX.md).

## Status

ROCm execution is a build-time option behind `--features onnx-rocm` (Linux
only). It reuses the existing ONNX family adapters: the eight families with
an ONNX-representable readout execute `model.onnx` through the ROCm execution
provider instead of the CPU provider. The ROCm execution provider is a
separate arithmetic path — kernels and math libraries differ numerically from
both the CPU and CUDA execution providers, and cross-device comparisons in
the ollaya decision-model registry's ROCm work (their pull request #33)
measured device-specific fast-math drift compounding through attention. No
ROCm host has run the parity gates yet: **CPU parity and CUDA parity do not
imply ROCm parity**. Treat every ROCm answer as an unpromoted candidate until
the gates below run on a ROCm host.

- The eight ONNX-capable surveyed families accept an ROCm selection;
  selection is per engine at load time and fails closed when the loaded
  ONNX Runtime cannot provide the ROCm execution provider, when the artifact
  is missing, or when the binary was built without `--features onnx-rocm`.
  There is no silent fallback to CPU.
- Execution identity follows the provider through `backend_id` strings
  (`encoder-nli/onnx-rocm:0`, `laya/onnx-rocm:1`), so ROCm answers can never
  be confused with CPU, CUDA, or MLX answers.
- Native candle execution on ROCm is surveyed, not implemented: candle 0.8.0
  has no ROCm backend upstream. The Qwen3.5 hybrid backbone, the GGUF
  families, and the pointer-head readouts (kev, strands-decider) have no
  ONNX export and therefore no ROCm path; selecting `onnx-rocm` for them
  fails closed with an explanation.
- `BackendCapabilities` for ROCm loads remain per-lane. No vectorized
  forward is claimed.

## Building

```bash
# Linux only. Offline: ort is built with load-dynamic, so nothing downloads
# at build time and no ROCm toolchain is required to compile the binary.
cargo build -p openkind-server --release --features onnx-rocm
```

The workspace defaults never enable `onnx-rocm`; CI and default builds stay
offline CPU builds. No prebuilt ONNX Runtime distribution ships the ROCm
execution provider, so the operator must place a ROCm-capable ONNX Runtime
shared library (built from source with `--use_rocm` against a supported ROCm
release) and point the daemon at it with `--onnx-runtime <path>` or
`ORT_DYLIB_PATH`, exactly as the CPU/CUDA runtime placement in
[`ONNX.md`](ONNX.md). A runtime without ROCm support fails closed at engine
load.

## Selecting ROCm at runtime

| Surface | Selection |
|---|---|
| Surveyed families with an ONNX export | `--<family>-backend onnx-rocm` (encoder-nli, decoder-letter, schema-scorer, qwen3guard, von, decoder-logit-qwen3, laya, encoder-instruct-label) |
| Device ordinal | `--rocm-device <N>` (default `0`, env `OPENKIND_ROCM_DEVICE`) |
| ONNX Runtime library | `--onnx-runtime <path>` or env `ORT_DYLIB_PATH` |

Families without an ONNX export (native Qwen3.5 backbone, kev, decoder-llm,
decider-4b, winnow, strands-decider, decoder-logit-qwen35, gemma4) and the
proxy-cache encoder offer no ROCm selection.

## Detection

On Linux, `openkind_runtime::detect_accelerators()` reports AMD GPUs whose
KFD topology entries map to physical PCI devices while `/dev/kfd` exists.
Each entry is matched to an
AMD display controller (PCI class `0x03`) or processing accelerator (class
`0x12`, including MI300X). Names come from the matched KFD node's `name`
file. Total VRAM comes from the physical PCI device's
`mem_info_vram_total` attribute. KFD memory banks can describe a GPU
partition, so they do not establish whole-card memory. Missing information
stays unknown, and nodes without a verified PCI identity are omitted.
The probe reads sysfs directly and needs no ROCm userland.

Discovery identifies physical cards by PCI address, for example
`rocm-pci:0000:03:00.0`. These identifiers are hardware inventory, not
values for `--rocm-device`. That flag accepts the process-visible HIP
ordinal, which can change with `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
or `CUDA_VISIBLE_DEVICES`. Backend identities such as `onnx-rocm:0` use
that execution ordinal.

Discovery does not establish execution support or HIP visibility.
`openkind_backends::device::execution_support()` reports compiled backend
support (`onnx_rocm`), and engine loading checks the runtime provider. The
daemon logs inventory and compiled support separately.

## Evidence gates (open)

Before any ROCm answer is trusted for a profile, record the following on a
real ROCm host using the [`BENCHMARKS.md`](BENCHMARKS.md) methodology:

1. A verified ONNX export for the profile per the [`ONNX.md`](ONNX.md)
   artifact contract, digested and mirrored before the run.
2. Decision argmax parity and calibrated-probability tolerance against the
   pinned CPU oracle fixtures, loaded explicitly through the ROCm execution
   path. Apply the same tolerances as the MLX gates: maximum probability
   error `0.005`, ordering tolerance `1e-5`, unchanged argmax and policy.
3. Request-path throughput and memory, labeled as `onnx-rocm:<device>`.
4. The native service load/soak gate for any service-level claim.
5. A clean-commit promotion record.

## Module map

- Device selection vocabulary: [`crates/openkind-backends/src/device.rs`](../crates/openkind-backends/src/device.rs)
- ONNX acceleration and execution-provider registration: [`crates/openkind-backends/src/onnx/mod.rs`](../crates/openkind-backends/src/onnx/mod.rs)
- Accelerator detection: [`crates/openkind-runtime/src/accelerators.rs`](../crates/openkind-runtime/src/accelerators.rs)
- Physical ROCm identities and KFD/PCI matching: [`crates/openkind-runtime/src/rocm.rs`](../crates/openkind-runtime/src/rocm.rs)
- Family execution seam: each family's `load_with_execution` entry point.

## Out of scope

- Native candle execution on ROCm (blocked upstream; revisit if candle
  grows a ROCm backend).
- The llama.cpp HIP backend (openkind has no llama.cpp dependency).
- Silent fallback to CPU on ROCm failure (violates the fail-closed rule).
- Windows ROCm (the ONNX Runtime ROCm execution provider is Linux-only).
- ROCm release packaging or container images (the ROCm runtime library is
  an explicit operator placement, like every ONNX Runtime build).
