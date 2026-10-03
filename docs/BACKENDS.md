# Backend releases and selection

Release archives contain `openkind` and `openkindd`. Keep the pair together:
CLI commands resolve `OPENKINDD_BINARY`, then the sibling executable, then
`PATH`. Homebrew installs the standard archive, using glibc on Linux.

| Platform | Standard archive | Separate `-onnx` archive |
|---|---|---|
| macOS arm64 | CPU + MLX FP32, including `mlx.metallib` | CPU + MLX + ONNX CPU |
| macOS Intel | CPU | CPU + ONNX CPU |
| Linux x86_64 glibc | CPU + native CUDA | CPU + native CUDA + ONNX CPU/CUDA |
| Windows x86_64 | CPU + native CUDA | CPU + native CUDA + ONNX CPU/CUDA |
| Linux x86_64 musl | Portable CPU | None |

Names follow `openkind-<version>-<target>.tar.gz` (Windows: `.zip`). ONNX
archives insert `-onnx` before the extension. Each archive has a SHA-256
sidecar. Cargo's default build remains CPU-only.

Native CUDA builds pin toolkit 12.6.0 and compute capability 8.0, targeting
Ampere or newer. The driver, CUDA user libraries, and cuDNN remain
operator-installed. [CUDA](CUDA.md) owns native build details;
[ONNX](ONNX.md) owns the pinned ORT 1.23.2 libraries and GPU requirements.
Runtime packaging downloads libraries, never model assets.

## Automatic selection

Daemon backend selectors default to `auto`. For each load, selection checks
the family's supported loaders, local artifacts, compiled features, and
successful runtime initialization. It tries:

1. MLX FP32 for a supported Apple Silicon loader.
2. Native CUDA on the configured execution ordinal.
3. ONNX CUDA when a compatible local export exists.
4. Native CPU.
5. ONNX CPU when the export is the compatible available artifact.

An export-only root omits native candidates. Every selected loader still
verifies its pinned metadata and artifact digests. The curated registry
currently publishes no ONNX exports. Existing installations require every
file in their registry manifest; adding an export does not waive those
checks. CLEF's public loader remains CPU-only. ROCm requires an explicit
selection and an operator-provided runtime.

The resolver covers startup installations, direct family aliases,
playground model loads, and proxy-cache encoders. Low-level engine loaders
remain explicit. The chosen backend is fixed for the engine's lifetime.
Startup logs record the backend and reasons for skipping or falling back.

`auto` falls back only during loading for runtime unavailability,
unsupported execution capabilities, or accelerator allocation failure.
Corrupt artifacts, profile mismatches, and invalid configuration abort the
load. Explicit selectors never fall back:

```bash
openkindd --installed-models laya-english:v1 --laya-backend native-cpu
openkindd --encoder-nli-backend onnx-cuda --cuda-device 1 \
  --models encoder-nli-native --encoder-nli-model-root /models/nli
```

These settings preserve the Jev HTTP and gRPC contracts. Runtime readiness
and automatic selection do not establish checkpoint parity or task quality.

## Diagnostics

```bash
openkind doctor --json
openkind doctor --json --cuda-device 1 --onnx-runtime /runtimes/libonnxruntime.so
openkindd --diagnose-backends --json
```

Diagnostics exit without loading models or opening listeners. JSON schema
version 1 includes hardware inventory, compiled support, initialization
results, requested execution ordinals, failure reasons, and the resolved
ORT path. NVML inventory indices can differ from execution ordinals under
`CUDA_VISIBLE_DEVICES`. Use the reported runtime result to check the
ordinal, rather than assuming inventory numbering.

Accelerator probes run in isolated child processes with a bounded timeout.
Native CUDA exercises cuBLAS and Candle kernels, MLX synchronizes an array
operation, and each ONNX provider executes the vendored tiny Gemm graph.
Missing libraries or provider panics cannot terminate the parent daemon.
ORT resolves the explicit setting, `ORT_DYLIB_PATH`, the executable-relative
bundle, then system lookup. The first initialized runtime remains selected
for that process. Restart after changing runtime placement.

## Release validation

[`build-release-artifacts.yml`](../.github/workflows/build-release-artifacts.yml)
builds all nine configurations. Separate fresh runners extract the archives
into paths containing spaces, audit binary imports, run diagnostics, and
send a mock HTTP request without CUDA/cuDNN installed. Bundled ONNX CPU
execution is required even for GPU packages: missing libraries fail the
check. macOS arm64 also requires MLX initialization. Homebrew installs and
tests each standard macOS/glibc archive before release publication.

The release-only Candle override removes the redundant CUDA linking feature
in the vendored manifest, updates its vendor checksum, and records the
original package digest and manifest hashes. No pinned architecture source
or tracked dependency lock is changed.

Packaging commands (Python 3.12; CUDA builds additionally require nvcc):

```bash
python3 -m unittest discover -s scripts/tests -p test_release.py
python3 scripts/release.py prepare-cuda
CUDA_COMPUTE_CAP=80 cargo build --release --locked \
  --config target/release-cargo-config.toml --features cuda \
  --bin openkind --bin openkindd
python3 scripts/release.py package --target x86_64-unknown-linux-gnu \
  --bin-dir target/release --version 0.1.0
python3 scripts/release.py smoke \
  dist/openkind-0.1.0-x86_64-unknown-linux-gnu.tar.gz --features cuda
```

The smoke command expects a clean host without an operational NVIDIA GPU.
Real checkpoint exports, NVIDIA decision parity, continuation and branch
isolation, throughput/memory qualification, and CUDA optimization remain
separate gates in [CUDA](CUDA.md), [ONNX](ONNX.md), and
[benchmarks](BENCHMARKS.md).
