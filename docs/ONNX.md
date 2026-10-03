# ONNX backend

> Implementation guide for ONNX artifact execution through ONNX Runtime.
> Benchmark methodology and recorded numbers remain canonical in
> [`BENCHMARKS.md`](BENCHMARKS.md).

## Status

ONNX execution is a build-time option behind `--features onnx` (CPU execution
provider), `--features onnx-cuda` (CUDA execution provider), and
`--features onnx-rocm` (AMD ROCm execution provider, Linux only; see
[`ROCM.md`](ROCM.md)). Family engines reuse their pinned tokenizer, renderer,
calibration, and wire mapping; only the forward pass swaps candle for ONNX
Runtime through the [`ort`](https://crates.io/crates/ort) bindings pinned at
`2.0.0-rc.13`.

An ONNX artifact is a **separate candidate identity**, exactly like the MLX
profiles: no ONNX export has run the parity gates, so ONNX answers do not
inherit CPU or MLX parity claims. The load-time signature check and the
gates below are the promotion path.

## Runtime library

Cargo builds stay offline: `ort` uses `load-dynamic` and never downloads a
runtime. Separate `-onnx` release archives bundle ONNX Runtime 1.23.2: CPU
packages on macOS and GPU packages on Linux glibc and Windows. URLs, byte
sizes, and SHA-256 digests are pinned in
[`packaging/onnxruntime.json`](../packaging/onnxruntime.json). Provider
libraries and upstream license notices accompany the runtime.

Library resolution is:

1. `--onnx-runtime <path>` on the daemon (sets `ORT_DYLIB_PATH`).
2. `ORT_DYLIB_PATH` in the environment.
3. Executable-relative `lib/onnxruntime` (Windows: DLLs beside `openkindd`).
4. Standard library directories (`/opt/homebrew/lib`, `/usr/local/lib`,
   `/usr/lib`, `~/.local/lib`), by file name `libonnxruntime.dylib` /
   `libonnxruntime.so` / `onnxruntime.dll`.

An explicit path is authoritative, including when it is missing. One runtime
is initialized per process. Explicit ONNX selections fail when initialization
fails; `auto` excludes unavailable providers during loading.

After a runtime initialization error, repair the library installation or path
and restart `openkindd` (or the embedding process). OpenKind retains the first
typed initialization failure because the pinned `ort` loader cannot safely
retry a failed library load in the same process. Artifact validation failures
before runtime initialization remain independent of this process-wide state.

NVIDIA drivers, CUDA user libraries, and cuDNN are operator-installed. The
bundled ORT 1.23.2 GPU package requires CUDA 12.8 or newer and cuDNN 9.x,
following [ORT requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
This differs from the native CUDA compilation toolkit (12.6). Install a release from the
[ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
page or the system package manager. For the CUDA execution provider the
library must be a CUDA-enabled ONNX Runtime build, and for the ROCm
execution provider a ROCm-enabled Linux build (default CPU builds report
the provider unavailable and loads fail closed). No prebuilt distribution
ships ROCm; that library must be built from source.

## Artifact contract

Each ONNX-capable family loads `model.onnx` and its companion
`model.onnx.manifest.json` from the same model root as the pinned checkpoint.
The export must use that checkpoint and the family's pinned tokenizer.
ONNX loading requires the pinned configuration, calibration, and tokenizer
files, plus the export manifest. Native checkpoint weights may be omitted
from an export-only directory; any native weights present still undergo
their pinned digest checks. Installed profiles also require their complete
registry manifest, verified before selection.
The manifest pins byte sizes and SHA-256 digests for the graph and every
external weight file, including weights referenced by nested graphs.
Paths must stay within the model root. Files are verified in place.

The manifest has this shape (replace the example sizes and digest values
with the reviewed export's values):

```json
{
  "schema_version": 1,
  "files": {
    "model.onnx": { "size_bytes": 1234, "sha256": "<64 hex digits>" },
    "model.onnx.data": { "size_bytes": 5678, "sha256": "<64 hex digits>" }
  }
}
```

Omit the external-weight entry for a self-contained graph. Keep exports
read-only during verification and execution. Local digest verification
establishes export integrity; it does not prove fidelity to the original
checkpoint or qualify the model's answers.

Names, element types, ranks, and compatible dimensions are checked at load.
Dynamic readout dimensions are checked again when executing a request:

| Family | Inputs (int64 `[1, L]`) | Output (float32) |
|---|---|---|
| encoder-nli | `input_ids`, `attention_mask` | `logits` `[1, 3]` |
| schema-scorer | `input_ids`, `attention_mask`, `token_type_ids` | `logits` `[1, 1]` |
| qwen3guard | `input_ids`, `attention_mask` | `logits` `[1, 3]` |
| decoder-letter | `input_ids`, `attention_mask` | `logits` `[1, L, vocab]`, last position read |
| decoder-logit-qwen3 | `input_ids`, `attention_mask` | `logits` `[1, L, vocab]`, last position read |
| laya | `input_ids`, `attention_mask`, `question_type` `[1]`, `marker_positions` `[K]` | `logits` `[1, K]` |
| von | `input_ids`, `attention_mask`, `marker_positions` `[K]` | `logits` `[1, K]` |
| encoder-instruct-label | `input_ids`, `attention_mask`, `marker_positions` `[K]` | `logits` `[1, K]` |

For laya, `question_type` selects choice `0`, score `1`, or noul `2`.
Marker positions follow the renderer's token order. Each graph includes
the full family readout head and emits logits before engine calibration.

Families whose readout is not representable as a single ONNX graph, or whose
format has no ONNX counterpart, do not accept ONNX selections and fail
closed with that explanation:

- **qwen35 / decider-4b / decoder-logit-qwen35 (JevK5, plumb-4b)**: the
  Qwen3.5 hybrid DeltaNet backbone has no ONNX export. Use `cuda`.
- **decoder-logit-llm**: the pinned artifact is a quantized GGUF checkpoint,
  not an ONNX graph. Use `cuda`.
- **kev**: the pointer-head readout over per-row hidden states has no pinned
  full-graph export.
- **winnow**: the router is a decoder plus engine-level routing logic.
- **router-script**: no model weights.

Selecting `onnx` for such a family returns
`execution backend unavailable` with that explanation instead of a silent
fallback.

## Selecting ONNX at runtime

```
--encoder-nli-backend onnx          # CPU execution provider
--encoder-nli-backend onnx-cuda     # CUDA execution provider (--cuda-device N)
--encoder-nli-backend onnx-rocm     # AMD ROCm execution provider (--rocm-device N, Linux)
--onnx-runtime /path/to/libonnxruntime.dylib
```

`--<family>-backend` accepts `onnx`/`onnx-cuda` (and `onnx-rocm`, Linux
only) for the families in the table above (`encoder-nli`, `decoder-letter`,
`schema-scorer`, `qwen3guard`, `von`, `decoder-logit-qwen3`, `laya`,
`encoder-instruct-label`). Backend identities read `encoder-nli/onnx-cpu`,
`encoder-nli/onnx-cuda:0`, or `encoder-nli/onnx-rocm:1`.

## Detection

`openkind doctor --json` and `openkindd --diagnose-backends --json` execute
the tiny Gemm graph in isolated provider probes without loading a model or
opening listeners. A missing runtime fails the release packaging check,
which never skips this execution. Diagnostic readiness does not establish
checkpoint parity. Automatic selection and archive layouts are documented
in [backend releases](BACKENDS.md).

The daemon startup log reports the compiled ONNX support (`onnx`,
`onnx_cuda`, `onnx_rocm`) next to the hardware detection. The search result
of the runtime library is visible through
`openkind_backends::onnx::resolve_dylib_path`.

## Testing

The ONNX module ships an offline functional test that hand-encodes a minimal
Gemm graph as ONNX protobuf, commits a session through the real runtime
library, and verifies execution plus signature enforcement:

```bash
cargo test -p openkind-backends --features onnx
```

The execution test skips when no runtime library is discoverable. Manifest
integrity, path confinement, and concurrent missing-library error tests run
without a runtime or checkpoint.

## Evidence gates (open)

Per family, before ONNX answers are trusted:

1. Export `model.onnx` from the pinned checkpoint (opset and exporter
   recorded in the family page).
2. Decision argmax parity and calibrated-probability tolerance against the
   pinned CPU oracle fixtures.
3. Publish the verified export and its digest manifest in the public mirror,
   and pin their identity in the family page following
   [`MODEL_REGISTRY.md`](MODEL_REGISTRY.md).
4. Throughput and memory labeled `onnx-cpu` / `onnx-cuda`.
