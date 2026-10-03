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

Builds stay offline: `ort` runs with `load-dynamic`, so nothing downloads at
build time and the shared library is an operator placement:

1. `--onnx-runtime <path>` on the daemon (sets `ORT_DYLIB_PATH`).
2. `ORT_DYLIB_PATH` in the environment.
3. Standard library directories (`/opt/homebrew/lib`, `/usr/local/lib`,
   `/usr/lib`, `~/.local/lib`), by file name `libonnxruntime.dylib` /
   `libonnxruntime.so` / `onnxruntime.dll`.

A missing or unloadable library fails closed at engine load with the path in
the error. Install a release from the
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
