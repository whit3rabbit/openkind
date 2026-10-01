# MLX Backend Internals

The backend guide keeps the execution-safety rules. This file holds checkpoint and kernel details that are useful during MLX implementation work. Keep qualification claims aligned with [`docs/MLX.md`](../../docs/MLX.md) and [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).

## Checkpoint layouts

- The pinned checkpoint stores `A_log` and `linear_attn.norm.weight` in FP32. The other weights use BF16.
- The MLX-community export uses `language_model.model.*` and `vision_tower.*` prefixes, a `__metadata__` safetensors header, `[C, K, 1]` convolution layout, and BF16 `linear_attn.norm.weight`.
- LayerNorm weights fold `(1 + w)` at load. DeltaNet `norm.weight` stays raw.
- `MlxSurveyCheckpoint` selects a surveyed family's own digest-verified artifacts. The loader does not infer the family from checkpoint names.

## Operators and branch state

- MLX `conv1d` performs true convolution with reversed kernels and weight shape `(C_out, K, C_in/groups)`. The recurrent path implements causal convolution explicitly.
- An MLX `Array` clone is a refcounted handle. Arrays are immutable, so branch isolation is structural and checked with strict fingerprints.
- Arithmetic identities include precision and kernel family, for example `mlx-core-0.32.2/fp32/reference-ops`. MLX state must not mix with Candle state.

## Kernels and qualification

- Generic masked/vector-gate and packed FP32 reduction-tree kernels are opt-in candidates. `ReferenceOps` remains the default because the packed sequence candidate was 14–28% slower in the recorded smoke sweep.
- Native BF16 uses `ReferenceOps` because its fused candidate failed the frozen model gate. Treat BF16 as a separately gated candidate, never as equivalent to the FP32 oracle.
- A different Xcode or Metal toolchain is a different runtime. Re-run `qwen35_mlx_qualify` before relying on MLX gates after a toolchain change.
- Surveyed-family MLX paths use Candle CPU as the oracle and golden-fixture replay gates with 0.005 probability tolerance and zero selection flips.
