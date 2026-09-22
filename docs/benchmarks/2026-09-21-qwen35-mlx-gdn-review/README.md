# Qwen 3.5 MLX Gated-DeltaNet review benchmark

This directory preserves the raw `opendecision-bench/v1` summaries and
per-strategy predictions used by the Phase 3M.5 production-default decision.
Both runs are dirty-working-tree, single-sample measurements anchored to
subject baseline `e492bab3e4428c0413365ed9d0699ebcb114c320`. They are not
clean-commit release evidence.

## Environment

- Host: `Mac16,5 Apple M4 Max 36 GiB (named Mac)`
- Runtime: MLX core 0.32.2, `mlx-rs 0.32.0`, vendored mlx-c
  `v0.6.0-7-gc74db53`
- Profile: `a047d6802c3f06f085b8`
- Checkpoint: pinned `Qwen/Qwen3.5-4B-Base` revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`
- Fixture: 12 rows over four states, SHA-256
  `3a673e843690b942658b4c9de6cc594770185756098d356e78dcd3efd5cffeeb`
- Timing: one timed repetition after warmup; model load and result writes
  excluded from strategy totals

## Results

| Arithmetic path | `repeated_full` | `nested_sequential` | `nested_batched` | `choose_strategy` | Peak RSS |
|---|---:|---:|---:|---:|---:|
| `reference-ops` | 29.998354292 s | 7.818630042 s | 7.860438125 s | 7.755205417 s | 11,838,046,208 bytes |
| `metal-tree-packed-dk128-v1` | 38.278743458 s | 9.054690417 s | 8.956889167 s | 9.060100959 s | 11,894,095,872 bytes |

The packed candidate is 1.14 to 1.28 times slower, so production retains
`ReferenceOps`. The candidate separately passes the frozen full and nested
parity gates. This benchmark rejects promotion on the tested workload; it
does not establish a stable performance distribution.

The v1 summary schema does not yet carry the arithmetic-family identifier.
The enclosing directory names and the linked verification record identify
which build produced each artifact.

## Files

- [`reference-ops/`](reference-ops/): final production-default rerun.
- [`metal-tree-packed-dk128-v1/`](metal-tree-packed-dk128-v1/): opt-in packed
  FP32 sequence-kernel candidate.
- [`../../verification/phase3m5-2026-09-21-working-tree.md`](../../verification/phase3m5-2026-09-21-working-tree.md): exact command shape, parity results,
  implementation review, and evidence boundaries.
- [`../../BENCHMARKS.md`](../../BENCHMARKS.md): canonical timing methodology
  and cross-run interpretation.

