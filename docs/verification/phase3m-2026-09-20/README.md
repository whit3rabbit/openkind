# Phase 3M verification record — MLX/Metal parity backend (first pass, 3M.0–3M.4)

Recorded 20 September 2026 on the named development Mac (`Christophers-MacBook-Pro-2`,
M4 Max class host per earlier Phase 3 records). Working tree at base commit
`413acdc10540773a9c8aedeb2e568a55f955646a` plus the Phase 3M changes
(uncommitted at recording time — the git commit fields inside the reports
name the base commit; re-run the gates at the landing commit to make this
evidence commit-stamped).

This directory is a historical bring-up record, not current landing evidence.
The MLX formal parity and qualification commands now refuse `--formal` on a
dirty worktree, so these reports must be regenerated after the implementation
is committed.

The current dirty-tree real-checkpoint rerun is recorded in
[`../phase3m-2026-09-21-working-tree.md`](../phase3m-2026-09-21-working-tree.md).
It confirms the FP32 gates but leaves the BF16 candidate open: the full path
is just outside the probability tolerance and nested continuation rejects a
layer-4 cache dtype. The BF16 pass claims below describe this historical
bring-up snapshot only.

## Runtime identity

- mlx-rs `0.32.0` / mlx-sys `0.6.0` (vendored mlx-c `v0.6.0-7-gc74db53`, MLX core `0.32.2`), linked statically from `libmlx.a`/`libmlxc.a`.
- Xcode 27.0 (27A266a), Metal toolchain `32023.921 (metalfe-32023.921.6)`, macOS 26.6.2 (25G83), rustc 1.98.1.
- Full archive/metallib SHA-256 hashes and memory snapshot: [`qwen35_mlx_runtime_provenance.json`](qwen35_mlx_runtime_provenance.json).
- Toolchain relevance: the open mlx-c issue #115 (BF16 gather→RMSNorm→matmul corruption under an Xcode 26.5 build) does **not** reproduce on this Xcode 27.0/Metal 32023.921 build — both precision paths passed the 3M.0 primitive gate, including the exact issue-#115 constant-tensor reproduction (first element ≈ 1024, not 512).

## Gate results (all runs `--release`, `--features mlx`, offline pinned checkpoint)

| Gate | Result | Key numbers |
|---|---|---|
| 3M.0 runtime qualification (fp32 + bf16 primitives, #115 repro) | PASS | see provenance; exit 0 |
| 3M.2 embedding exactness (Phase 3B trace input) | PASS (bit-exact) | `embedding_exact: true`, max_abs `0` |
| 3M.3 layer_00 / 34-stage trace (fp32) | PASS (diagnostics) | layer_00 max_abs `3.34e-06`, final norm max_abs `4.01e-05`, rms `8.02e-06` |
| 3M.4 full 10-candidate decision gate (fp32) | PASS | probability max `4.5169e-06` (tolerance 0.005), 0 argmax, 0 policy changes |
| 3M.4 sequential nested continuation (fp32) | PASS | probability max `2.3747e-06`, 0/0, root immutability (strict fingerprint) holds and equals an independent prefill, positions match fixtures, sibling isolation holds, cached-vs-full `7.44e-05` within the 1e-4 Phase 3B guard |
| bf16 candidate profile — full gate B | PASS | probability max `1.4603e-03` (within frozen 0.005), 0 argmax, 0 policy; feature max `0.1215` |
| bf16 candidate profile — nested gate B | PASS (decision gates) | probability max `1.4447e-03`, 0/0, immutability/isolation hold; cached-vs-full `2.37e-04` reported as a diagnostic (the 1e-4 self-consistency guard is an fp32-reference assertion, not a bf16 decision gate) |

Reports: `qwen35_mlx_full_parity_{fp32,bf16}_*.json`, `qwen35_mlx_nested_parity_{fp32,bf16}_*.json`.

Weight loading streamed tensor-by-tensor from both digest-verified shards
(~4.3 s, peak active MLX ≈ 16.1 GB in fp32, peak process RSS ≈ 10–11 GB;
the checkpoint stores `A_log` and `linear_attn.norm.weight` in FP32 and all
other tensors in BF16 — both are read exactly).

## Findings recorded during bring-up

1. **MLX `conv1d` is true convolution** (kernel reversed) with weight layout `(C_out, K, C_in/groups)`, unlike torch/candle cross-correlation. The reference recurrence therefore implements the causal depthwise conv explicitly over its per-token window; a qualification test pins the convention.
2. **mlx-c streams are thread-affine** (per-thread stream registries). `MlxRuntime` serializes all MLX work behind one execution mutex on per-thread default GPU streams; `mlx_stream_new_thread_unsafe` is the designated future refinement.
3. A conv-window shift bug (keeping the oldest column instead of dropping it) was found by the offline synthetic differential test and fixed before any model gate ran; the differential test remains in the suite as a permanent transcription guard.
4. Arithmetic identities `mlx-core-0.32.2/fp32/reference-ops` and `mlx-core-0.32.2/bf16/reference-ops` (kernel family included) are distinct from `candle-cpu-fp32`, so cross-backend state mixing fails closed by construction.

## Scope boundary

This is MLX **parity** evidence for the frozen Phase 3B fixtures on one named
runtime build. It is not: fused-Gated-DeltaNet-kernel evidence (the recurrent
layers run the per-token reference path over ordinary array ops), performance
evidence, vectorized batch-forward evidence, daemon integration, quantization,
or release promotion. Candle CPU remains the correctness oracle. Rebuilding
mlx-c under a different Xcode/Metal toolchain is a different runtime identity
and must re-run 3M.0 before any model gate.
