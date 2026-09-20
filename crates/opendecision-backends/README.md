# opendecision-backends

> Model artifact loaders and parity-checked decision readouts for `opendecision`.

`opendecision-backends` houses model-facing loaders and forward-pass components that will eventually sit behind the `DecisionEngine` trait defined in `opendecision-engine`.

## Current status: Phase 3.1 through 3.5 CPU parity, branch-state contract, and sequential nested execution complete

The `branch` module defines the backend-neutral continuation-state contract, and the `qwen35` module implements the deterministic feature-to-probability slice for selected profile `a047d6802c3f06f085b8`:

- fail-closed reference metadata and safetensors validation.
- f64 normalization, linear projection, score-summary rejection, temperature calibration, and stable softmax.
- native semantic-none probability for Choice only.
- frozen threshold policy returning accept or review.
- offline replay of four exported golden fixtures.
- digest-locked Qwen tokenizer loading.
- exact state-first root, question, candidate-suffix, and full-sequence IDs.
- validated Phase 3B architecture, 47-vector safetensors export, and 34-stage
  diagnostic trace.
- max-absolute, RMS, and cosine comparison helpers for backbone bring-up.
- verified loading of both pinned checkpoint shards.
- exact BF16-to-FP32 parity for the diagnostic sequence's last-token embedding.
- FP32 Candle CPU execution through 24 DeltaNet and 8 full-attention blocks.
- final RMSNorm, all 34 diagnostic stages, and 10 native candidate features.
- all four Phase 3B probability/argmax/policy gates.
- immutable Qwen continuation state containing attention KV, DeltaNet
  recurrent state, convolution state, and absolute position.
- profile/model/tokenizer/renderer/arithmetic state identity, structural and
  strict content fingerprints, exact hybrid byte accounting, immutable-root
  fork, batched fork, and gather/select through `BranchableState`.
- sequential nested `state → question → candidate` execution through
  `run_sequential_nested`: one immutable state prefill, question forks,
  per-candidate forks, fail-closed position and immutability checks, and
  exact replay determinism.

This establishes the frozen Phase 3.3 CPU backbone fixtures, the Phase 3.4
backend-neutral branch-state contract, and Phase 3.5 sequential nested
execution parity for the CPU path. Metal, batched nested Q/K execution,
backend registration, and Jev wire mapping
remain later gates. It is not a claim of full Rust parity or release
promotion. CPU native parity does not imply Metal or accelerated parity.

The branchable Qwen state includes attention KV, DeltaNet recurrent state,
and convolution state. A KV-only abstraction is incomplete.

## Verification

```bash
cargo test -p opendecision-backends

# Requires an already-downloaded immutable base checkpoint. Never downloads.
cargo run -p opendecision-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  trace

cargo run --release -p opendecision-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  continuation

cargo run --release -p opendecision-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  branch

cargo run --release -p opendecision-backends --example qwen35_full_parity -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  crates/opendecision-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8

cargo run --release -p opendecision-backends --example qwen35_nested_parity -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  crates/opendecision-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8
```

The embedding-only probe hashes the complete 5.3 GB first shard. Decoder,
continuation, branch, full, and nested modes verify both immutable shards,
about 9.3 GB total, before execution. None of these commands download model assets.
