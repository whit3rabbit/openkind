# opendecision-backends

> Model artifact loaders and parity-checked decision readouts for `opendecision`.

`opendecision-backends` houses model-facing loaders, forward-pass components,
and the direct native adapter behind the `DecisionEngine` trait defined in
`opendecision-engine`.

## Current status: native CPU reference path through direct service registration

`opendecision-runtime` owns the backend-neutral continuation-state contract.
The compatibility `branch` module re-exports that contract, and `qwen35`
implements the deterministic feature-to-probability slice for selected profile
`a047d6802c3f06f085b8`:

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
- profile/model/tokenizer/renderer/arithmetic state identity, distinct
  scheduling and content fingerprints, exact tensor-payload accounting, immutable-root
  fork, batched fork, and gather/select through `BranchableState`.
- sequential nested `state → question → candidate` execution through
  `run_sequential_nested`: one immutable state prefill, question forks,
  per-candidate forks, fail-closed position and immutability checks, and
  exact replay determinism.
- breadth-first batched Q/K execution through `run_batched_nested`:
  `fork_batch` question lanes from one immutable root, per-question candidate
  fan-outs, fail-closed root/sibling/position checks, exact sequential-baseline
  parity, and fan-out byte accounting.
- the Phase 3.8 strategy layer: `run_strategy`/`run_repeated_full` across all
  three parity-proven strategies with forward-call and staged-token
  accounting, and `choose_strategy` selecting by the lowest measured crossover
  ratio (2.52), declared vectorization capabilities, lane limits, a retained
  tensor ceiling, and a process-memory envelope.
- state/scheduler high-cardinality stress cases at K = 32, 64, 128, and 255.
- a tenant-isolated, TTL-bound, tensor-byte-limited branch-state cache contract.
- an atomic, versioned, digest-checked pinned-state snapshot whose restore gate
  verifies execution identity, exact layer layout, and strict content identity.
- direct `Qwen35DecisionEngine` registration using explicit offline artifact
  paths, bounded concurrency/queueing, explicit Choice `__none__`, and
  normalized-entropy confidence. Process admission refreshes loaded peak RSS at
  request time and apportions remaining headroom across concurrent executions.

This establishes the frozen Phase 3.3 CPU backbone fixtures, the Phase 3.4
backend-neutral branch-state contract, Phase 3.5 sequential nested
execution parity, Phase 3.6/3.7 batched Q/K parity, and the Phase 3.8
measured scheduler for the CPU path. The current per-lane CPU backend declares
no vectorized suffix capability, so its scheduler uses nested-sequential when
sharing is admitted. `nested_batched` remains a state/lane topology with exact
parity, not a claim of vectorized compute.

Model-backed high-K execution, checkpoint-backed fresh-process replay,
cooperative cancellation of already-running native compute, load/soak, Metal,
and release promotion remain later gates. A cancelled caller retains its queue
and execution permits until the blocking native work actually finishes.
CPU native parity does not imply Metal or accelerated parity.

The branchable Qwen state includes attention KV, DeltaNet recurrent state,
and convolution state. A KV-only abstraction is incomplete.

## Verification

```bash
cargo test -p opendecision-backends

# No checkpoint required: admission and accounting at K = 32/64/128/255.
cargo run -p opendecision-backends --example qwen35_scheduler_stress

# Requires an already-downloaded immutable base checkpoint. Never downloads.
cargo run -p opendecision-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  trace

cargo run --release -p opendecision-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  continuation

# Two separate invocations form the checkpoint-gated fresh-process replay.
cargo run --release -p opendecision-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  persist-save path/to/root-state.bin

cargo run --release -p opendecision-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  persist-replay path/to/root-state.bin

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

cargo run --release -p opendecision-backends --example qwen35_batched_parity -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z \
  crates/opendecision-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8

# Phase 3.8 measurement harness: roughly 35-45 minutes at default settings.
cargo run --release -p opendecision-backends --example qwen35_scheduler_bench -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z

# Model-backed streaming stress. Requires the pinned checkpoint.
cargo run --release -p opendecision-backends --example qwen35_model_stress -- \
  path/to/checkpoint \
  research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z
```

The embedding-only probe hashes the complete 5.3 GB first shard. Decoder,
continuation, branch, full, nested, and batched modes verify both immutable
shards, about 9.3 GB total, before execution. None of these commands download
model assets.
