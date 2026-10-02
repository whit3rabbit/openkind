# openkind-backends

> Model artifact loaders and parity-checked decision readouts for `openkind`.

`openkind-backends` houses model-facing loaders, forward-pass components,
and the direct native adapter behind the `DecisionEngine` trait defined in
`openkind-engine`.

## Current status: native CPU reference path through direct service registration

`openkind-runtime` owns the backend-neutral continuation-state contract.
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

The CPU path now also has bounded model-backed K=32/64/128/255 execution,
checkpoint-backed fresh-process feature/decision replay, cooperative
cancellation of already-running native compute, and named-machine daemon
load/soak evidence in the [RUST11 report](../../docs/verification/native-service-gate/2026-09-22-rerun2/README.md).
Practical high-K latency, model-quality review, and release promotion remain
open. A cancelled caller retains its queue and execution permits until the
blocking native work actually finishes.

The optional pinned-base MLX path passes FP32 full, nested, and variable-length
vectorized batch parity. Native BF16 fails the frozen probability tolerance in
both full and nested modes. A forced FP32 vectorized daemon request and bounded
unified-memory admission/recovery measurements are recorded in the
[Phase 3M follow-up](../../docs/verification/phase3m-2026-09-22/README.md).
Automatic vectorized scheduling and the packed Metal kernel remain opt-in
pending matched performance evidence. MLX service load/soak and production
promotion remain open. CPU native parity does not imply MLX or accelerated
production promotion.

The branchable Qwen state includes attention KV, DeltaNet recurrent state,
and convolution state. A KV-only abstraction is incomplete.

## Verification

Default offline tests verify the tracked Phase 3B identities and golden-vector
digests, exact embedding widening, frozen-head probability/policy replay,
renderer semantics, cardinality and input limits. The renderer unit tests use
a deterministic synthetic byte tokenizer. The reference subset's provenance
and file digests are in [its source record](tests/fixtures/qwen35_backbone_phase3b_a047d6802c3f06f085b8/SOURCE.md).

The two pretrained-tokenizer tests and the catalog tokenizer-byte check require
an explicitly supplied artifact. They remain separate qualifications rather
than downloads in the default suite. With the already acquired Qwen3.5-4B-Base
tokenizer (19,989,325 bytes, SHA-256
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`), run:

```bash
export OPENKIND_QWEN35_TOKENIZER=path/to/pinned/tokenizer.json
cargo test --locked -p openkind-backends --test qwen35_tokenizer_parity -- --ignored
cargo test --locked -p openkind-model-store --test catalog -- --ignored
```

These commands verify the artifact's digest, exact exported segment IDs,
renderer limits and catalog size/hash. They do not run full checkpoint or
accelerated inference. Missing or changed artifacts fail qualification.

```bash
cargo test -p openkind-backends

# No checkpoint required: admission and accounting at K = 32/64/128/255.
cargo run -p openkind-backends --example qwen35_scheduler_stress

# Requires an already-downloaded immutable base checkpoint. Never downloads.
cargo run -p openkind-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results \
  trace

cargo run --release -p openkind-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results \
  continuation

# Two separate invocations form the checkpoint-gated fresh-process replay.
cargo run --release -p openkind-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results \
  persist-save path/to/root-state.bin

cargo run --release -p openkind-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results \
  persist-replay path/to/root-state.bin \
  crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8

cargo run --release -p openkind-backends --example qwen35_parity_probe -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results \
  branch

cargo run --release -p openkind-backends --example qwen35_full_parity -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results \
  crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8

cargo run --release -p openkind-backends --example qwen35_nested_parity -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results \
  crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8

cargo run --release -p openkind-backends --example qwen35_batched_parity -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results \
  crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8

# Phase 3.8 measurement harness: roughly 35-45 minutes at default settings.
cargo run --release -p openkind-backends --example qwen35_scheduler_bench -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results

# Model-backed streaming stress. Requires the pinned checkpoint.
cargo run --release -p openkind-backends --example qwen35_model_stress -- \
  path/to/checkpoint \
  research/14_phase3b_backbone_parity_results
```

The embedding-only probe hashes the complete 5.3 GB first shard. Decoder,
continuation, branch, full, nested, and batched modes verify both immutable
shards, about 9.3 GB total, before execution. None of these commands download
model assets.

The fresh-process replay filters the token fixtures to the persisted root's
fixture case. It compares every restored candidate feature with an independent
full-sequence forward (maximum absolute difference `1e-4`), then compares the
full head probabilities (`0.005`), argmax, and policy action. The replay fails
if the saved trace is missing, fixture segments do not reconstruct the exported
full sequences, or any decision differs.
