# opendecision-backends

> Model artifact loaders and parity-checked decision readouts for `opendecision`.

`opendecision-backends` houses model-facing loaders and forward-pass components that will eventually sit behind the `DecisionEngine` trait defined in `opendecision-engine`.

## Current Status: Phase 3.1

The `qwen35` module implements the deterministic feature-to-probability slice for selected profile `a047d6802c3f06f085b8`:

- fail-closed reference metadata and safetensors validation;
- f64 normalization, linear projection, score-summary rejection, temperature calibration, and stable softmax;
- native semantic-none probability for Choice only;
- frozen threshold policy returning accept or review;
- offline replay of four exported golden fixtures.

This is head/probability parity only. Tokenizer execution, state-first rendering, Qwen inference, hardware acceleration, backend registration, and Jev wire mapping remain later phases.

The future branchable Qwen state must include attention KV, DeltaNet recurrent state, and convolution state. A KV-only abstraction is incomplete.

## Verification

```bash
cargo test -p opendecision-backends
```
