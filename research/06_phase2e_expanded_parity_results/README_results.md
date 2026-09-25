# OpenKind Phase 2E — expanded execution review
Version: 2e.2.0
Run: 20260918T114914072764Z
Execution status: completed

## Readout
Inspect `parity_summary` for probabilities, selected classes, and answer/review output changes. Completion does not imply equivalence.
Inspect independent complete-request timings separately from intrusive component profiles.
FP32/BF16 workers are separate foreground processes; the notebook coordinator never owns the model.
No head coefficients or acceptance thresholds were fitted in this run. Archived tests are reused for regression only.

## Files
`experiment_inputs.json`: exact episode panel and frozen policy manifest.
`<mode>/memory_snapshots` in worker_report.json: tensor, allocator, and driver memory snapshots.
`<mode>/parity_rows.json`: raw scores, all three probability models, policy decisions and directed disagreements.
`<mode>/component_profiles.json`: synchronized component wall times.
`<mode>/request_benchmarks.json`: independent uninstrumented cold-request timings.
`cross_precision.json`: between-mode full-sequential comparison; FP32 is not assumed more accurate.

## Limitations
- All Qwen weights, C/D heads and E acceptance policies are frozen; no refitting or threshold search.
- Tests reuse archived E messages and selected policies: execution regression, not independent generalization.
- Complete factorial variants share messages; episode counts are not independent sample counts.
- FP32 loads in an independent fresh subprocess; no GPU offload and no in-place precision sweep before it.
- Fresh-process isolation does not guarantee enough physical GPU memory; preflight and OOM are reported.
- Equal-length suffix batching expands all hybrid state with repeated beam indices; no suffix padding.
- Profile component times include synchronization and are intrusive; uninstrumented timings are reported separately.
- All request timings use a cold prefix; no cross-request cache, Rust/Metal or HTTP benchmark.
- A passed numerical tolerance is not a safety guarantee; mode completion is distinct from equivalence acceptance.
- Long-prefix mechanics use synthetic token sequences; they do not measure long-context decision quality.
