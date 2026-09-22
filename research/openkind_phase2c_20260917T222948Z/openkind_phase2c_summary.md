# OpenKind Phase 2C — measured readout

Run: 20260917T222948Z
GPU: NVIDIA L4; dtype: torch.bfloat16
Stability status: needs_review
NLI development-selected head: last_linear_seed17
Dynamic stage: completed
LoRA stage: disabled

## Selected NLI results
- test_matched: accuracy 0.8700; NLL 0.3400; ECE 0.0194; temperature 1.0000
- test_mismatched: accuracy 0.8880; NLL 0.3246; ECE 0.0247; temperature 1.0000

## Limitations
- Fresh tests are fresh relative to Phase 2B; repeating this notebook's seed reuses them.
- No Qwen pretraining decontamination is established.
- Multiple head seeds do not measure every source of uncertainty.
- Numerical tolerance is an engineering diagnostic, not a task-safety guarantee.
- Dynamic head is trained in one banking-intent domain, with sampled distractors.
- Explicit none means omitted annotated intent; not a universal unknown/abstention detector.
- No shared-prefix branching, ordinal Score training, Rust/Metal execution, or HTTP compatibility test.
- LoRA comparison, when enabled, is one adapter seed and a different optimization budget.