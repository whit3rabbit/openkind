# OpenDecision Phase 2D

Run: 20260917T234417Z

Source: 20260917T222948Z

Numerical verdicts: {'bf16_default': 'needs_review', 'bf16_math': 'needs_review', 'bf16_strict_math': 'needs_review', 'fp32_strict_math': 'within_sampled_tolerance'}

Development-selected none model: set_linear

## Interpretation boundaries
- Frozen Qwen3.5 and frozen NLI/candidate heads. Only small none heads and optional temperature are fitted.
- The numerical sample deliberately includes old high-drift examples: debugging, not a frequency estimate.
- Fresh dynamic test messages exclude every Phase 2C message; rerunning this data_seed reuses them.
- No Qwen pretraining decontamination or cross-domain validity is established.
- None means omitted annotated banking intent, not a universal unknown/safety detector.
- Lexical hard negatives are label-description based, not model-mined hard negatives.
- Paired tests are 50% absent. Weighted training/selection uses a declared 25% absent prior, not a measured deployment prior.
- The scorer re-encodes the state for each candidate. No KV branching, generation, LoRA, quantization, Rust/Metal or HTTP validation.
- Timing is a bounded single-process experiment; percentile estimates are descriptive, not production latency guarantees.