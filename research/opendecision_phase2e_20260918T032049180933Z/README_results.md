# OpenDecision Phase 2E

Run: 20260918T032049180933Z

Stage status:
```json
{
  "indexing_selftest": "completed",
  "setup": "completed",
  "tiny_cache_api_selftest": "completed",
  "data": "completed",
  "model": "completed",
  "model_forward_smoke": "completed",
  "precision": "partial",
  "shared_prefix": "partial",
  "request_benchmark": "partial",
  "policies": "completed",
  "export": "completed",
  "report": "completed"
}
```

- Frozen Qwen3.5-4B, NLI/candidate heads, and all Phase 2D none-head coefficients. No LoRA or neural training.
- Precision islands promote whole modules and cast outputs back. This is not a custom FP32 accumulation kernel.
- NLI accuracy is a regression subset of previously reviewed Phase 2C tests, not new evidence of generalization.
- Shape inputs reuse the high-drift Phase 2D debugging sample; they cannot estimate normal production failure frequency.
- Shared cache is scoped to ONE question/request and deep-copies every mutable state tensor. No cross-request cache or question batching.
- Cache reuse changes the chunking/execution graph; numerical agreement must be measured, not assumed.
- Banking policy dev/test messages exclude C and D messages. Repeating the same seed reuses E messages.
- Banking held-out labels remain in one domain; pretraining decontamination is unknown; none is omitted annotated intent, not a safety detector.
- Absent priors and error/review costs are declared scenarios, not measured deployment rates or universal policies.
- Synthetic long-prefix results concern execution mechanics only, not long-context classification accuracy.
- Timing is single-process Python without server/network; full prefix creation and deep-copy work included; no production latency guarantee.
- No quantization, Rust/Metal, Jev HTTP, ordinal Score training, or proprietary Jev reproduction is established.