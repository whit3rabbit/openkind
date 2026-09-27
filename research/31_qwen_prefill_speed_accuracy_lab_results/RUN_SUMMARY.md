# OpenKind modern Qwen MoE / PrivateMode-style decision study

Status: **EXPLORATORY_COMPLETE**

Run key: `e15e1e9f7a64e464a38354c59b0c79805d59bc13d517f7e4fca66873e5d5ff2e`

Model and cache promotion: **false**.

Timing includes one generated answer-code token and localhost HTTP overhead. Earlier direct-forward PyTorch timings are historical only.
SNLI is public and conditioned on three-label premise groups; synthetic templates are shared across splits. Neither is a production generalization gate.

| arm | dataset | n | accuracy | nll | coverage | accepted_error | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|---|
| qwen35_4b | snli | 288 | 0.7535 | 0.5791 | 0.0000 | undefined | 63.1026 | 65.9620 |
| qwen35_4b | synthetic | 288 | 0.7569 | 0.4031 | 0.5903 | 0.0235 | 63.3983 | 67.2732 |
| qwen35_moe_int4 | snli | 288 | 0.8472 | 0.4754 | 0.0000 | undefined | 126.3202 | 131.9207 |
| qwen35_moe_int4 | synthetic | 288 | 0.9271 | 0.1498 | 0.8819 | 0.0039 | 126.7273 | 130.7483 |

## Cache qualification
- qwen35_4b: {"full_concurrent": false, "cold_staged": false, "warm_state": false}; option-order changes 8/24.
- qwen35_moe_int4: {"full_concurrent": false, "cold_staged": false, "warm_state": false}; option-order changes 3/24.

Only QUALIFIED_MEASUREMENT performance rows may support a cache/concurrency speed claim. Warm timings exclude the separately reported state-prime time. Different quantizations/backends remain different profiles.

## Sources
- https://www.privatemode.ai/blog/system-one-from-glm-flash
- https://github.com/edgelesssys/privatemode-decisions
- Model repositories and immutable revisions: manifest.json. Original engine commands and startup logs are retained.

A completed study can contain failed scientific gates. Rerun only an incomplete matching identity; never mix changed code, model, data or runtime in an old run.