# 2026-09-20 — qwen35-native CPU smoke sweep

First recorded `opendecision-bench` native run: 12 decisions (4 states × 3
primitives) through the full request path of `Qwen35DecisionEngine` — render,
state-first tokenization, backbone execution, readout, answer mapping — under
each execution strategy. **Smoke-scale evidence**: single sample per strategy
on a short-state fixture; not a throughput benchmark.

- Schema: `opendecision-bench/v1` (`summary-qwen35.json`)
- Engine: `qwen35-native-cpu`, profile `a047d6802c3f06f085b8`,
  model revision `1001bb4d826a52d1f399e183466143f4da7b741b`,
  bundle version `2ij.2.0`, candle-cpu-fp32
- Fixture: `crates/opendecision-bench/fixtures/decisions_smoke.jsonl`,
  SHA-256 `3a673e843690b942658b4c9de6cc594770185756098d356e78dcd3efd5cffeeb`,
  12 rows grouped per state (4 groups)
- Host: Mac16,5 Apple M4 Max 36 GiB (named Mac), warm process, untimed warmup
- Commit: `9d086107bb017bf721d815bb5bcb8ba516ce0e6e`
- Timing scope: request construction + dispatch + answer extraction; model
  load (~17.3 s per strategy instance) excluded and reported separately

## Results (single sample per strategy)

| Strategy | Total (s) | Decisions/s | Speedup vs repeated_full |
|---|---:|---:|---:|
| `repeated_full` | 124.88 | 0.096 | 1.00× |
| `nested_sequential` | 73.64 | 0.163 | 1.70× |
| `nested_batched` | 69.99 | 0.171 | 1.78× |
| `choose_strategy` | 73.70 | 0.163 | 1.69× |

`cross_strategy_answer_parity_clean: true` — all four strategies produced
identical typed answers for all 12 decisions, exercising the shared-state
correctness invariant on real checkpoint weights end to end. The measured
scheduler matched `nested_sequential` performance, consistent with the
per-lane CPU backend selecting the sequential shared path.

## Interpretation boundaries

- Sharing gains here (≈1.7×) are bounded by the short states: the fixture's
  root prefix is small relative to per-candidate suffix work. The mechanics
  grid in [`ARCHITECTURE.md`](../../ARCHITECTURE.md) shows the crossover
  grows steeply with state length (up to 18.29× at L=1024/Q=16/K=2).
- CPU parity and this parity check do not imply Metal or accelerated parity,
  and do not constitute release promotion.
- Single-sample cells: quote ratios, not absolutes; re-run with more
  repetitions for latency claims.
