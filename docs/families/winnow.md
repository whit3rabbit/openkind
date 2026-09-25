# Family: winnow

> Decoder backbone + LoRA, script- and language-aware router via the
> label-logit readout. Mirrors the `winnow-v1` family in ollaya's
> nomenclature, used there to pick between sibling `laya` models.

## Status in openkind

**Surveyed.** No profile, no vendored parity fixtures, no daemon
registration. Implementation would require a new profile id, new
parity fixtures, and a fresh review through the active milestone sequence
in [`../../ROADMAP.md`](../../ROADMAP.md).

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Decoder backbone (Gemma 4-class) with a LoRA adapter trained against sibling-model decision text |
| Tokenization | One prompt: the state plus lettered routing labels (`A. english / B. multilingual / …`) |
| Forward pattern | Single forward pass; next-token logits restricted to letter tokens |
| Readout | Argmax over letter logits → sibling-family identifier |
| Continuation state | KV cache only |
| Text generation | None — answer is the routed label |
| Calibration | Per-LoRA temperature; the routed label is treated as a probability over sibling families |
| Semantic none | A `__none__` routing label means "do not route; reject the request" |

## Why this family is a candidate

- A learned router can pick up signals (topical domain, formality,
  mixed-script text) that a script-based router
  (see [`router-script.md`](router-script.md)) cannot detect.
- The router is small enough to be served alongside the main sibling
  family without doubling the latency budget.

## What blocks implementation

- The M0 supported workload in [`../../ROADMAP.md`](../../ROADMAP.md)
  has not selected the sibling families the router would dispatch to.
  Choosing a router before its target set is locked would invert the
  evidence order.
- Learned routers are themselves decision models and need their own
  reviewed operating point. They cannot ride on the M2 gate inherited
  from a sibling family.
- The LoRA training data and license story for any candidate backbone
  would have to be reviewed before the family could match the
  off-the-shelf license posture in
  [`../../crates/openkind-backends/AGENTS.md`](../../crates/openkind-backends/AGENTS.md).

## Open questions

- Is a learned router strictly better than the rule-based
  [`router-script.md`](router-script.md) on the M1 repaired evidence
  contract?
- Can the router be served at the same cost as the script-based
  alternative? Cost methodology is owned by
  [`../../BENCHMARKS.md`](../../BENCHMARKS.md).
- Where does the router sit relative to the Jev wire contract documented
  in [`../../JEV_COMPATIBILITY.md`](../../JEV_COMPATIBILITY.md)?

## What this page does not say

No timings, no accuracy numbers, no routing error rates. The cited
comparison values belong to ollaya's family page on `winnow`, not to
`openkind`. `openkind` has no measurements to report.
