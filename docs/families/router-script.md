# Family: router-script

> Lightweight language/script detector, no model forward pass; branches
> between sibling families.

## Status in openkind

**Surveyed — internal routing primitive.** Not a standalone decision
model; the family only routes between sibling families (for example
between an English-only encoder and a multilingual encoder). Routing
cost should be sub-millisecond and produce a deterministic branch.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | None — pure Unicode and script detection plus optional heuristic language identification |
| Tokenization | Stateless. The router inspects the state and question text, not the candidate list |
| Forward pattern | None |
| Readout | A single label: the sibling family identifier to dispatch to, plus a small set of policy fields (truncation flags, rejected-script markers) |
| Continuation state | None |
| Text generation | None |
| Calibration | N/A — the router emits a discrete branch, not a probability distribution. Probabilities come from the sibling family |
| Semantic none | Inherited from the sibling family |

## Why this family is a candidate

- It allows `openkind` to ship an English-only encoder and a multilingual
  encoder side by side without forcing the caller to pick at the wire
  level. The router maps to the sibling family automatically.
- Routing latency is microseconds; the cost is dominated by the sibling
  family's forward pass.
- No new parity fixtures are required: the router contract is a function
  of the input text and a fixed rule table, not of model output.

## What blocks implementation

- The M0 supported workload in [`../../ROADMAP.md`](../../ROADMAP.md)
  has not locked the languages or scripts the engine must serve. The
  router rule table cannot be written before that decision.
- Branching on script alone (Latin vs Cyrillic vs Han etc.) is
  sufficient for the multilingual decision but cannot tell the difference
  between, say, German and Turkish Latin-script text. The family is
  honest about that limit; a stronger language identifier would
  effectively become a small NLI-family model and would need its own
  parity fixtures.
- Routing decisions must never influence the wire semantics. The router
  sits behind the Jev wire contract documented in
  [`../../JEV_COMPATIBILITY.md`](../../JEV_COMPATIBILITY.md); any
  visible divergence between router branches would be a wire-level bug.

## Open questions

- Which scripts and languages does the M0 workload actually require?
  Owned by [`../../ROADMAP.md`](../../ROADMAP.md).
- Does the router need to participate in the M2 reviewed-decision
  contract, or only in M3 lower-cost evaluation?
- Where does the router live in the crate topology? The current
  architecture has the family selection inside the profile contract
  owned by [`../../crates/openkind-engine/AGENTS.md`](../../crates/openkind-engine/AGENTS.md)
  rather than at the daemon entry point.

## What this page does not say

No timings, no language coverage numbers, no error rates. The cited
comparison values belong to ollaya's `laya:latest` router page, not to
`openkind`. `openkind` has no router implementation to measure.
