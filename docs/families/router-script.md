# Family: router-script

> Lightweight language/script detector, no model forward pass; branches
> between sibling families.

## Status in openkind

**Rust-loadable (prototype).** The router is implemented as a composite
[`DecisionEngine`](../../crates/openkind-backends/src/families/router_script/engine.rs):
Unicode script detection over the state plus question text, a fixed rule
table, and delegation to registered sibling engines. It registers in
`openkindd` via `--router-script-aliases` and
`--router-script-rules "script=sibling,...,default=sibling"`; the rule table
fails construction closed when a referenced sibling alias is not served
through `--models`. It is benchmarked through `openkind-bench --engine
router-script` (over mock siblings, recording routing overhead only).

The rule table cannot distinguish languages sharing a script (German and
Turkish Latin text both route as `latin`); that limit is inherent to script
detection and is preserved rather than papered over. Routing never changes
wire semantics: `request.model` and answer shapes are the routed sibling's,
and the router adds no admission control of its own.

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

## What remains open

- The supported workload scope
  has not locked the languages or scripts the engine must serve. The
  shipped rule table is provisional (daemon default:
  `latin`, `cyrillic`, and `default` lanes over decoder-letter and
  encoder-nli siblings); future qualification owns the final script scope.
- Branching on script alone (Latin vs Cyrillic vs Han etc.) is
  sufficient for the multilingual decision but cannot tell the difference
  between, say, German and Turkish Latin-script text. The family is
  honest about that limit; a stronger language identifier would
  effectively become a small NLI-family model and would need its own
  parity fixtures.
- Routing decisions must never influence the wire semantics. The router
  sits behind the Jev wire contract documented in
  [`../JEV_COMPATIBILITY.md`](../JEV_COMPATIBILITY.md); any
  visible divergence between router branches would be a wire-level bug.

## Open questions

- Which scripts and languages does the supported workload actually require?
- Does the router need to participate in the M2 reviewed-decision
  contract, or only in M3 lower-cost evaluation?
- Where does the router live in the crate topology? The current
  architecture has the family selection inside the profile contract
  owned by [`../../crates/openkind-engine/AGENTS.md`](../../crates/openkind-engine/AGENTS.md)
  rather than at the daemon entry point.

## What this page does not say

No language coverage numbers, no error rates. The cited
comparison values belong to ollaya's `laya:latest` router page, not to
`openkind`. Routing-overhead measurements over mock siblings are recorded
in [`../BENCHMARKS.md`](../BENCHMARKS.md).
