# Family: winnow

> Decoder backbone + LoRA, script- and language-aware router via the
> label-logit readout. Mirrors the `winnow-v1` family in ollaya's
> nomenclature, used there to pick between sibling `laya` models.

## Status in openkind

**Rust-loadable (prototype profile, trained in-house).** The pinned profile
`4dff8c5b03cfbf680db6` fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` at
`7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0) with a rank-8 LoRA
(scale 20, last 8 layers, 300 iterations, Adam, lr 1e-4) trained with MLX on
a synthetic 800-example English/multilingual routing corpus owned by this
repository (validation loss 0.030). The adapter is vendored in-repo at
[`tests/fixtures/winnow_adapter/adapters.safetensors`](../../crates/openkind-backends/tests/fixtures/winnow_adapter/adapters.safetensors)
and digest-pinned; the Rust loader merges the LoRA delta into the base
weights at load ([`families/winnow/`](../../crates/openkind-backends/src/families/winnow/mod.rs)),
serves through `openkindd` via `--winnow-aliases` / `--winnow-model-root` /
`--winnow-adapter` / `--winnow-siblings "A=<alias>,B=<alias>"`, and is
benchmarked through `openkind-bench --engine winnow` (over mock siblings,
measuring the learned routing pass only).

It is catalog-installable offline-first: `openkind pull winnow:4dff8c5b03cfbf680db6` downloads the pinned artifacts, verifies every SHA-256, and installs them for `--installed-models` (see [`../MODELS.md`](../MODELS.md)).

The sibling target set locked for this profile: label `A` (english) and
label `B` (multilingual) map to served sibling aliases at the daemon layer;
the routing distribution is telemetry, and the routed sibling's answer is
the wire answer. Routing runs one forward pass per request with the frozen
trained prompt; no token is ever sampled. This resolves the page's
prerequisite (a locked sibling target set exists across the surveyed-family
profiles), and the licensed training data is synthetic and owned here. No
M2 reviewed-decision gate has run for the router itself; its operating point
is provisional.

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

## What remains open

- The sibling target set is locked for this profile (daemon default
  `A=decoder-letter-native`, `B=encoder-nli-native`); the survey-era
  prerequisite is resolved.
- Learned routers are themselves decision models and need their own
  reviewed operating point. They cannot ride on the M2 gate inherited
  from a sibling family; none has run for this router.
- The LoRA training-data and license story was reviewed for this
  profile: the corpus is synthetic and owned by this repository. Any
  future backbone needs the same review against the license posture in
  [`../../crates/openkind-backends/AGENTS.md`](../../crates/openkind-backends/AGENTS.md).

## Open questions

- Is a learned router strictly better than the rule-based
  [`router-script.md`](router-script.md) on the M1 repaired evidence
  contract?
- Can the router be served at the same cost as the script-based
  alternative? Cost methodology is owned by
  [`../BENCHMARKS.md`](../BENCHMARKS.md).
- Where does the router sit relative to the Jev wire contract documented
  in [`../JEV_COMPATIBILITY.md`](../JEV_COMPATIBILITY.md)?

## What this page does not say

No model-quality or accuracy numbers, no routing error rates. The cited
comparison values belong to ollaya's family page on `winnow`, not to
`openkind`. Routing-cost measurements over mock siblings are recorded in
[`../BENCHMARKS.md`](../BENCHMARKS.md).
