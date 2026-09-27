# Family: qwen3guard

> Decoder preset that generates a fixed safety-verdict text and is mapped
> to typed questions through an embedded question schema. Mirrors the
> `qwen3guard-gen-v1` family in ollaya's nomenclature, used there to
> serve a guardrail preset (`Safe` / `Unsafe` / `Controversial` plus
> nine category labels).

## Status in openkind

**Rust-loadable (prototype profile, Stream variant).** The pinned profile
`0fcf416cab16d94f933d` loads `Qwen/Qwen3Guard-Stream-0.6B` at
`419364a715de9840d47b1457982f64ff37f90ed4` (Apache-2.0) through a
hand-implemented Qwen3 dense architecture (per-head q/k RMSNorm, explicit
head_dim, causal attention) in
[`families/qwen3guard/`](../../crates/openkind-backends/src/families/qwen3guard/mod.rs),
FP32 on CPU, verified token-level against the PyTorch reference. It
implements `DecisionEngine`, registers in `openkindd` via
`--qwen3guard-aliases` / `--qwen3guard-model-root`, and is benchmarked
through `openkind-bench --engine qwen3-guard`.

**Architecture-invariant resolution.** The surveyed `gen` variant parses a
generated verdict string and is barred by the workspace no-generation rule.
This profile implements the **Stream** variant instead: a token-level
classification head produces the risk distribution directly from the hidden
state — no token is ever sampled. The reference scores the user-turn
position at the closing `<|im_end|>`; the profile renders
`<|im_start|>user\n{state}<|im_end|>` and reads the query-side risk head.

**Wire mapping.** `Noul` maps to the `Unsafe` class probability (the
`Controversial` class is not surfaced in the scalar answer; serve a Choice
question for the full distribution). `Choice` options map by label
(`safe`/`unsafe`/`controversial`, case-insensitive) to their class logits,
and the reserved `__none__` key maps to the `Controversial` class — the
checkpoint's failure-to-decide mass; offering both `controversial` and
`__none__` fails closed. `Score` questions are rejected: the fixed preset
has no ordinal contract. Probability space:
`ConditionalOnOfferedOptions` (softmax over offered class logits). No M0/M2
reviewed-decision evidence exists for this profile.

## Architectural shape

The shape below describes the **surveyed `gen` variant** — the ollaya
`qwen3guard-gen-v1` pattern. The landed Rust-loadable profile is the
**Stream variant** described under Status above: a token-level
classification head, no generated text, no string parsing.

| Aspect | Pattern |
|---|---|
| Backbone class | Decoder-only fine-tune of a small Qwen-3-class backbone, trained to emit a fixed safety-verdict text in a chat-template-bound format |
| Tokenization | Hard-coded chat template carrying the policy, the nine category labels, and the verdict output format; the user message is only the text to judge |
| Forward pattern | Single forward pass per request; the model emits a short verdict (`Safety: Safe\|Unsafe\|Controversial` plus a categories line) |
| Readout | String-parse the verdict back into a typed distribution over the verdict classes plus the nine categories; mapping onto `Choice` / `Score` / `Noul` happens at the runtime boundary |
| Continuation state | Decoder KV cache; no DeltaNet or convolution state on the published Qwen-3 0.6B backbone |
| Text generation | Yes — limited to the verdict text. `openkind`'s general rule against autoregressive text generation is family-specific, not global; this family is the documented exception and is reviewed per profile |
| Calibration | Per-profile temperature fitted offline; calibrated probabilities are derived from the verdict-string parse, not from raw logits |
| Semantic none | The `Controversial` class absorbs the failure-to-decide mass; `openkind`'s `__none__` semantics are owned by [`encoder-state-first.md`](encoder-state-first.md) and any profile here would have to be reviewed against them |

## Why this family is a candidate

- Guardrail decisions are a common request shape on top of decision
  models and would let `openkind` serve a known customer use case
  directly without forcing the caller to write their own guardrail
  classifier.
- A guardrail-specialised head could be a smaller model than the
  current encoder-state-first profile and could serve as a candidate
  for the M3 lower-cost gate in [`../ROADMAP.md`](../ROADMAP.md).
- The family ships as a fixed-preset model (the question schema is
  embedded), which simplifies the wire contract for the caller and
  removes the need for arbitrary-question prompt engineering.

## What blocks the generated-verdict (`gen`) variant

The Stream variant implemented above sidesteps every item here. They gate
only a hypothetical `gen` profile that generates and parses verdict text:
- `openkind`'s general architecture rule
  ([`../ARCHITECTURE.md`](../ARCHITECTURE.md)) forbids
  autoregressive text generation in the engine. Adopting this family
  would be a documented family-level exception, not a relaxation of
  the global rule; the exception would need its own review against
  the M0/M1/M2 milestone sequence in
  [`../ROADMAP.md`](../ROADMAP.md).
- The published upstream preset ignores arbitrary custom
  instructions; any profile here can answer only the embedded
  question schema. Questions outside the embedded schema must be
  rejected at the wire with a 422.
- The M0 supported workload in [`../ROADMAP.md`](../ROADMAP.md)
  has not selected guardrail decisions as in-scope, and no guardrail
  training or evaluation contract is owned by `openkind`.
- The verdict is parsed back from generated text, which adds a
  parsing failure mode absent from the implemented
  encoder-state-first family. The parsing contract would need its
  own fixtures and review.

## Open questions

- Does the M0 supported workload in
  [`../ROADMAP.md`](../ROADMAP.md) authorize guardrail-shaped
  decisions?
- How would the family's guardrail class set interact with the
  `__none__` semantic-none contract owned by the implemented
  encoder-state-first family?
- Does the verdict-parse failure mode belong in the wire error model
  owned by [`../../crates/openkind-core/AGENTS.md`](../../crates/openkind-core/AGENTS.md),
  or as a profile-level rejection? That decision is unresolved.

## What this page does not say

No model-quality or accuracy numbers, no guardrail-class distributions. The
cited comparison values belong to ollaya's family page on
`qwen3guard-gen-v1`, not to `openkind`. Request-path timing and peak-RSS
measurements for the pinned Stream profile are recorded in
[`../BENCHMARKS.md`](../BENCHMARKS.md).
