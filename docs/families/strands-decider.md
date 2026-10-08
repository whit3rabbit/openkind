# Family: strands-decider

> LoRA adapter + pointer head over the Qwen3.5-2B hybrid text backbone,
> serving the published Strands Decider 2B (Hobson v19 and v21) decision contracts.

## Status in openkind

**Implemented (2026-10-01).** AWS's Strands team released Hobson v19 as
Strands Decider 2B with the adapter, pointer head, tokenizer, serving code,
training recipes, and evaluation records under Apache-2.0
(`StrandsAgents/strands-decider-2B-hobson-v19`,
github.com/strands-labs/strands-decider). The pinned-source review lives in
[`../RESEARCH.md`](../RESEARCH.md) (reviewed 2026-10-01); this page records
the openkind implementation, which carries no model-quality claim from that
review.

This family is distinct from [`decider`](./decider.md): Mapika's decider
checkpoints read tied-output-embedding slot logits off a merged checkpoint,
while the Strands readout scores each option from its own hidden state with
a separate pointer head. A changed readout is a new family per
[`NEW_FAMILY.md`](./NEW_FAMILY.md), not a second `decider` profile.

It is catalog-installable offline-first: `openkind pull
strands-decider-2b:6a02bb0d1c6b25cae74b` or `openkind pull
strands-decider-2b:f7156bf28400a79ea1b8` downloads the pinned artifacts,
verifies every SHA-256, and installs them for `--installed-models` (see
[`../MODELS.md`](../MODELS.md)).

Two pinned profiles are supported:
- **Hobson v19**: `StrandsAgents/strands-decider-2B-hobson-v19` at
  revision `bb282d786bc251fd4e3068de3ada9ddbb38127cd` over
  `Qwen/Qwen3.5-2B-Base` at `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`,
  profile ID `6a02bb0d1c6b25cae74b`.
- **Hobson v21**: `StrandsAgents/strands-decider-2B-hobson-v21` at
  revision `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5` over
  `Qwen/Qwen3.5-2B-Base` at `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`,
  profile ID `f7156bf28400a79ea1b8`.

The release's `provenance.json` labels that base revision "inferred" (training hosts did not pin it); it is the
base repository's current and only revision, and these profiles pin the exact
bytes verified. The profiles implement `DecisionEngine` behind the
bounded family scaffold, register in `openkindd` via
`--strands-decider-aliases` / `--strands-decider-model-root` /
`--strands-decider-base-root`, and are benchmarked through
`openkind-bench --engine strands-decider-2b`.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Qwen3.5-2B hybrid text decoder (24 layers: 18 gated-DeltaNet + 6 full attention at interval 4, hidden 2048, GQA 8/2 heads, head 256) with a rank-16 LoRA (α=32, scale 2.0) merged into every attention, DeltaNet, and MLP projection at load |
| Tokenization | One row per question: the `<state>` document first (keep-first truncated to the remaining window), then the question block — `<question type="...">`, the reference header, the rendered instructions, options numbered from 1 as one line each (`"{n}. {name} — {description}"`, whitespace-collapsed), closing `</question><answer>`. Option read positions come from the tokenizer's offset mapping |
| Window policy | The reference `_fit`: the question claims up to 3072 tokens of the 4096 window first (front-truncated, keeping the tail, so options and the `<answer>` marker survive); the state receives the rest |
| Forward pattern | Prefill-only causal forward; no text generation and no LM head. The reference's shared-prefix KV path is answer-equivalent to full-row encoding (pinned by the reference's own tests); openkind encodes the full row |
| Readout | Pointer head in FP32: LayerNorm (torch default ε 1e-5), query from the final `<answer>` position, keys from each option's last token, score `q(h_answer) · k(h_option) / sqrt(256)`; softmax over offered options |
| Continuation state | None — rows are independent; nothing is retained across questions or requests |
| Text generation | None |
| Calibration | Per-type temperatures shipped fitted in the release config (`hobson_config.json` for v19: choice 0.7342, score 1.3278, noul 0.9107; `strands_decider_config.json` for v21: choice 0.8177, score 1.1922, noul 0.8177), pinned verbatim and enforced against the artifact at load |
| Semantic none | ConditionalOnOfferedOptions — no none mass is invented; an offered `__none__` key is scored as an ordinary option |

## Implementation notes

- The Qwen3.5 backbone reuses the parity-verified shared `TextBackbone` with
  a new `Qwen35Geometry::BASE_2B` const. The base checkpoint ships the
  multimodal container layout (`model.language_model.*` tensor prefix, plus
  unused `model.visual.*` and `mtp.*` tensors the loader never reads), which
  is the layout the shared backbone already expects.
- The LoRA merge follows the `kev` family's PEFT pattern (`lora_A`/`lora_B`
  merged as `W += (B @ A) · α/r`), applied through a `VarBuilder` backend
  view over the memory-mapped checkpoint so the merge happens on tensor read
  and no merged checkpoint file is ever written. Adapter keys map
  `base_model.model.layers.N.…` onto the checkpoint's
  `model.language_model.layers.N.…` names.
- The reference serves Choice options in caller insertion order; our wire
  criteria map is a hash map, so options render in the wire's sorted label
  order — a declared determinism difference, as in the other surveyed
  families. The reference also re-fits the window per 32-question batch;
  openkind budgets from the longest question in the whole request, which is
  identical for every request the reference would process in one batch.
- The head's `head.safetensors` ships from the release as safetensors (FP32
  `norm`/`q`/`k` tensors), so no conversion step exists.

## Validation

- Digest-checked golden fixtures over 16 stated-fact decision cases
  (including a structured object state and a multi-question request):
  `crates/openkind-backends/tests/fixtures/strands_decider_6a02bb0d1c6b25cae74b/`.
  Every case assigns the correct stated option the highest probability.
- Cross-check against the pinned PyTorch reference
  (`strands-labs/strands-decider` at `f91487ab…`, transformers 5.17.0 +
  peft 0.21.1, CPU FP32, identical prompts): argmax agrees in 17/17
  answers. Noul and Score probability drift stays ≤ 8.5e-4, inside the
  0.005 parity gate. Choice drifts 2e-5 to 5.9e-2, one-sided (the reference
  is sharper): the 0.734 choice temperature amplifies a small
  input-dependent logit difference between the two hybrid-recurrence
  implementations. The fixture provenance and the
  [benchmark record](../benchmarks/2026-10-01-strands-decider/README.md)
  record the residual; the golden replay remains the self-consistency
  regression gate.
- The release's JevBench numbers (167/231 public tasks) belong to the
  authors and are development-exposed observations, not openkind
  measurements; see the [research review](../RESEARCH.md) for the evidence
  limits.

## Limits

- `MAX_ROW_TOKENS = 4096`, `MAX_QUESTION_TOKENS = 3072` (the reference
  window and question fraction); encoded rows beyond the window fail closed.
  The reference truncates a too-long state keep-first, the same policy. An
  option list that truncation destroys fails the load here.
- The pointer head has no slot ceiling (option count is bounded by the
  window and the wire's 255-option maximum).
- Zero-shot quality is unmeasured here. No M2 useful-decision gate has run
  for this profile. The release's own preregistrations also show
  retention/benchmark guards failing on later versions. Treat task-quality
  claims as unqualified until measured on openkind's own protocol.

## What this page does not say

No claim that this profile reproduces the authors' hosted behavior beyond
the checkpoint's own wire compatibility; no M2 model-quality evidence; no
comparison against Mapika's `decider-2b` or the 4B `decider-4b` profile.
