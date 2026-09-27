# Family: kev

> LoRA adapter + pointer head on a Qwen base, serving the published
> decision-model contract.

## Status in openkind

**Implemented (unblocked 2026-09-26).** The family's original blocker —
"`openkind` does not own the `kev` training pipeline, weights, or evaluation
contract, and no open checkpoint reproducing the reference exists" — lapsed
when the reference author published the checkpoints, the reference
implementation, the training configs, and the evaluation protocol under
Apache-2.0 (`jaredpalmer/kev-*`, github.com/jaredpalmer/kev). The family page
itself records the unblock condition: implement once the reference is owned
or granted. Apache-2.0 grants the implementation and redistribution rights,
and the published `head.pt` / `training_config.json` / `provenance.json`
make the contract readable instead of invented.

The pinned profile is the small member, `jaredpalmer/kev-0.6b` at revision
`dece6dba8d43f0f7ded45e9f5b9df12474d90843` on `Qwen/Qwen3-0.6B-Base` at
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd` (the exact base revision the
checkpoint's `head.pt` records), profile ID `39d88c11faeb4ac165fa`. It
implements `DecisionEngine` behind the bounded family scaffold, registers in
`openkindd` via `--kev-aliases` / `--kev-model-root` / `--kev-base-root`,
and is benchmarked through `openkind-bench --engine kev`.

## Architectural shape

| Aspect | Pattern |
|---|---|
| Backbone class | Qwen3 dense decoder (0.6B: 28 layers, hidden 1024, GQA 16/8 heads, head 128) with a rank-16 LoRA merged into every attention and MLP projection at load |
| Tokenization | One row per question: the state document after `<\|fim_prefix\|>`, then `<\|fim_middle\|>` instruction, per option `<\|box_start\|>` option `<\|box_end\|>`, then the `<\|fim_suffix\|>` decide token. Caller text is rewritten (`<\|name\|>` → `<¦name¦>`) so option boundaries are unforgeable |
| Forward pattern | The reference's *row form*: one causal row of state + branch per question, proven equivalent to the packed block-causal form by the reference's own tests. No text generation — the backbone runs prefill-only with no LM head |
| Readout | Pointer head: option score `k(h_opt) · q(h_decide) / 16` where `h_opt` is the hidden state at each option's `<\|box_end\|>` and `h_decide` the `<\|fim_suffix\|>` state; softmax over offered options |
| Continuation state | None — rows are independent; the packed block-causal form would prefill the state once (openkind's own continuation-state machinery maps naturally onto the reference's prefix cache, left unimplemented here) |
| Text generation | None |
| Calibration | The checkpoint ships raw logits (no fitted temperature recorded; published ECE numbers are raw). The openkind NLL fit is degenerate in the `qwen3guard` sense (the merged model already assigns ≥ 0.95 to the correct side of 14 of 15 stated-fact cases), so `T = 1.0` is pinned with that rationale |
| Semantic none | The reference contract's none-of-the-above handling is part of its training data, not its head; the profile treats a reserved `__none__` key as an ordinary offered option |

## Implementation notes

- The Qwen3 backbone reuses the hand-rolled implementation the `qwen3guard`
  family validated; the LoRA merge follows the `winnow` family's pattern
  (PEFT `lora_A`/`lora_B` merged into the base weights once at load, scale
  α/r = 32/16). The merged weights are bit-identical to the reference's
  PyTorch merge.
- The checkpoint's `head.pt` is a torch pickle. The profile pins a one-time
  operator conversion to `head.safetensors` (the four FP32 tensors copied
  verbatim); the loader verifies the converted digest and, when the
  original `head.pt` is present, also verifies it against its recorded
  source digest before refusing a pickle load.
- `Choice` options are evaluated in the shared sorted-label order; the
  reference evaluates in caller order and trains with option-permutation
  augmentation, so the ordering is quality-neutral and owned by the profile.

## Validation

- Digest-checked golden fixtures over 15 stated-fact decision cases:
  `crates/openkind-backends/tests/fixtures/kev_39d88c11faeb4ac165fa/`.
- Cross-implementation check against a PyTorch reference running the same
  merged weights: probabilities agree to `≤ 3e-4` on 14 of 15 cases and
  `≤ 1.9e-2` on the single near-degenerate case (top-two logits within
  1.4), argmax preserved everywhere. The residual is fp32 GEMM
  accumulation-order difference between Apple Accelerate (candle) and MKL
  (PyTorch): merged weights are bit-identical and the per-layer divergence
  starts at the first GEMM.

## Benchmark record

`openkind-bench score` over the standard shape777 workload (777 rows):
4.84 decisions/s, 4.22 GB peak RSS, 5.61 s model load; recorded in
[`../BENCHMARKS.md`](../BENCHMARKS.md) and
[`benchmarks/2026-09-26-surveyed-families/`](../benchmarks/2026-09-26-surveyed-families/).
Request-path timing only; the published model-card accuracies (0.801
in-distribution / 0.620 out-of-domain on the author's suites) belong to the
upstream evaluation and are not `openkind` measurements.

## Limits

- `MAX_STATE_TOKENS = 384`, `MAX_ROW_TOKENS = 1024` (the reference training
  context); over-long requests fail closed, truncation is forbidden. The
  reference's serving context reaches 64k states; extending the profile
  there is unverified.
- The pointer head envelope is the trained `max_num_classes` equivalent
  (255 per the reference API) bounded in practice by the row limit.
- Zero-shot outside its training distribution: the checkpoint is a 0.6B
  model with published out-of-domain accuracy ~0.62 and near-chance
  held-out policy reasoning. The M2 useful-decision gate in
  [`../ROADMAP.md`](../ROADMAP.md) has not run for this profile.

## What this page does not say

No claim that this profile reproduces TypeSafe's hosted reference behavior
beyond the checkpoint author's own wire-compatibility statement; no M2
model-quality evidence; no benchmark comparison against the larger Kev
checkpoints.
