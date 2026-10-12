# Family: sieve

> The sthanika-ai Sieve checkpoints prompt every question as a tag-delimited
> branch over a fenced `<state>` prefix and read a trained 255-row float32
> head (`head.safetensors`) at the final branch position. The serving
> checkpoint is a LoRA adapter over a pinned base plus the head; attention
> stays fully causal, so the prefix-plus-branch row is one ordinary causal
> sequence. No token is ever sampled.

## Status in openkind

**Rust-loadable (prototype profile).** The pinned profile
`sieve-4b:8b97b56524b55c41fbf4` serves `sthanika-ai/Sieve-4B` (Apache-2.0)
at `ea7cd1edee29590a82ed9fbfcb644227a8eed5f1`: a rank-32 LoRA (alpha 64,
scale 2.0, the full hybrid projection set) over `Qwen/Qwen3.5-4B` at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` — the pinned qwen3.5 dims — plus
a trained `[255, 2560]` float32 head.

The loader verifies every artifact by SHA-256 in place (base `config.json`,
adapter, head, `sieve_config.json`, tokenizer, tokenizer config,
provenance, license), merges the adapter once per load through the shared
LoRA-merged backend (PEFT `merge_and_unload` semantics:
`base + (B @ A) · alpha/rank`, first proven on the strands-decider family),
and applies the head at the final branch position. Profiles serve through
`openkindd` via `--sieve-aliases` / `--sieve-model-root` and are
benchmarked through `openkind-bench score --engine sieve-4b`.
Catalog-installable offline-first: `openkind pull sieve-4b:8b97b56524b55c41fbf4`
(alias `sieve:4b`) downloads the base shards, the adapter, and the head and
verifies each digest.

Profile semantics:

- Probability space: `ConditionalOnOfferedOptions` — the per-type
  temperature softmax over the offered options' code logits sums to one
  (`sieve_config.json`: choice 1.1608, noul 1.4403, score 1.0). An offered
  `__none__` key is scored as an ordinary option.
- Continuation state: none — every branch is an independent
  prefix-plus-branch forward (the reference's shared-prefix KV cache is a
  serving optimization this family does not need).
- Text generation: none.
- Noul: the reference answers every Noul question twice — Yes-first and
  No-first listing — and averages the distributions to cancel listing
  bias; this family reproduces that.

## Contract details the renderer reproduces

- **State fencing** (`defang`): user text cannot emit special tokens
  (`<|`/`|>`) or close the `<state>`/`<question>` tags; option lines are
  single-line with literal `\n` escapes.
- **Piecewise prefix encoding**: the reference encodes the prefix head and
  the state piece separately; whole-string encoding would merge the
  boundary newlines of an empty state into one token.
- **Code table**: 255 single-token codes verified with the reference's
  three checks (single token, line-start stability, post-tail stability)
  and pinned against `sieve_config.json` `codes`/`code_ids`.
- **Choice option lines**: `key — description` (em dash) with the wire's
  sorted label order — the declared determinism difference; object states
  serialize in the wire's hash-key order.

## Reference evidence

- **Prompt byte-parity.** Seven fixture cases (choice, choice with
  `__none__`, noul default and described criteria, score levels, empty
  state) rendered by the Rust engine and by the shipped `sieve4b/prompt.py`
  over the checkpoint's real tokenizer produce **byte-identical token
  sequences** for every branch. The reference run is recorded in
  [`research/handoffs/registry-expansion-2026-10-12/sieve-4b-reference-prompts.json`](../../research/handoffs/registry-expansion-2026-10-12/sieve-4b-reference-prompts.json);
  the engine-side stats are pinned in the golden fixture
  (`prompt_stats`).
- **LoRA merge.** The merge backend is the one first proven on the
  strands-decider family; the sieve adapter's shapes and digests are
  verified at load, and the delta is recomputed per load in FP32.
- **Golden replay.** `crates/openkind-backends/tests/sieve_4b_parity.rs`
  replays the committed golden fixture against the pinned artifacts
  (env-gated via `OPENKIND_SIEVE_MODEL_ROOT`); always-on tests pin the
  profile constants, the code-table derivation, and the fail-closed load
  path. No torch cross-check ran for the forward pass (no torch on this
  host); the backbone kernels are the parity-verified CPU oracle, and the
  answer distributions on the fixture cases behave as the state text
  implies.

## Benchmarks

Measured request-path benchmarks are pending; the profile ships with
research status and no task-quality claim.

## Sources

- Checkpoint: [`sthanika-ai/Sieve-4B`](https://huggingface.co/sthanika-ai/Sieve-4B)
  (Apache-2.0) at `ea7cd1edee29590a82ed9fbfcb644227a8eed5f1`; base
  [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) at
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Family survey: [`registry/v1/jev-decision-index-0.3.1.json`](../../registry/v1/jev-decision-index-0.3.1.json).
- Reference code shipped beside the checkpoint (`sieve4b` package,
  Apache-2.0).
