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
provenance, license), enforces the LoRA pins below, merges the adapter once
per load through the shared LoRA-merged backend (PEFT `merge_and_unload`
semantics: `base + (B @ A) · alpha/rank`, first proven on the
strands-decider family), and applies the head at the final branch
position. Profiles serve through
`openkindd` via `--sieve-aliases` / `--sieve-model-root` and are
benchmarked through `openkind-bench score --engine sieve-4b`.
Catalog-installable offline-first: `openkind pull sieve-4b:8b97b56524b55c41fbf4`
(alias `sieve:4b`) downloads the base shards, the adapter, and the head and
verifies each digest.

LoRA pins: `adapter_config.json` must declare `peft_type` `LORA`, base
`Qwen/Qwen3.5-4B`, `r` 32, `lora_alpha` 64, `bias` `none`, `fan_in_fan_out`
false, `use_rslora` and `use_dora` false, empty `alpha_pattern` and
`rank_pattern`, `lora_bias` false, and null `modules_to_save`,
`layer_replication`, and `trainable_token_indices`. Each of these would
change the merged weights. The profile's merge scale must equal
`lora_alpha / r` (2.0). An always-on unit test checks the pins against a
vendored copy of the pinned file.

Resident memory: the shared backend keeps each target projection's delta
`(B @ A) · 2.0` as a dense FP32 tensor on the selected device for the
engine's lifetime. For sieve-4b that is 3,569,090,560 elements (all twelve
target modules in all 32 layers), about 14.3 GB (13.3 GiB), on top of the
memory-mapped base shards; the adapter tensors are not retained after the
merge. Every forward re-reads each layer's weights and adds its delta, so
a merged layer exists only while that layer runs.

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
- Option counts: Choice serves 1 to 255 options (the reference accepts a
  single option); Score serves 2 to 10 levels.
- Token limits (reference `contract.py`): the state plus the request's
  longest question branch is capped at 32,768 tokens, and the state once
  plus every question's longest branch at 65,536. The fixed preamble head
  (system prompt through `<state>\n`) counts toward neither. Every question
  is validated and rendered, and both caps are checked, before any forward.
  An over-length request fails as an unsupported request (HTTP 422
  `invalid_body`, gRPC `INVALID_ARGUMENT`) with the reference's message
  text; the reference itself answers 400 `max_tokens_exceeded`.
- Usage (reference `engine.py`, default `bill_preamble=False`):
  `input_tokens` is the state tokens once plus each question's longest
  branch, so noul's two listings bill as one branch. `output_tokens` stays
  0; the reference bills the token count of its serialized answers.

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
- **Choice option lines**: `key — description` (the reference's em dash
  character) in the wire's sorted label order, the declared determinism
  difference.
- **Object-state key order**: object states serialize with keys sorted
  byte-lexicographically at every nesting level (openkind's canonical
  state keys), while the reference's `json.dumps` keeps the client's
  insertion order. Object-state prompts match the reference only for
  clients that send keys already sorted; text states are unaffected. The
  one object-state fixture case (`noul-default-criteria`) uses sorted keys.

## Reference evidence

- **Prompt byte-parity.** Seven fixture cases (choice, choice with
  `__none__`, noul default and described criteria, score levels, empty
  state) rendered by the Rust engine and by the shipped `sieve4b/prompt.py`
  over the checkpoint's real tokenizer produce **byte-identical token
  sequences** for every branch. The reference run is recorded in
  [`research/handoffs/registry-expansion-2026-10-12/sieve-4b-reference-prompts.json`](../../research/handoffs/registry-expansion-2026-10-12/sieve-4b-reference-prompts.json);
  the engine-side stats are pinned in the golden fixture
  (`prompt_stats`). The fixture generator renders those rows through the
  engine's request planner (`plan_sieve_request`), and
  `planner_reproduces_the_recorded_prompt_stats_and_usage` (gated on
  `OPENKIND_SIEVE_MODEL_ROOT`; it reads only `checkpoint/tokenizer.json`
  and `checkpoint/sieve_config.json`) replays every case through that
  planner, checking the prefix length, every row digest, and the billed
  `input_tokens`.
- **LoRA merge.** The merge backend is the one first proven on the
  strands-decider family; the sieve adapter's shapes, digests, and merge
  settings are verified at load, and the delta is recomputed per load in
  FP32. Always-on unit tests cover the backend's merge arithmetic and its
  fail-closed target and shape checks on synthetic tensors.
- **Golden replay.** `crates/openkind-backends/tests/sieve_4b_parity.rs`
  replays the committed golden fixture against the pinned artifacts
  (env-gated via `OPENKIND_SIEVE_MODEL_ROOT`). The goldens are a Rust
  self-replay that pins this engine's own outputs on the fixture cases.
  They are not a forward-pass cross-check: no torch reference forward has
  run (no torch on the generating host), and the backbone kernels are the
  parity-verified CPU oracle. Always-on tests pin the profile constants,
  the token caps and billing rule, request planning, the adapter pins, and
  the fail-closed load paths (missing artifacts and a digest mismatch).
  The code-table test runs the derivation over a synthetic WordLevel
  tokenizer, so it checks the selection and ordering logic only; the real
  table is checked at load against `sieve_config.json`.

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
