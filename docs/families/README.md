# openkind — Decision-model families

> Catalogue of decision-model architectures considered by `openkind`. Each
> family page describes one architectural approach to producing typed
> `Noul` / `Choice` / `Score` answers from a shared state document without
> autoregressive text generation, and records whether `openkind` has
> implemented or evaluated it.

This directory is the **survey** of architectures that the `openkind` engine
might host. It is intentionally broader than the current implementation.

## Canonical ownership rules

Per [`docs/AGENTS.md`](../AGENTS.md), this directory does not duplicate facts
that already live elsewhere. Each family page links to the canonical owner
for every claim it makes:

| Subject | Canonical owner |
|---|---|
| Current implementation status, milestones, and remaining work | [`../ROADMAP.md`](../ROADMAP.md) |
| Landed crate boundaries and module topology | [`../ARCHITECTURE.md`](../ARCHITECTURE.md) |
| Benchmark methodology and recorded numbers | [`../BENCHMARKS.md`](../BENCHMARKS.md) |
| Landed MLX runtime contract and limitations | [`../MLX.md`](../MLX.md) |
| Research dossier and prior-art evidence | [`../RESEARCH.md`](../RESEARCH.md) |
| Scientific rationale and measured results | [`../whitepaper/WHITEPAPER.md`](../whitepaper/WHITEPAPER.md) |
| Jev wire contract compatibility claims | [`../JEV_COMPATIBILITY.md`](../JEV_COMPATIBILITY.md) |
| Wire types, JSON Schema, OpenAPI, Protobuf | [`../../crates/openkind-core/AGENTS.md`](../../crates/openkind-core/AGENTS.md), [`../../crates/openkind-api/openapi.yaml`](../../crates/openkind-api/openapi.yaml), [`../../proto/proto/openkind.proto`](../../proto/proto/openkind.proto) |

Family pages **never** republish benchmark numbers, roadmap status, or
architecture contract text. They summarise each family, point at the
canonical owner for every quantitative or status claim, and note the open
questions that would need to be resolved before `openkind` could ship the
family.

## Family index

| Family | Backbone pattern | Status in openkind |
|---|---|---|
| [encoder-state-first](./encoder-state-first.md) | Encoder backbone (BERT-style), state-first segmented tokenization, score-summary readout over candidate suffixes | **Implemented** — provisional profile `a047d6802c3f06f085b8` over `Qwen/Qwen3.5-4B-Base`; see `../ROADMAP.md` and `../ARCHITECTURE.md` |
| [encoder-nli](./encoder-nli.md) | Encoder backbone, one premise–hypothesis forward pass per candidate, softmax over entailment probabilities | Surveyed — no Rust implementation, no parity fixtures |
| [encoder-instruct-label](./encoder-instruct-label.md) | Instruction-tuned encoder, all candidate label markers in one sequence, span pooling, sigmoid per label | Surveyed — no Rust implementation, no parity fixtures |
| [decoder-logit-letter](./decoder-logit-letter.md) | Decoder backbone, prompt with lettered options, next-token logits restricted to option-letter token ids | Surveyed — `decider` adapter design notes only |
| [decoder-logit-llm](./decoder-logit-llm.md) | Decoder backbone (any chat/instruct GGUF), prompt with lettered options, label-logit readout via llama.cpp | Surveyed — design notes only; mirrors the `llm-logits-v1` common core |
| [router-script](./router-script.md) | Lightweight language/script detector, no model forward pass; branches between sibling families | Surveyed — internal-only routing primitive |
| [winnow](./winnow.md) | Decoder backbone + LoRA, script- and language-aware router via label-logit readout | Surveyed — design notes only |
| [kev](./kev.md) | LoRA adapter + pointer head on a Qwen base, trained against the TypeSafe `kev` reference contract | Surveyed — contract mapping only |
| [von](./von.md) | Encoder head trained against TypeSafe's published `von` reference contract | Surveyed — contract mapping only |
| [schema-scorer](./schema-scorer.md) | DeBERTa-v3-large cross-encoder trained against the TypeSafe question schema, all three question types | Surveyed — contract mapping only |
| [qwen3guard](./qwen3guard.md) | Decoder fine-tune, fixed-preset safety verdict, embedded question schema | Surveyed — no Rust implementation, no parity fixtures |

The surveyed families are tracked as **evaluation backlog**. None has a
pinned profile, parity fixtures, or implementation commitment. They appear
here so that future selection work can pick up the survey with the same
shape of evidence that the implemented family carries.

## What "implemented" vs "surveyed" means

- **Implemented** — A profile is registered against this family in
  `openkind-engine`, parity fixtures are vendored in `crates/openkind-backends/`,
  the daemon registers the engine, and recorded numbers exist in
  `verification/`. Quantitative claims live in `../ROADMAP.md` and
  `../BENCHMARKS.md`.
- **Surveyed** — The architectural pattern is documented here and in the
  research record, but no profile, no vendored fixtures, no daemon
  registration, and no recorded numbers exist. Implementation would
  require a new profile ID, new parity fixtures, and a fresh review
  through M0–M2 of the active milestone sequence
  (see `../ROADMAP.md`).

## Adding a new family

1. Add a `<family>.md` page to this directory using the existing pages as a
   template.
2. Add a row to the family index table above with the correct status.
3. Do **not** add quantitative claims to the page. If you have measurements,
   they belong in `../BENCHMARKS.md` or a verification report under
   `../verification/`. Link to them.
4. If the family is being considered for implementation, link to the
   relevant `../ROADMAP.md` milestone rather than restating scope.
5. Do not duplicate the canonical-ownership rules — link to
   [`docs/AGENTS.md`](../AGENTS.md) instead.
