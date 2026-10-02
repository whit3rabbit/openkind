# Adding a model or family

Use this guide to move a surveyed architecture toward a real Rust loader and
runtime registration. The [family registry](README.md) is the support index;
[`../RESEARCH.md`](../RESEARCH.md) and [`../whitepaper/WHITEPAPER.md`](../whitepaper/WHITEPAPER.md)
own project priority and evidence gates.

## Name the thing being added

- A **family** is an inference pattern, such as encoder NLI or decoder
  letter-logit scoring. Give it a page when the architecture or wire mapping
  is materially different.
- A **model profile** pins a concrete checkpoint, tokenizer, readout,
  calibration, renderer, and execution contract. A second checkpoint in an
  existing family is a new profile, not automatically a new family.
- A **backend** is the execution implementation, such as the current Candle
  CPU path or optional MLX path.
- An **alias** is the caller-facing name registered to an initialized engine.
  `EngineRegistry` stores aliases; it does not discover model files or choose a
  loader from a model ID.

Keep these states separate:

| State | Meaning |
|---|---|
| Surveyed | Architecture notes exist; no Rust model loader is implied. |
| Prototype | Research code or experiments exist; not available through the Rust runtime. |
| Rust-loadable | A pinned profile loads local artifacts, has offline parity fixtures, implements `DecisionEngine`, and can be registered by the daemon. This does not establish useful decisions. |
| Task-qualified | The profile passes the registered M2 quality and policy gates for its declared workload. |
| Release-promoted | The locked profile passes M4 fresh confirmation and service requirements. |

The current native Rust-loadable profile is `encoder-state-first`. A second
profile must not reuse its identity or overwrite its golden fixtures.

## Before implementation

1. **Fit the supported task.** Wait for M0 to define the request population,
   wire semantics, language and option bounds, truncation behavior, and
   deployment envelope. Wait for M1 to repair and review the source-to-input
   record. Do not pick a family from exploratory scores alone.
2. **Own the model.** Pin the model and tokenizer revisions, read their
   licenses, identify who can redistribute weights and fixtures, and record
   the training and evaluation data rights. For a contract-mapping family,
   obtain the contract and implementation rights before reproducing it.
3. **Set an M2 question.** State what the family is expected to improve, which
   retained comparators it will face, and what result stops the work. Run one
   model hypothesis at a time. Parity is not a model-quality result.
4. **Choose the smallest valid surface.** Reuse a family when only the
   checkpoint changes. Add a new family when tokenization, forward graph,
   readout, state semantics, or supported question contract changes.

If a prerequisite fails, leave the entry surveyed or mark it blocked with the
reason. Do not create a loader to make the registry look complete.

Verify the current source before calling a checkpoint gated. A Hub 401/404
can indicate a moved repository or a serving recipe with no weight files.
Record the actual base checkpoint and serving configuration separately.
Environment-controlled readout settings need pinned values before they can
define a reproducible profile.

## Rust integration

The example in the [family registry](README.md#load-the-profile-from-rust)
shows the current `Qwen35DecisionEngine::load` and `EngineRegistry` APIs. Use
the same boundaries for a new implementation, while keeping family-specific
types and behavior in their own module.

### 1. Define and pin the profile

- Give every executable profile a new stable ID and an immutable source
  revision. Record the backbone, tokenizer, renderer, readout, calibration,
  probability space, and parity tolerances together.
- Use the profile-contract types in `openkind-engine` where they fit. The
  selected profile's validation path is in
  [`qwen35/profile.rs`](../../crates/openkind-backends/src/qwen35/profile.rs).
- Keep model, tokenizer, and fitted-head roots explicit in the loader config.
  Verify required files and digests before trusting them. Do not fetch assets
  from constructors, builds, tests, or CI. Do not copy multi-gigabyte shards
  into temporary directories; verify them in place.
- Make unsupported profile values fail closed. Do not silently change
  probability semantics, renormalize away a declared class, or substitute a
  different checkpoint.

### 2. Implement the backend and engine

- Put model-specific loading and forward execution under
  `crates/openkind-backends/src/families/<family>/` (the native Qwen 3.5
  profile sits directly under `src/qwen35/`). Keep hardware-neutral scheduling
  and state contracts in `openkind-runtime` and wire dispatch in
  `openkind-engine`.
- Implement a concrete loader config and a `DecisionEngine` adapter. Loading
  must use local pinned artifacts and return an error for missing, malformed,
  or mismatched artifacts. Do not add a `BackendType` enum variant as a
  substitute for an implementation; enum configuration alone does not load a
  model.
- Keep response construction in host Rust. Preserve `f64` probabilities and
  request-bound response validation. A model that uses semantic-none mass must
  define how it maps to the request's `__none__` option and reject unsupported
  probability spaces.
- If the model reuses a prefix, implement complete branch isolation. Qwen
  3.5 needs attention KV, DeltaNet recurrent state, and convolution state;
  KV-only copies are insufficient for that model. Advertise vectorized
  execution only when the backend actually performs vectorized forwards.
- Do not add autoregressive token generation. The surveyed `qwen3guard` page
  describes a possible family-specific exception, but it conflicts with the
  current architecture invariant and cannot be implemented without a separate
  approved architecture and wire-contract decision.

### 3. Register the engine in the daemon

- Add explicit configuration for the family to `openkind-server`, including
  artifact paths, backend choice, and resource limits. Keep it offline and
  fail closed when a native alias is requested without its configuration.
- Load the engine once, then register its `Arc<dyn DecisionEngine>` under the
  configured alias through `EngineRegistry`. Add the alias to the server's
  model list. The model name reported by `GET /v1/models` is this alias.
- Add CLI and environment-variable coverage in `openkind-cli` and
  `openkind-server`. A configured alias must not silently fall back to
  `MockEngine` when its native loader fails or is unavailable.
- Update the relevant crate `AGENTS.md`, the family page, and the runnable
  profile table in [`README.md`](README.md). The static family table does not
  replace the runtime alias registry.

### Benchmark integration

For each profile exposed through `openkind-bench`, update the complete
dispatch and provenance path:

- Add the CLI `EngineArg` in
  [`args.rs`](../../crates/openkind-bench/src/args.rs), its `EngineKind`
  conversion, and the loader dispatch in
  [`score/mod.rs`](../../crates/openkind-bench/src/score/mod.rs).
- Update `engine_slug`, `is_family_engine_public`, and the exhaustive
  `family_identity` mapping in
  [`score/types.rs`](../../crates/openkind-bench/src/score/types.rs).
  Include family variants in the rejected arms of both conditional
  `native_backend` implementations. A missing family identity can panic
  during summary construction after an expensive scoring run.
- Add offline summary regression coverage in
  [`score/summary.rs`](../../crates/openkind-bench/src/score/summary.rs).
  Assert the pinned profile and checkpoint revision without loading weights.
  Family runs retain the family slug in `engine`; `engine_variant` identifies
  the selected profile and backend.
- Compile optional MLX variants on macOS arm64 with `--features mlx` and
  `--all-targets`, following the
  [benchmark crate checks](../../crates/openkind-bench/AGENTS.md#verification-commands).
  Default workspace builds omit these variants. CPU and MLX variants of one
  profile share provenance and use distinct output slugs.

### 4. Qualify behavior independently

- Vendor small, digest-checked parity fixtures. Test artifact validation,
  tokenizer and renderer identity, readout math, probability keys, argmax,
  policy behavior, malformed requests, cancellation, and memory admission as
  applicable. Tests and builds must run without model downloads.
- Compare numerical parity to an identified reference implementation. Keep
  hidden-vector diagnostics separate from decision acceptance tolerances.
- Evaluate semantic quality on the M1-reviewed inputs with M2 comparators and
  all declared task-specific metrics. Report exclusions and licenses. Do not
  infer quality from successful loading or request-path timing.
- If cost is the reason to add the family, use M3's matched workload and
  retain the same useful operating point. Release status still requires M4.

If a wire type changes, follow the schema-generation and SDK synchronization
steps in the root [`AGENTS.md`](../../AGENTS.md). Benchmark taxonomy and
provenance belong in [`../BENCHMARKS.md`](../BENCHMARKS.md), not in the family
page.

## Plan for the surveyed families

The roadmap queues this work after the active M0-M4 evidence sequence. It
processes one family at a time and gives every current entry a gate. Most
entries have since landed Rust-loadable prototype loaders; the table records
the current loading status and the gate that still stands. The plan does not
waive license, wire, quality, or architecture requirements, and a
Rust-loadable prototype still carries no model-quality claim.

| Family | Loading status and remaining gate |
|---|---|
| [`encoder-nli`](encoder-nli.md) | Rust-loadable prototype (DistilBERT MNLI). Task qualification through the M2 gates remains separate. |
| [`encoder-instruct-label`](encoder-instruct-label.md) | Rust-loadable prototype (GLiClass label markers on ModernBERT). Task qualification remains separate. |
| [`decoder-logit-letter`](decoder-logit-letter.md) | Rust-loadable prototype (Qwen2.5-0.5B letter logits; no sampled tokens). Task qualification remains separate. |
| [`decoder-logit-llm`](decoder-logit-llm.md) | Rust-loadable prototype (candle GGUF q8_0 loader; no llama.cpp binding). Task qualification remains separate. |
| [`kev`](kev.md) | Rust-loadable prototype over the published open checkpoint (unblocked 2026-09-26). Task qualification remains separate. |
| [`von`](von.md) | Remains external-reference-only until contract, weights, training pipeline, and evaluation rights are available. |
| [`schema-scorer`](schema-scorer.md) | Rust-loadable prototype realized on open weights (MS MARCO MiniLM cross-encoder). Task qualification remains separate. |
| [`router-script`](router-script.md) | Rust-loadable composite; deterministic Unicode-script rules over registered sibling engines. |
| [`winnow`](winnow.md) | Rust-loadable learned router (in-house LoRA); sibling set locked to decoder-letter and encoder-nli. |
| [`qwen3guard`](qwen3guard.md) | Rust-loadable prototype (Stream variant, token-level head). A generated-verdict variant would still need a separate decision on the no-generation invariant, fixed-schema rejection, and parser failures. |
| [`clef`](clef.md) | Rust-loadable prototype (2026-10-01/02): Cloudflare Clef joint-schema models — BF16 CPU oracle, Q4_K_M GGUF flash and 27B, all reference-parity against the model card's `joint_schema_model.py`. The MLX 4-bit path is in tree but fails joint-head parity on quantized inputs; it stays non-loadable until the fixtures pass. The BF16 oracle's per-execution arithmetic (`cpu-bf16w-fp32c`) is a deliberate deviation from the strict-FP32 family convention, pinned by its own fixtures. Task qualification remains separate. |

The family is Rust-loadable only after the code and offline qualification
steps above land. Task support and release promotion are separate statuses.
The full sequence and ordering dependencies are documented in
[`../RESEARCH.md`](../RESEARCH.md) and [`../whitepaper/WHITEPAPER.md`](../whitepaper/WHITEPAPER.md).
