# Changelog

## Unreleased

### Added

- Model support for the d3 family: the pinned `d3-edge:664f9ead734372935286`
  profile serves `vllm-sr/d3-edge` at the evaluated v3.1.0 release commit
  (`18d61df78684675463af9ae732872478845c5a79`) — the pruned 12-layer
  Qwen3.5-0.8B hybrid with noncausal full attention and the
  `d3-code-readout-v1` 255-row float32 readout. Backbone changes: a causal
  flag on `Qwen35Geometry` with the noncausal branch pinned against a naive
  bidirectional reference, an explicit per-layer mixer table (the prune keeps
  an irregular `layer_types` subset), and a text-only tensor-namespace
  constructor. Prompt rendering is byte-verified against the shipped
  `d3_format.py` over the checkpoint's real chat template and tokenizer
  (all seven fixture cases match); golden-fixture replay, fail-closed load
  tests, daemon aliases, and bench engine (`openkind-bench score --engine
  d3-edge`) are wired.
- A second d3-family profile, `d3-lite:e0dd32dbc16b27546692`, serving
  `vllm-sr/d3-lite` at its v3.1.0 release commit
  (`a8d25a91ef75552803f7009d25b7bd435991abd9`) — the full 24-layer
  Qwen3.5-0.8B hybrid with the same noncausal code-readout contract,
  prompt byte-parity, golden replay, and catalog pull support (alias
  `d3:lite`).
- A third d3-family profile, `d3-nano:0198af5b5c8e8835605e`, serving
  `vllm-sr/d3-nano` at its v3.1.0 release commit
  (`3aeb1c22a5632aac72353c36b3d0111c40e7dbf4`) — the full 24-layer
  Qwen3.5-2B hybrid with the same noncausal code-readout contract, prompt
  byte-parity, golden replay, and catalog pull support (alias `d3:nano`).
- A fourth d3-family profile, `d3-mini:434dbc50d3b581663e52`, serving
  `vllm-sr/d3-mini` at its v3.1.0 release commit
  (`a475367de0d8133a3c1a2a0dc12303d89cfa2ab4`) — the full 32-layer
  Qwen3.5-4B hybrid (the pinned qwen3.5 dims) with the same noncausal
  code-readout contract; its chat template inverts the thinking
  conditional, which renders the same empty `<think>` block for the
  reference's calls. Prompt byte-parity, golden replay, and catalog pull
  support (alias `d3:mini`).
- A fifth d3-family profile, `d3-flash:b2a4750f8bf4a1fa58cb`, serving
  `vllm-sr/d3-flash` at its v3.1.0 release commit
  (`521525effa79469d899a570726547dd7f00da3af`) — the full 32-layer
  Qwen3.5-9B hybrid (the clef-flash dims) with the same noncausal
  code-readout contract, prompt byte-parity, golden replay, and catalog
  pull support (alias `d3:flash`). This completes every d3 sibling that
  fits the project's 36 GB reference host; the 27B d3 stays pinned as
  research metadata above that host.
- Model support for the sieve family: the pinned
  `sieve-4b:8b97b56524b55c41fbf4` profile serves `sthanika-ai/Sieve-4B`
  (rank-32 LoRA over `Qwen/Qwen3.5-4B`, causal, fenced-state branch
  prompts, per-type temperatures, two-order noul averaging) with the
  LoRA merge computed per load through the shared merged backend, prompt
  byte-parity against the shipped `sieve4b/prompt.py` (all seven fixture
  cases match), golden replay, fail-closed load tests, daemon aliases
  (`--sieve-aliases` / `--sieve-model-root`), and bench engine
  (`openkind-bench score --engine sieve-4b`) are wired.
- `registry/v1/jev-decision-index-0.3.1.json` (schema `openkind-jev-decision-index/v2`),
  superseding the 0.2.1 top-10 survey: 35 tier-1 and tier-2 entries covering the
  Decision Index 0.3.1 board leaders (d3, Decider 31B, Darwin-27B-ZTC, pplx v1.1), the
  whole six-model d3 family, and the remaining surveyed open entries, plus 11 pinned
  base models. Every revision was re-resolved against the Hub API on 2026-10-12 with
  file-level pin-versus-HEAD equivalence checks; all three board bundle files
  hash-match the pinned Space revision (`f6571b3`). Research metadata only —
  `installable: false` throughout, enforced by a new structural test
  (`jev_decision_index_031.rs`); the 0.2.1 file is retained as history.
- A decision record
  (`research/handoffs/OWNER_DECISION_remaining_entries_2026-10-12.md`)
  for the remaining 0.3.1 entries: the 27B-class checkpoints (above the
  36 GB reference host), NVFP4 weights, gemma4 MoE, torch-pickle heads,
  and CC BY-NC entries stay as pinned research metadata per their
  documented blockers, with the conditions that would reopen each group.

### Changed

- The native Qwen3.5 sequential nested runner advances each question and
  candidate from a borrowed parent state instead of deep-copying the parent
  first. `continue_from` already returns a new state that owns its tensors,
  so the copy only added a transient state-sized allocation per question and
  per candidate. `run_strategy` also moves candidate states out of the run
  rather than cloning them, and the full-attention KV cache is sized once
  instead of copied twice per layer. Results are unchanged: tests compare the
  runner to the fork-then-advance sequence and the layer to a one-pass
  forward, bit for bit (mutation-checked). Nested states now carry the root's
  lineage depth rather than a forked one, and nothing outside tests and one
  probe reads it. The structural fingerprint covers shape and position, not
  tensor values, so the runner's check never proved content immutability. For
  CPU states the shared borrow does. No timing is claimed, and the retained
  per-candidate states and their admission accounting are unchanged.

- `--qwen35-resident-weights` now also governs the Qwen3.5-text family
  engines (d3, sieve, decider, decoder-logit-qwen35, strands). Their
  backbone re-mapped the shards on every forward. For BF16-stored shards it
  also widened every decoder layer to FP32 each time, about 40% of busy CPU
  samples in the pinned backbone's profile. F32-stored shards map directly
  and gain nothing, and each family's stored dtype is unchecked. The
  decision is made once when the engine loads, with the same free-memory and
  process-ceiling guard. `auto` covers CPU shard-backed weights only, and
  adapter-backed engines need `on`. Outputs are bit-identical to the
  per-forward path (synthetic-checkpoint test over both mixers,
  mutation-checked). No timing is claimed. The speedup is unmeasured on this
  host and the checkpoint-gated parity replays were skipped.

- The d3 daemon flags take per-profile size entries —
  `--d3-aliases "edge=<alias>;lite=<alias>;nano=<alias>;mini=<alias>;flash=<alias>"`
  and `--d3-model-roots "edge=<path>;…"` — replacing the single-model
  form; the fixture generator selects the profile with `--profile` and the
  replay tests select their fixture with `OPENKIND_D3_FIXTURE`.
- The shared LoRA-merged backend (first proven on the strands-decider
  family) is parameterized over adapter rank and scale and supports
  multi-shard base checkpoints; the sieve family reuses it for its
  rank-32 merge, and the strands-decider load path is unchanged.
- Releases fetch the catalog from the mirror's immutable
  `registry/v1/catalogs/<CATALOG_SHA256>.json`, derived from the pinned
  digest, instead of the mutable `registry/v1/catalog.json`. Manifests
  still resolve under `registry/v1/`. The mirror keeps `catalog.json`
  frozen at the v0.2.1 bytes so that release keeps pulling, and
  `scripts/sync-model-registry.py` publishes each catalog under its digest.
- d3 enforces 65,536 prompt tokens per question and 262,144 per request,
  stricter than the reference's model context, and checks both before any
  forward. Full attention observes cancellation inside the kernel on the
  d3 path. Causal profiles run the unchanged kernel.
- sieve enforces the reference contract (32,768 tokens for state plus the
  longest branch, 65,536 per request, preamble excluded), validates every
  question before any forward, bills usage the way the reference does, and
  pins the adapter's `adapter_config.json` fields.
- d3 and sieve accept one-option Choice questions, and d3 rejects empty
  Choice keys and duplicate Score levels, matching their references.
- `jev-decision-index-0.3.1.json` marks the six landed d3 and sieve
  entries `openkind_loadable: true` with their catalog profile, and its
  shard counts no longer count a separate readout as a model shard.

### Fixed

- The shared LoRA-merged backend reads target shapes from the safetensors
  header instead of loading each base tensor onto the device, and no
  longer folds load failures (such as accelerator OOM) into a "missing
  tensor" contract error that hid them from backend fallback.
- d3 profile descriptions no longer contain a run of spaces from a lost
  line continuation.
- `docs/MODELS.md` download sizes now equal each manifest's total bytes,
  and profile counts read thirty-two.

## 0.2.1 (2026-10-07)

### Added

- Model support for Strands Decider 2B (Hobson v21, `StrandsAgents/strands-decider-2B-hobson-v21`
  at revision `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5`, profile ID `f7156bf28400a79ea1b8`)
  with fitted calibration temperatures (`strands_decider_config.json`) and multi-profile
  daemon loading.
- Leaderboard survey metadata in `registry/v1/jev-decision-index.json` indexing the top 10
  open reproductions on the Jev Decision Index (0.2.1).
- Local decision fine-tuning, post-training (RL/PPO/DPO), and release packaging notebooks
  and test validation tooling under `research/`.

### Changed

- Updated curated catalog to 26 models, including Hobson v19 and Hobson v21 profiles.
- Synced `whit3rabbit/openkind-model-registry` with the Hobson v21 profile and updated manifests.

## 0.2.0 (2026-10-04)

### Added

- `openkind batch`: Sequential, resumable JSONL evaluation jobs (`run`, `status`,
  `stop`, `resume`, `export`) with bounded streaming input, line range controls
  (`--start-row`, `--end-row`), a crash-resilient SQLite recovery journal, runner
  file locking, and graceful signal handling.
- `RequestLimits`: Independent failed-authentication rate limiting budget in
  `openkind-api` so unauthorized attempts do not exhaust valid callers' quotas.
- Output file reservation (`reserve_paths`, `reserve_outputs`) in `openkind-bench`
  with exclusive creation prior to model loading and evaluation, preventing
  accidental clobbering of prior benchmark evidence.
- Full prediction ID coverage and canonical score rubric key validation in
  `openkind-bench dataset eval`.
- Observability counters `openkind_request_outcomes_total` (fixed outcome labels)
  and `openkind_auth_failures_total` (fixed transport label), with fractional
  millisecond durations in `openkind_request_duration_ms`.
- Host hardware reports `available_parallelism` separately from OS `logical_cores`.
- `ClientBuilder::allow_unauthenticated()` in `openkind-client` for keyless daemons.
- Unattended Colab T4 NF4 FP16 pilot training notebook and generator with
  single-dimension parameter sweeps (`learning_rate`, `rank`).
- Verification workflow for published archive attestations, fresh Cargo
  installations, and Homebrew installations on macOS and Linux.

### Changed

- Minimum supported Rust version raised from 1.88 to 1.90.
- `openkind serve` and `openkindd` default HTTP and gRPC bind addresses to
  loopback (`127.0.0.1:8080` / `127.0.0.1:9090`) rather than `0.0.0.0`.
- Graceful shutdown handles a second `SIGINT` or `SIGTERM` to force an immediate
  nonzero exit while the first signal drains active in-flight requests.
- Proxy-cache verified key registry sweeps expired credentials on admission and
  caps in-memory entries at 4,096 keys.
- Proxy-cache sample store pruning bounds `co_deferred` diagnostic rows
  independently using the training budget, preserving IID calibration capacity.
- Model and dataset stores verify existing cached blobs before reuse, unlinking
  corrupt blobs for clean refetching while clearing abandoned `.stage-*`
  directories under lock.
- Explicit store directories must be nonempty; empty platform fallback variables
  are safely ignored.
- Macro-F1 in dataset evaluation averages only classes with gold support in the split.
- `ApiError` display formatting bounds error message strings to 200 characters
  with ellipsis while retaining the full raw message in programmatic fields.

### Fixed

- Score rubric index validation in `openkind-core` rejects non-canonical integer
  string keys (`+1`, `01`, `007`).
- Von decision Python repr rendering preserves the full 64-bit unsigned integer range.
- Client retry-after header parsing preserves 64-bit integer millisecond bounds
  without float precision loss and safely ignores overflowing values.
- gRPC overload responses include `retry-after` and `retry-after-ms` metadata keys.
- Backend response validation failures surface as `EngineError::BackendValidation`
  mapped to HTTP 500 / gRPC Internal.
- Documented OpenAPI error response schemas (`413`, `502`, `504`, `529`) are
  executable and tested on both `/v1/systemone` and `/v1/system_one`.

## 0.1.0 (2026-10-03)

### Added

- Typed Jev HTTP and gRPC interfaces for Noul, Choice, and Score decisions,
  with request-bound answer validation.
- `openkind` CLI and `openkindd` daemon, curated model installation, local
  playground controls, backend diagnostics, and a Rust client SDK.
- Native CPU execution, optional MLX and CUDA backends, optional ONNX execution,
  and Linux ROCm provider selection. Backend compilation and discovery have
  separate runtime and model qualification gates.
- Pinned evaluation-dataset installation and a native benchmark harness.
- TypeScript, Python, and Swift HTTP clients and daemon wrappers available from
  source. npm and PyPI packages remain unpublished.
- Native API-key generation in the CLI and binding wrappers.
- Standard and ONNX release archive automation, checksums, relocated archive
  checks, and Homebrew packaging.
- A release checklist and serial crates.io publisher with version validation,
  per-crate checksums, upload timing, and resumable reports.

### Changed

- Rust crates use one workspace version and the repository's MIT license.
  Every crate archive includes the license text.
- API-key configuration rejects empty, invalid, or conflicting values before
  model loading. Remove unused migration aliases from the environment rather
  than setting them to empty. See [ENV.md](docs/ENV.md).
- Release publication waits for CI and archive verification on the same commit.
  GitHub assets and Homebrew updates follow successful crate publication.

### Fixed

- macOS release command handling under Bash 3.2, Windows MSVC linker selection,
  and Linux CI/CUDA installation disk pressure.
- Windows dataset and model staging renames while metadata files remained open.
- Daemon startup stack overflow in Windows debug builds.
- Proxy-cache integration assertions that treated random upstream audits as failures.
- ROCm backend test compilation and missing optional-feature CI coverage.
- CUDA family loader error conversion and an unused backbone import.
- Repeated ONNX initialization after a missing-library failure. Repair the
  runtime path and restart the process before retrying.
- Binding daemon lifecycle, request snapshots, packaged license metadata,
  probability validation, and authentication forwarding.
